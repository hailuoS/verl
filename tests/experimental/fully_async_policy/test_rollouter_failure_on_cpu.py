"""Exercise fully-async task failure and cleanup without Ray or devices."""

import asyncio
import time
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from .test_external_manager_seams_on_cpu import source_method

ROOT = Path(__file__).resolve().parents[3]
ROLLOUTER = ROOT / "verl/experimental/fully_async_policy/fully_async_rollouter.py"


class TrackedQueue(asyncio.Queue):
    def __init__(self):
        super().__init__()
        self.waiters = asyncio.Queue()
        self.acquired = asyncio.Queue()
        self.task_done_calls = 0

    async def get(self):
        if self.empty():
            self.waiters.put_nowait(asyncio.current_task())
        item = await super().get()
        self.acquired.put_nowait(item)
        return item

    def task_done(self):
        self.task_done_calls += 1
        super().task_done()


def test_real_rollout_config_accepts_external_manager():
    from verl.workers.config.rollout import RolloutConfig

    config = RolloutConfig(name="vllm", llm_server_manager_class="some.module.Service")
    assert config.llm_server_manager_class == "some.module.Service"


def make_rollouter(method_names):
    tasks = {}

    def create_task(coro, name, task_set=None):
        task = asyncio.create_task(coro, name=name)
        tasks[name] = task
        if task_set is not None:
            task_set.add(task)
        return task

    namespace = {
        "asyncio": asyncio,
        "time": time,
        "safe_create_task": create_task,
        "np": NS(array=lambda values, dtype: values),
        "RolloutSample": NS,
    }
    methods = {
        name: source_method(ROLLOUTER, "FullyAsyncRollouter", name, namespace)
        for name in method_names
    }
    rollouter = type("RollouterHarness", (), methods)()
    rollouter.lock = asyncio.Lock()
    rollouter._resume_event = asyncio.Event()
    rollouter.pending_queue = asyncio.Queue()
    rollouter.active_tasks = set()
    rollouter.max_concurrent_samples = 2
    rollouter.staleness_samples = 0
    rollouter.message_queue_client = NS(put_sample=lambda **kwargs: asyncio.sleep(0))
    rollouter._record_active_count = lambda: None
    return rollouter, tasks


def test_completed_active_task_does_not_lose_sample_acquired_during_reap():
    async def run():
        rollouter, tasks = make_rollouter(("_reap_active_tasks", "_processor_worker"))
        queue = TrackedQueue()
        rollouter.pending_queue = queue
        rollouter.paused = False
        finished = asyncio.Event()
        reaping = asyncio.Event()
        resume_reap = asyncio.Event()
        processed = []
        active_task = asyncio.create_task(finished.wait())
        rollouter.active_tasks.add(active_task)
        original_reap = rollouter._reap_active_tasks

        async def reap(done_tasks):
            if not reaping.is_set():
                reaping.set()
                await resume_reap.wait()
            await original_reap(done_tasks)

        async def process(sample):
            processed.append(sample.sample_id)

        async def should_pause():
            return False

        rollouter._reap_active_tasks = reap
        rollouter._process_single_sample_streaming = process
        rollouter._should_pause_generation = should_pause
        processor = asyncio.create_task(rollouter._processor_worker())
        try:
            waiter = await asyncio.wait_for(queue.waiters.get(), 2)
            finished.set()
            await asyncio.wait_for(reaping.wait(), 2)
            sample = NS(sample_id="race")
            await queue.put(sample)
            assert await asyncio.wait_for(queue.acquired.get(), 2) is sample
            resume_reap.set()
            await queue.put(None)
            await asyncio.wait_for(processor, 2)
            await asyncio.wait_for(queue.join(), 2)

            assert processed == ["race"]
            assert queue.task_done_calls == 2  # sample and end signal
            assert queue._unfinished_tasks == 0
            assert rollouter.staleness_samples == 2
            assert active_task.done() and waiter.done()
            assert all(task.done() for task in tasks.values())
        finally:
            resume_reap.set()
            processor.cancel()
            active_task.cancel()
            await asyncio.gather(processor, active_task, return_exceptions=True)

    asyncio.run(run())


def test_active_failure_wins_over_sample_acquired_during_reap():
    methods = (
        "_reap_active_tasks",
        "_processor_worker",
        "_process_single_sample_streaming",
        "_streaming_generation_main",
        "fit",
    )

    async def run():
        rollouter, tasks = make_rollouter(methods)
        queue = TrackedQueue()
        rollouter.pending_queue = queue
        second_started = asyncio.Event()
        monitor_started = asyncio.Event()
        fail_first = asyncio.Event()
        reaping = asyncio.Event()
        resume_reap = asyncio.Event()
        cancelled = set()
        attempts = []
        original_reap = rollouter._reap_active_tasks

        class Batch:
            def __init__(self, tag):
                self.tag = tag
                self.non_tensor_batch = {}

            def __len__(self):
                return 1

        async def reap(done_tasks):
            if not reaping.is_set():
                reaping.set()
                await resume_reap.wait()
            await original_reap(done_tasks)

        async def generate(batch):
            attempts.append(batch.tag)
            if batch.tag == "first":
                await fail_first.wait()
                raise RuntimeError("receiver crashed during queue race")
            second_started.set()
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.add(batch.tag)

        async def feed():
            for tag in ("first", "second"):
                await queue.put(NS(sample_id=tag, full_batch=Batch(tag)))
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.add("feed")

        async def monitor():
            monitor_started.set()
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.add("monitor")

        async def should_pause():
            return False

        rollouter._reap_active_tasks = reap
        rollouter._feed_samples = feed
        rollouter._async_monitor_loop = monitor
        rollouter._should_pause_generation = should_pause
        rollouter.async_rollout_manager = NS(generate_sequences_single=generate)
        fit_task = asyncio.create_task(rollouter.fit())
        try:
            await asyncio.wait_for(second_started.wait(), 2)
            await asyncio.wait_for(monitor_started.wait(), 2)
            waiter = await asyncio.wait_for(queue.waiters.get(), 2)
            assert [
                (await asyncio.wait_for(queue.acquired.get(), 2)).sample_id for _ in range(2)
            ] == ["first", "second"]
            fail_first.set()
            await asyncio.wait_for(reaping.wait(), 2)
            new_sample = NS(sample_id="third", full_batch=Batch("third"))
            await queue.put(new_sample)
            assert await asyncio.wait_for(queue.acquired.get(), 2) is new_sample
            resume_reap.set()
            with pytest.raises(RuntimeError, match="receiver crashed during queue race"):
                await asyncio.wait_for(fit_task, 2)

            assert set(attempts) == {"first", "second"} and len(attempts) == 2
            assert cancelled == {"second", "feed", "monitor"}
            assert waiter.done() and waiter.result() is new_sample
            assert rollouter.active_tasks == set()
            assert all(task.done() for task in tasks.values())
        finally:
            resume_reap.set()
            fit_task.cancel()
            await asyncio.gather(fit_task, return_exceptions=True)

    asyncio.run(run())


def test_processor_cancellation_cleans_queue_waiter_and_active_tasks():
    methods = (
        "_reap_active_tasks",
        "_processor_worker",
        "_process_single_sample_streaming",
        "_streaming_generation_main",
        "fit",
    )

    async def run():
        rollouter, tasks = make_rollouter(methods)
        queue = TrackedQueue()
        rollouter.pending_queue = queue
        active_started = asyncio.Event()
        monitor_started = asyncio.Event()
        cancelled = set()
        attempts = []

        class Batch:
            def __init__(self):
                self.non_tensor_batch = {}

            def __len__(self):
                return 1

        async def generate(batch):
            attempts.append("active")
            active_started.set()
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.add("active")

        async def feed():
            await queue.put(NS(sample_id="active", full_batch=Batch()))
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.add("feed")

        async def monitor():
            monitor_started.set()
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.add("monitor")

        async def should_pause():
            return False

        rollouter._feed_samples = feed
        rollouter._async_monitor_loop = monitor
        rollouter._should_pause_generation = should_pause
        rollouter.async_rollout_manager = NS(generate_sequences_single=generate)
        fit_task = asyncio.create_task(rollouter.fit())
        try:
            await asyncio.wait_for(active_started.wait(), 2)
            await asyncio.wait_for(monitor_started.wait(), 2)
            waiter = await asyncio.wait_for(queue.waiters.get(), 2)
            fit_task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(fit_task, 2)

            assert attempts == ["active"]
            assert all(task.done() for task in tasks.values())
            assert cancelled == {"active", "feed", "monitor"}
            assert waiter.done() and waiter.cancelled()
            assert rollouter.active_tasks == set()
            assert all(task.done() for task in tasks.values())
        finally:
            fit_task.cancel()
            await asyncio.gather(fit_task, return_exceptions=True)

    asyncio.run(run())


def test_active_generation_failure_reaches_fit_and_awaits_whole_task_tree():
    methods = (
        "_reap_active_tasks",
        "_processor_worker",
        "_process_single_sample_streaming",
        "_streaming_generation_main",
        "_async_monitor_loop",
        "fit",
    )

    async def run():
        rollouter, tasks = make_rollouter(methods)
        second_started = asyncio.Event()
        cancelled = set()

        class Batch:
            def __init__(self):
                self.non_tensor_batch = {}

            def __len__(self):
                return 1

        async def generate(batch):
            if batch.non_tensor_batch["uid"] == ["uid_first"]:
                await second_started.wait()
                raise RuntimeError("receiver crashed")
            second_started.set()
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.add("second")

        async def feed():
            for sample_id in ("first", "second"):
                await rollouter.pending_queue.put(NS(sample_id=sample_id, full_batch=Batch()))
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.add("feed")

        async def should_pause():
            return False

        rollouter.async_rollout_manager = NS(generate_sequences_single=generate)
        rollouter._feed_samples = feed
        rollouter._should_pause_generation = should_pause

        with pytest.raises(RuntimeError, match="receiver crashed"):
            await asyncio.wait_for(rollouter.fit(), timeout=2)

        assert cancelled == {"feed", "second"}
        assert rollouter.active_tasks == set()
        assert {"first", "second", "feed_task", "processor_task", "generation_task", "monitor_task"} <= tasks.keys()
        assert all(task.done() for task in tasks.values())
        assert tasks["second"].cancelled()
        assert tasks["feed_task"].cancelled()
        assert tasks["monitor_task"].cancelled()
        assert isinstance(tasks["first"].exception(), RuntimeError)
        assert isinstance(tasks["processor_task"].exception(), RuntimeError)
        assert isinstance(tasks["generation_task"].exception(), RuntimeError)

    asyncio.run(run())


def test_monitor_failure_cancels_active_generation_tree():
    methods = (
        "_reap_active_tasks",
        "_processor_worker",
        "_process_single_sample_streaming",
        "_streaming_generation_main",
        "fit",
    )

    async def run():
        rollouter, tasks = make_rollouter(methods)
        active_started = asyncio.Event()
        cancelled = set()

        class Batch:
            def __init__(self):
                self.non_tensor_batch = {}

            def __len__(self):
                return 1

        async def generate(batch):
            active_started.set()
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.add("active")

        async def feed():
            await rollouter.pending_queue.put(NS(sample_id="active", full_batch=Batch()))
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.add("feed")

        async def monitor():
            await active_started.wait()
            raise RuntimeError("monitor failed")

        async def should_pause():
            return False

        rollouter.async_rollout_manager = NS(generate_sequences_single=generate)
        rollouter._feed_samples = feed
        rollouter._async_monitor_loop = monitor
        rollouter._should_pause_generation = should_pause

        with pytest.raises(RuntimeError, match="monitor failed"):
            await asyncio.wait_for(rollouter.fit(), timeout=2)

        assert cancelled == {"active", "feed"}
        assert rollouter.active_tasks == set()
        assert all(task.done() for task in tasks.values())
        assert tasks["active"].cancelled()
        assert tasks["feed_task"].cancelled()
        assert tasks["processor_task"].cancelled()
        assert tasks["generation_task"].cancelled()
        assert isinstance(tasks["monitor_task"].exception(), RuntimeError)

    asyncio.run(run())


@pytest.mark.parametrize("failing_task", ["generation", "monitor"])
def test_top_level_failure_cancels_and_awaits_sibling(failing_task):
    async def run():
        rollouter, tasks = make_rollouter(("fit",))
        sibling_started = asyncio.Event()
        cancelled = []

        async def fail():
            await sibling_started.wait()
            raise RuntimeError(f"{failing_task} failed")

        async def block():
            sibling_started.set()
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.append(True)

        rollouter._streaming_generation_main = fail if failing_task == "generation" else block
        rollouter._async_monitor_loop = fail if failing_task == "monitor" else block
        with pytest.raises(RuntimeError, match=f"{failing_task} failed"):
            await asyncio.wait_for(rollouter.fit(), timeout=2)

        sibling = "monitor_task" if failing_task == "generation" else "generation_task"
        assert cancelled == [True]
        assert all(task.done() for task in tasks.values())
        assert tasks[sibling].cancelled()

    asyncio.run(run())
