"""Exercise the two fully-async manager seams without Ray or device imports."""

import ast
import asyncio
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

ROOT = Path(__file__).resolve().parents[3]


class Config(dict):
    __getattr__ = dict.__getitem__


def source_method(path, class_name, method_name, namespace):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
    method = next(node for node in cls.body if getattr(node, "name", None) == method_name)
    method.decorator_list = []
    module = ast.Module(body=[method], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace[method_name]


@pytest.mark.parametrize("custom", [False, True])
def test_rollouter_selects_manager_and_uses_its_own_client(custom):
    events = []

    class NativeManager:
        @classmethod
        async def create(cls, **kwargs):
            events.append(("create", cls, kwargs))
            return cls()

        def get_client(self, **kwargs):
            events.append(("get_client", kwargs))
            return "native-client"

    class ExternalManager(NativeManager):
        def get_client(self, **kwargs):
            events.append(("get_client", kwargs))
            return "external-client"

    class AgentLoopManager:
        @staticmethod
        async def create(**kwargs):
            events.append(("agent_loop", kwargs))
            return object()

    def load(fqn, description):
        assert (fqn, description) == ("external.Service", "LLMServerManager")
        return ExternalManager

    method = source_method(
        ROOT / "verl/experimental/fully_async_policy/fully_async_rollouter.py",
        "FullyAsyncRollouter",
        "_init_async_rollout_manager",
        {
            "FullyAsyncLLMServerManager": NativeManager,
            "FullyAsyncAgentLoopManager": AgentLoopManager,
            "load_class_from_fqn": load,
        },
    )
    config = Config(
        actor_rollout_ref=Config(
            rollout=Config(mode="async", llm_server_manager_class="external.Service" if custom else None)
        )
    )
    worker_group = object()
    rollouter = NS(
        use_rm=False,
        config=config,
        reward_loop_manager=NS(reward_loop_workers="reward-workers"),
        teacher_model_manager=None,
        get_hybrid_worker_group=lambda: worker_group,
    )
    asyncio.run(method(rollouter))
    assert events[0] == (
        "create",
        ExternalManager if custom else NativeManager,
        {
            "config": config,
            "worker_group": worker_group,
        },
    )
    assert events[1] == ("get_client", {})
    assert events[2][0] == "agent_loop"
    assert events[2][1]["llm_client"] == ("external-client" if custom else "native-client")


@pytest.mark.parametrize("custom", [False, True])
def test_trainer_selects_checkpoint_manager_and_preserves_empty_replicas(custom):
    events = []

    class NativeManager:
        def __init__(self, **kwargs):
            events.append((type(self), kwargs))

    class ExternalManager(NativeManager):
        pass

    def load(fqn, description):
        assert (fqn, description) == ("external.Checkpoint", "CheckpointEngineManager")
        return ExternalManager

    method = source_method(
        ROOT / "verl/experimental/fully_async_policy/fully_async_trainer.py",
        "FullyAsyncTrainer",
        "_setup_checkpoint_manager",
        {
            "CheckpointEngineManager": NativeManager,
            "load_class_from_fqn": load,
            "omega_conf_to_dataclass": lambda value: value,
        },
    )

    async def get_replicas():
        return []

    checkpoint_config = NS(backend="elastic_hccl")
    actor_wg = object()
    trainer = NS(
        config=Config(
            actor_rollout_ref=Config(
                rollout=Config(
                    checkpoint_manager_class="external.Checkpoint" if custom else None,
                    checkpoint_engine=checkpoint_config,
                )
            )
        ),
        rollouter=NS(get_replicas=NS(remote=get_replicas)),
        actor_wg=actor_wg,
    )
    asyncio.run(method(trainer))
    assert events == [
        (
            ExternalManager if custom else NativeManager,
            {"config": checkpoint_config, "actor_wg": actor_wg, "replicas": []},
        )
    ]
    assert isinstance(trainer.checkpoint_manager, ExternalManager if custom else NativeManager)
