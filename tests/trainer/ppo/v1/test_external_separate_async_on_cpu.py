"""V1 separate-async external rollout contract without Ray/NPU workers."""

from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from verl.trainer.ppo.utils import Role
from verl.trainer.ppo.v1 import get_trainer_cls
from verl.trainer.ppo.v1.trainer_base import PPOTrainer
from verl.trainer.ppo.v1.trainer_separate_async import HybridEngineMode, PPOTrainerSeparateAsync
from verl.workers.engine_workers import ActorRolloutRefWorker
from verl.workers.rollout.llm_server import FullyAsyncLLMServerClient


def _trainer(external: bool):
    trainer = PPOTrainerSeparateAsync.__new__(PPOTrainerSeparateAsync)
    trainer.config = OmegaConf.create(
        {
            "actor_rollout_ref": {
                "hybrid_engine": not external,
                "rollout": {
                    "llm_server_manager_class": "external.Service" if external else None,
                    "checkpoint_manager_class": "external.Checkpoint" if external else None,
                    "checkpoint_engine": {"backend": "elastic_hccl"},
                },
            }
        }
    )
    trainer.actor_rollout_wg = object()
    trainer.global_steps = 0
    trainer.timing_raw = {}
    return trainer


def test_v1_dispatch_selects_separate_async_trainer():
    assert get_trainer_cls("separate_async") is PPOTrainerSeparateAsync


@pytest.mark.parametrize(
    ("role", "hybrid_engine", "expect_rollout"),
    [
        ("actor_rollout", True, True),
        ("actor_rollout", False, False),
        ("actor_rollout_ref", True, True),
        ("actor_rollout_ref", False, False),
        ("rollout", False, True),
    ],
)
def test_worker_rollout_capability_respects_role_and_hybrid_engine(role, hybrid_engine, expect_rollout):
    config = MagicMock()
    config.get.side_effect = lambda key, default=None: hybrid_engine if key == "hybrid_engine" else default
    config.actor.strategy = "fsdp2"
    config.actor.get.return_value = {}
    config.ref.get.return_value = {}
    config.rollout.get.return_value = {}
    config.actor.use_dynamic_bsz = False
    config.rollout.log_prob_use_dynamic_bsz = False
    config.actor.ppo_micro_batch_size_per_gpu = 1
    config.rollout.log_prob_micro_batch_size_per_gpu = 1
    config.rollout.checkpoint_engine.backend = "elastic_hccl"
    config.rollout.checkpoint_engine.update_weights_bucket_megabytes = 1
    config.rollout.checkpoint_engine.engine_kwargs = {"elastic_hccl": {}}
    config.rollout.checkpoint_engine.custom_backend_module = None

    rollout_config = SimpleNamespace(
        tensor_model_parallel_size=1,
        data_parallel_size=1,
        pipeline_model_parallel_size=1,
        name="vllm",
        mode="async",
    )
    checkpoint_config = config.rollout.checkpoint_engine

    def convert(value, **_kwargs):
        return rollout_config if value is config.rollout else value if value is checkpoint_config else MagicMock()

    def init_worker(worker):
        worker._rank = 0
        worker._world_size = 1

    actor = MagicMock()
    actor.get_dispatch_collect.return_value = {}
    ref = MagicMock()
    ref.get_dispatch_collect.return_value = {}
    rollout_class = MagicMock()

    with (
        patch("verl.workers.engine_workers.Worker.__init__", init_worker),
        patch("verl.workers.engine_workers.DistProfilerExtension.__init__", return_value=None),
        patch("verl.workers.engine_workers.DistProfiler"),
        patch("verl.workers.engine_workers.omega_conf_to_dataclass", side_effect=convert),
        patch("verl.workers.engine_workers.TrainingWorkerConfig", return_value=MagicMock()),
        patch("verl.workers.engine_workers.open_dict", return_value=nullcontext()),
        patch.object(ActorRolloutRefWorker, "actor_worker_cls", return_value=actor) as actor_class,
        patch.object(ActorRolloutRefWorker, "ref_worker_cls", return_value=ref),
        patch.object(ActorRolloutRefWorker, "set_dispatch_collect"),
        patch("verl.workers.engine_workers.get_device_name", return_value="cpu"),
        patch("verl.workers.engine_workers.init_device_mesh"),
        patch("verl.workers.engine_workers.get_rollout_class", return_value=rollout_class) as get_rollout_class,
        patch("verl.workers.engine_workers.CheckpointEngineRegistry.new") as new_checkpoint_engine,
        patch("verl.workers.engine_workers.torch.distributed.get_rank", return_value=0),
        patch("verl.workers.engine_workers.import_external_libs"),
        patch("verl.workers.engine_workers.aggressive_empty_cache"),
    ):
        worker = ActorRolloutRefWorker(config=config, role=role)
        worker.init_model()

    assert worker._is_rollout is expect_rollout
    if "actor" in role:
        actor_class.assert_called_once()
        new_checkpoint_engine.assert_called_once()
    else:
        actor_class.assert_not_called()
        new_checkpoint_engine.assert_not_called()
    if expect_rollout:
        get_rollout_class.assert_called_once_with("vllm", "async")
        rollout_class.assert_called_once()
        assert worker.rollout is rollout_class.return_value
    else:
        get_rollout_class.assert_not_called()
        rollout_class.assert_not_called()
        assert worker.rollout is None


def test_real_hydra_external_config_passes_v1_constructor_checks():
    config_dir = Path(__file__).resolve().parents[4] / "verl" / "trainer" / "config"
    overrides = [
        "trainer.use_v1=true",
        "trainer.resume_mode=disable",
        "trainer.v1.trainer_mode=separate_async",
        "trainer.v1.separate_async.parameter_sync_step=1",
        "data.train_batch_size=1",
        "actor_rollout_ref.actor.ppo_mini_batch_size=1",
        "actor_rollout_ref.hybrid_engine=false",
        "actor_rollout_ref.rollout.nnodes=1",
        "actor_rollout_ref.rollout.n_gpus_per_node=1",
        "actor_rollout_ref.rollout.free_cache_engine=false",
        "actor_rollout_ref.rollout.llm_server_manager_class=elastic_rollout.integrations.verl.rollout_service.ElasticRolloutService",
        "+actor_rollout_ref.rollout.checkpoint_manager_class=elastic_rollout.integrations.verl.checkpoint_manager.ElasticCheckpointEngineManager",
        "actor_rollout_ref.rollout.checkpoint_engine.backend=elastic_hccl",
        "actor_rollout_ref.rollout.checkpoint_engine.custom_backend_module=elastic_rollout.integrations.verl.hccl_checkpoint_engine",
        "++actor_rollout_ref.rollout.checkpoint_engine.engine_kwargs.elastic_hccl.endpoint=http://CONTROL_HOST:PORT",
        "++actor_rollout_ref.rollout.checkpoint_engine.engine_kwargs.elastic_hccl.deployment_id=DEPLOYMENT",
        "++actor_rollout_ref.rollout.checkpoint_engine.engine_kwargs.elastic_hccl.bootstrap_model_path=MODEL_PATH",
        "++actor_rollout_ref.rollout.checkpoint_engine.engine_kwargs.elastic_hccl.packed=true",
        "++actor_rollout_ref.rollout.custom.elastic.endpoint=http://CONTROL_HOST:PORT",
        "++actor_rollout_ref.rollout.custom.elastic.deployment_id=DEPLOYMENT",
        "++actor_rollout_ref.rollout.custom.elastic.fixed_policy=false",
        "++actor_rollout_ref.rollout.custom.elastic.weight_sync=hccl",
        "++actor_rollout_ref.rollout.custom.elastic.request_timeout=300",
    ]
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        config = compose(config_name="ppo_trainer", overrides=overrides)

    rollout = config.actor_rollout_ref.rollout
    assert config.trainer.use_v1 is True
    assert config.trainer.resume_mode == "disable"
    assert config.trainer.v1.trainer_mode == "separate_async"
    assert config.actor_rollout_ref.hybrid_engine is False
    assert rollout.nnodes > 0
    assert rollout.n_gpus_per_node > 0
    assert rollout.free_cache_engine is False
    assert rollout.llm_server_manager_class.endswith("ElasticRolloutService")
    assert rollout.checkpoint_manager_class.endswith("ElasticCheckpointEngineManager")
    assert rollout.checkpoint_engine.backend == "elastic_hccl"
    assert rollout.checkpoint_engine.custom_backend_module.endswith("hccl_checkpoint_engine")
    assert rollout.checkpoint_engine.engine_kwargs.elastic_hccl.endpoint == rollout.custom.elastic.endpoint
    assert rollout.custom.elastic.deployment_id == rollout.checkpoint_engine.engine_kwargs.elastic_hccl.deployment_id
    assert rollout.checkpoint_engine.engine_kwargs.elastic_hccl.bootstrap_model_path == "MODEL_PATH"
    assert rollout.custom.elastic.fixed_policy is False
    assert rollout.custom.elastic.weight_sync == "hccl"
    assert config.data.train_batch_size == (
        config.trainer.v1.separate_async.parameter_sync_step * config.actor_rollout_ref.actor.ppo_mini_batch_size
    )

    # Keep the real V1 constructor checks; skip only base initialization that builds runtime state.
    with patch.object(PPOTrainer, "__init__", return_value=None):
        PPOTrainerSeparateAsync(config)
        bad_backend = OmegaConf.merge(
            config, {"actor_rollout_ref": {"rollout": {"checkpoint_engine": {"backend": "naive"}}}}
        )
        with pytest.raises(AssertionError, match="please use nccl/nixl/mooncake"):
            PPOTrainerSeparateAsync(bad_backend)
        bad_batch = OmegaConf.merge(config, {"data": {"train_batch_size": 2}})
        with pytest.raises(AssertionError, match="train_batch_size must equal"):
            PPOTrainerSeparateAsync(bad_batch)


def test_base_setup_does_not_launch_hybrid_servers_when_disabled():
    trainer = PPOTrainerSeparateAsync.__new__(PPOTrainerSeparateAsync)
    trainer.config = OmegaConf.create(
        {
            "global_profiler": {"steps": None},
            "trainer": {"device": "cpu"},
            "actor_rollout_ref": {"hybrid_engine": False, "model": {"lora_rank": 0}, "rollout": {}},
            "reward": {"reward_model": {"enable": False}},
        }
    )
    trainer.use_critic = False
    trainer.use_reference_policy = False
    trainer.use_teacher_policy = False
    trainer.role_worker_mapping = {Role.ActorRolloutRef: object()}
    pool = object()
    trainer.resource_pool_manager = MagicMock()
    trainer.resource_pool_manager.resource_pool_dict = {"actor": pool}
    trainer.resource_pool_manager.get_resource_pool.return_value = pool
    worker = MagicMock()
    worker_group = MagicMock()
    worker_group.spawn.return_value = {str(Role.ActorRolloutRef): worker}

    with (
        patch.object(trainer, "_init_tokenizer"),
        patch.object(trainer, "_init_dataloader"),
        patch.object(trainer, "_init_dump_executor"),
        patch.object(trainer, "_init_resource_pool_mgr"),
        patch.object(trainer, "_load_checkpoint"),
        patch("verl.trainer.ppo.v1.trainer_base.RayClassWithInitArgs"),
        patch("verl.trainer.ppo.v1.trainer_base.create_colocated_worker_cls"),
        patch("verl.trainer.ppo.v1.trainer_base.RayWorkerGroup", return_value=worker_group),
        patch("verl.trainer.ppo.v1.trainer_base.RewardLoopManager"),
        patch("verl.trainer.ppo.v1.trainer_base.LLMServerManager.create") as native_server,
        patch("verl.trainer.ppo.v1.trainer_base.CheckpointEngineManager") as native_checkpoint,
    ):
        PPOTrainer._setup(trainer)

    worker.init_model.assert_called_once_with()
    native_server.assert_not_called()
    native_checkpoint.assert_not_called()
    assert trainer.llm_server_manager is None
    assert trainer.checkpoint_manager is None


def test_external_v1_uses_only_external_managers_and_one_client():
    trainer = _trainer(external=True)
    service = MagicMock()
    service.get_replicas.return_value = []
    service.get_client.return_value = object()
    service_class = MagicMock()
    service_class.create.return_value = service
    checkpoint_class = MagicMock()

    def base_setup(self):
        self.llm_server_manager = None
        self.checkpoint_manager = None

    with (
        patch.object(PPOTrainer, "_setup", base_setup),
        patch("verl.trainer.ppo.v1.trainer_separate_async.omega_conf_to_dataclass", side_effect=lambda value: value),
        patch(
            "verl.trainer.ppo.v1.trainer_separate_async.load_class_from_fqn",
            side_effect=[service_class, checkpoint_class],
        ),
        patch.object(trainer, "add_replicas_to_balancer") as add_hybrid,
        patch("verl.trainer.ppo.v1.trainer_separate_async.marked_timer", return_value=nullcontext()),
    ):
        trainer._setup()
        assert trainer.current_mode is HybridEngineMode.TRAINER
        add_hybrid.assert_not_called()
        service_class.create.assert_called_once_with(config=trainer.config, start_rank=0)
        checkpoint_class.assert_called_once_with(
            config=trainer.config.actor_rollout_ref.rollout.checkpoint_engine,
            actor_wg=trainer.actor_rollout_wg,
            replicas=[],
        )
        assert trainer.get_llm_client() is service.get_client.return_value
        service.get_client.assert_called_once_with()
        trainer.on_init_end()
        checkpoint_class.return_value.update_weights.assert_called_with(0)
        trainer.global_steps = 1
        trainer.on_step_end()
        checkpoint_class.return_value.update_weights.assert_called_with(1)
        trainer.on_sample_begin()
        trainer.on_sample_end()
        add_hybrid.assert_not_called()


def test_native_v1_keeps_fully_async_client_and_hybrid_lifecycle():
    trainer = _trainer(external=False)
    hybrid = MagicMock()
    hybrid.rollout_replicas = [object()]
    standalone = MagicMock()
    standalone.get_replicas.return_value = [object()]

    def base_setup(self):
        self.llm_server_manager = hybrid
        self.checkpoint_manager = MagicMock()

    with (
        patch.object(PPOTrainer, "_setup", base_setup),
        patch("verl.trainer.ppo.v1.trainer_separate_async.omega_conf_to_dataclass", side_effect=lambda value: value),
        patch("verl.trainer.ppo.v1.trainer_separate_async.LLMServerManager.create", return_value=standalone),
        patch("verl.trainer.ppo.v1.trainer_separate_async.CheckpointEngineManager") as checkpoint_class,
        patch.object(trainer, "add_replicas_to_balancer") as add_hybrid,
    ):
        trainer._setup()
        assert trainer.current_mode is HybridEngineMode.ROLLOUT
        add_hybrid.assert_called_once_with()
        trainer.get_llm_client()
        standalone.get_client.assert_called_once_with(client_cls=FullyAsyncLLMServerClient)
        checkpoint_class.assert_called_once()
