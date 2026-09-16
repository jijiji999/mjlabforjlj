"""LowBody velocity task."""

from mjlab.tasks.registry import register_mjlab_task
from src.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import lowbody_flat_env_cfg
from .rl_cfg import lowbody_ppo_runner_cfg

register_mjlab_task(
  task_id="lowbody",
  env_cfg=lowbody_flat_env_cfg(),
  play_env_cfg=lowbody_flat_env_cfg(play=True),
  rl_cfg=lowbody_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)
