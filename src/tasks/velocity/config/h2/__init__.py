from mjlab.tasks.registry import register_mjlab_task
from src.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import (
  unitree_h2_flat_env_cfg,
  unitree_h2_rough_env_cfg,
)
from .rl_cfg import unitree_h2_ppo_runner_cfg

register_mjlab_task(
  task_id="Unitree-H2-Rough",
  env_cfg=unitree_h2_rough_env_cfg(),
  play_env_cfg=unitree_h2_rough_env_cfg(play=True),
  rl_cfg=unitree_h2_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-H2-Rough-H2ActionScale",
  env_cfg=unitree_h2_rough_env_cfg(use_h2_action_scale=True),
  play_env_cfg=unitree_h2_rough_env_cfg(play=True, use_h2_action_scale=True),
  rl_cfg=unitree_h2_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-H2-Flat",
  env_cfg=unitree_h2_flat_env_cfg(),
  play_env_cfg=unitree_h2_flat_env_cfg(play=True),
  rl_cfg=unitree_h2_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-H2-Flat-H2ActionScale",
  env_cfg=unitree_h2_flat_env_cfg(use_h2_action_scale=True),
  play_env_cfg=unitree_h2_flat_env_cfg(play=True, use_h2_action_scale=True),
  rl_cfg=unitree_h2_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)
