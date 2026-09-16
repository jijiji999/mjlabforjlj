"""RL configuration for the Unitree G1 lower-body locomotion task."""

from mjlab.rl import RslRlOnPolicyRunnerCfg

from ..g1.rl_cfg import unitree_g1_ppo_runner_cfg


def unitree_g1_loco_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Reuse the G1 PPO settings with a separate experiment directory."""
  cfg = unitree_g1_ppo_runner_cfg()
  cfg.experiment_name = "g1_loco"
  cfg.max_iterations = 15001
  return cfg
