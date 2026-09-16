"""Unitree G1 flat locomotion with policy-controlled legs only."""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg

from src.assets.robots.unitree_g1.g1_constants import G1_ACTION_SCALE

from ..g1.env_cfgs import unitree_g1_flat_env_cfg
from . import mdp


LOWER_BODY_JOINT_PATTERN = (
  r"^(left|right)_(hip_(pitch|roll|yaw)|knee|ankle_(pitch|roll))_joint$"
)
UPPER_BODY_AND_WAIST_JOINT_PATTERN = (
  r"^(waist_.*|left_(shoulder_.*|elbow|wrist_.*)|"
  r"right_(shoulder_.*|elbow|wrist_.*))_joint$"
)

# Absolute target bounds in radians.  Values are deliberately small because
# this signal represents a separate manipulation controller moving near zero.
UPPER_BODY_MOTION_AMPLITUDE = {
  r"waist_.*_joint": 0.05,
  r".*_shoulder_.*_joint": 0.10,
  r".*_elbow_joint": 0.12,
  r".*_wrist_.*_joint": 0.08,
}
UPPER_BODY_MOTION_DURATION_RANGE_S = (1.5, 3.5)

HEIGHT_COMMAND_MIN_OFFSET_M = -0.15
HEIGHT_COMMAND_MAX_OFFSET_M = 0.0
HEIGHT_TRACKING_STD_M = 0.05

LOWER_BODY_ACTION_SCALE = {
  pattern: scale
  for pattern, scale in G1_ACTION_SCALE.items()
  if any(name in pattern for name in ("hip", "knee", "ankle"))
}


def unitree_g1_loco_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create the G1 flat task with a 12-DoF lower-body policy action."""
  cfg = unitree_g1_flat_env_cfg(play=play)

  robot_cfg = cfg.scene.entities["robot"]
  assert robot_cfg.init_state.joint_pos is not None
  # The standard G1 home pose bends the arms.  This task instead defines all
  # non-leg defaults at zero so resets, relative observations, and the random
  # manipulation-controller targets share the same center.
  robot_cfg.init_state.joint_pos = {
    pattern: value
    for pattern, value in robot_cfg.init_state.joint_pos.items()
    if not any(
      name in pattern for name in ("waist", "shoulder", "elbow", "wrist")
    )
  }

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.actuator_names = (LOWER_BODY_JOINT_PATTERN,)
  joint_pos_action.scale = LOWER_BODY_ACTION_SCALE

  default_height = robot_cfg.init_state.pos[2]
  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)
  cfg.commands["height"] = mdp.UniformHeightCommandCfg(
    entity_name="robot",
    resampling_time_range=twist_cmd.resampling_time_range,
    height_range=(
      default_height + HEIGHT_COMMAND_MIN_OFFSET_M,
      default_height + HEIGHT_COMMAND_MAX_OFFSET_M,
    ),
    default_height=default_height,
    moving_command_name="twist",
  )

  # Make the full-body proprioceptive input explicit.  Only last_action shrinks
  # to 12 dimensions; joint positions and velocities still contain all 29 DoF.
  # The existing 3-D velocity command becomes [vx, vy, wz, pelvis_height].
  for group_name in ("actor", "critic"):
    terms = cfg.observations[group_name].terms
    terms["command"].func = mdp.velocity_height_commands
    terms["command"].params = {
      "velocity_command_name": "twist",
      "height_command_name": "height",
    }
    terms["joint_pos"].params["asset_cfg"] = SceneEntityCfg(
      "robot", joint_names=(".*",)
    )
    terms["joint_vel"].params["asset_cfg"] = SceneEntityCfg(
      "robot", joint_names=(".*",)
    )

  cfg.events["upper_body_motion"] = EventTermCfg(
    func=mdp.SmoothRandomJointPositionTargets,
    mode="step",
    params={
      "asset_cfg": SceneEntityCfg(
        "robot", joint_names=(UPPER_BODY_AND_WAIST_JOINT_PATTERN,)
      ),
      "amplitude": UPPER_BODY_MOTION_AMPLITUDE,
      "duration_range_s": UPPER_BODY_MOTION_DURATION_RANGE_S,
    },
  )

  # Do not penalize motion commanded by the simulated manipulation controller.
  # Whole-body dynamics terms remain active so the leg policy still learns to
  # reject the upper-body disturbance.
  lower_body_cfg = SceneEntityCfg(
    "robot", joint_names=(LOWER_BODY_JOINT_PATTERN,)
  )
  pose_params = cfg.rewards["pose"].params
  pose_params["asset_cfg"] = lower_body_cfg
  for std_name in ("std_walking", "std_running"):
    pose_params[std_name] = {
      pattern: value
      for pattern, value in pose_params[std_name].items()
      if any(name in pattern for name in ("hip", "knee", "ankle"))
    }
  cfg.rewards["stand_still"].params["asset_cfg"] = SceneEntityCfg(
    "robot", joint_names=(LOWER_BODY_JOINT_PATTERN,)
  )
  cfg.rewards["joint_acc_l2"].params["asset_cfg"] = SceneEntityCfg(
    "robot", joint_names=(LOWER_BODY_JOINT_PATTERN,)
  )
  cfg.rewards["track_pelvis_height"] = RewardTermCfg(
    func=mdp.track_pelvis_height,
    weight=1.0,
    params={
      "command_name": "height",
      "std": HEIGHT_TRACKING_STD_M,
      "asset_cfg": SceneEntityCfg("robot"),
    },
  )

  return cfg
