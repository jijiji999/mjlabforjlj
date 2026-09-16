"""JLJ A00 whole-body flat velocity environment configuration."""

import math

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.command_manager import CommandTermCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig
from src.assets.robots.jljwhole.jljwhole_constants import (
  JLJWHOLE_ACTION_SCALE,
  JLJWHOLE_BASE_BODY,
  JLJWHOLE_FOOT_COLLISION_NAMES,
  JLJWHOLE_FOOT_SITE_NAMES,
  get_jljwholebody_robot_cfg,
)

from . import mdp
from .mdp import UniformVelocityCommandCfg

_BASE_BODY_NAME = JLJWHOLE_BASE_BODY
_TORSO_BODY_NAME = "waist_pitch"
_FOOT_SITE_NAMES = JLJWHOLE_FOOT_SITE_NAMES
_FOOT_SUBTREE_PATTERN = r"^(left_ankle_roll|right_ankle_roll)$"
JLJWHOLE_WAIST_JOINT_NAMES = ("waist_yaw", "waist_roll", "waist_pitch")
JLJWHOLE_WAIST_JOINT_POS_L2_WEIGHT = -1.0
JLJWHOLE_TORSO_ANG_VEL_WEIGHT = -0.05
JLJWHOLE_ARM_JOINT_NAMES = (
  "left_shoulder_pitch",
  "left_shoulder_roll",
  "left_shoulder_yaw",
  "left_elbow",
  "left_wrist_yaw",
  "left_wrist_roll",
  "left_wrist_pitch",
  "right_shoulder_pitch",
  "right_shoulder_roll",
  "right_shoulder_yaw",
  "right_elbow",
  "right_wrist_yaw",
  "right_wrist_roll",
  "right_wrist_pitch",
)
JLJWHOLE_ARM_JOINT_POS_L2_WEIGHT = -0.3

JLJWHOLE_ENABLE_MASS_RANDOMIZATION = True
JLJWHOLE_MASS_SCALE_RANGE = (0.95,1.1)

JLJWHOLE_ENABLE_CONTROL_DELAY_RANDOMIZATION = True
JLJWHOLE_CONTROL_DELAY_LAG_RANGE = (0, 3)

JLJWHOLE_ENABLE_FOOT_HARDNESS_RANDOMIZATION = True
JLJWHOLE_FOOT_SOLREF_TIMECONST_RANGE = (0.004, 0.02)

JLJWHOLE_ENABLE_JOINT_FRICTION_RANDOMIZATION = True
JLJWHOLE_JOINT_FRICTION_RANGE = (0.0, 0.6)


def _body_mass_scale_range_to_alpha_range(
  body_mass_scale_range: tuple[float, float],
) -> tuple[float, float]:
  """Convert direct mass scale bounds to ``dr.pseudo_inertia`` alpha bounds."""
  min_scale, max_scale = body_mass_scale_range
  if min_scale <= 0.0 or max_scale <= 0.0:
    raise ValueError("body mass scale range values must be positive.")
  if min_scale > max_scale:
    raise ValueError("body mass scale range min must be <= max.")

  return (0.5 * math.log(min_scale), 0.5 * math.log(max_scale))


def _validate_randomization_ranges() -> None:
  delay_min_lag, delay_max_lag = JLJWHOLE_CONTROL_DELAY_LAG_RANGE
  if delay_min_lag < 0 or delay_max_lag < 0:
    raise ValueError("control delay lag range values must be non-negative.")
  if delay_min_lag > delay_max_lag:
    raise ValueError("control delay lag range min must be <= max.")

  hardness_min, hardness_max = JLJWHOLE_FOOT_SOLREF_TIMECONST_RANGE
  if hardness_min <= 0.0 or hardness_max <= 0.0:
    raise ValueError("foot solref time constant range values must be positive.")
  if hardness_min > hardness_max:
    raise ValueError("foot solref time constant range min must be <= max.")

  friction_min, friction_max = JLJWHOLE_JOINT_FRICTION_RANGE
  if friction_min < 0.0 or friction_max < 0.0:
    raise ValueError("joint friction range values must be non-negative.")
  if friction_min > friction_max:
    raise ValueError("joint friction range min must be <= max.")


def jljwholebody_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create JLJWHOLE flat-terrain velocity configuration."""
  _validate_randomization_ranges()

  actor_terms = {
    "base_ang_vel": ObservationTermCfg(
      func=mdp.builtin_sensor,
      params={"sensor_name": "robot/imu_ang_vel"},
      noise=Unoise(n_min=-0.2, n_max=0.2),
    ),
    "projected_gravity": ObservationTermCfg(
      func=mdp.projected_gravity,
      noise=Unoise(n_min=-0.08, n_max=0.08),
    ),
    "command": ObservationTermCfg(
      func=mdp.generated_commands,
      params={"command_name": "twist"},
    ),
    "phase": ObservationTermCfg(
      func=mdp.phase,
      params={"period": 0.6, "command_name": "twist"},
    ),
    "joint_pos": ObservationTermCfg(
      func=mdp.joint_pos_rel,
      noise=Unoise(n_min=-0.02, n_max=0.02),
    ),
    "joint_vel": ObservationTermCfg(
      func=mdp.joint_vel_rel,
      noise=Unoise(n_min=-0.7, n_max=0.7),
    ),
    "actions": ObservationTermCfg(func=mdp.last_action),
  }

  critic_terms = {
    **actor_terms,
    "base_lin_vel": ObservationTermCfg(
      func=mdp.builtin_sensor,
      params={"sensor_name": "robot/imu_lin_vel"},
      noise=Unoise(n_min=-0.5, n_max=0.5),
    ),
    "foot_height": ObservationTermCfg(
      func=mdp.foot_height,
      params={"asset_cfg": SceneEntityCfg("robot", site_names=_FOOT_SITE_NAMES)},
    ),
    "foot_air_time": ObservationTermCfg(
      func=mdp.foot_air_time,
      params={"sensor_name": "feet_ground_contact"},
    ),
    "foot_contact": ObservationTermCfg(
      func=mdp.foot_contact,
      params={"sensor_name": "feet_ground_contact"},
    ),
    "foot_contact_forces": ObservationTermCfg(
      func=mdp.foot_contact_forces,
      params={"sensor_name": "feet_ground_contact"},
    ),
  }

  observations = {
    "actor": ObservationGroupCfg(
      terms=actor_terms,
      concatenate_terms=True,
      enable_corruption=True,
      history_length=5,
    ),
    "critic": ObservationGroupCfg(
      terms=critic_terms,
      concatenate_terms=True,
      enable_corruption=False,
      history_length=5,
    ),
  }

  actions: dict[str, ActionTermCfg] = {
    "joint_pos": JointPositionActionCfg(
      entity_name="robot",
      actuator_names=(".*",),
      scale=JLJWHOLE_ACTION_SCALE,
      use_default_offset=True,
    )
  }

  commands: dict[str, CommandTermCfg] = {
    "twist": UniformVelocityCommandCfg(
      entity_name="robot",
      resampling_time_range=(3.0, 8.0),
      rel_standing_envs=0.05,
      heading_command=True,
      heading_control_stiffness=0.5,
      debug_vis=True,
      ranges=UniformVelocityCommandCfg.Ranges(
        lin_vel_x=(-1.0, 2.0),
        lin_vel_y=(-1.0, 1.0),
        ang_vel_z=(-1.0, 1.0),
        heading=(-math.pi, math.pi),
      ),
    )
  }

  feet_ground_cfg = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(
      mode="subtree",
      pattern=_FOOT_SUBTREE_PATTERN,
      entity="robot",
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
  )
  self_collision_cfg = ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(mode="subtree", pattern=_BASE_BODY_NAME, entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern=_BASE_BODY_NAME, entity="robot"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )

  events = {
    "reset_base": EventTermCfg(
      func=mdp.reset_root_state_uniform,
      mode="reset",
      params={
        "pose_range": {
          "x": (-0.5, 0.5),
          "y": (-0.5, 0.5),
          "z": (0.0, 0.0),
          "yaw": (-3.14, 3.14),
        },
        "velocity_range": {},
      },
    ),
    "reset_robot_joints": EventTermCfg(
      func=mdp.reset_joints_by_offset,
      mode="reset",
      params={
        "position_range": (-0.0, 0.0),
        "velocity_range": (-0.0, 0.0),
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
      },
    ),
    "push_robot": EventTermCfg(
      func=mdp.push_by_setting_velocity,
      mode="interval",
      interval_range_s=(5.0, 6.0),
      params={
        "velocity_range": {
          "x": (-0.5, 0.7),
          "y": (-0.5, 0.5),
          "z": (-0.4, 0.4),
          "roll": (-0.52, 0.52),
          "pitch": (-0.52, 0.52),
          "yaw": (-0.78, 0.78),
        },
      },
    ),
    "foot_friction": EventTermCfg(
      mode="startup",
      func=dr.geom_friction,
      params={
        "asset_cfg": SceneEntityCfg(
          "robot", geom_names=JLJWHOLE_FOOT_COLLISION_NAMES
        ),
        "operation": "abs",
        "ranges": (0.1, 1.6),
        "shared_random": True,
      },
    ),
    "encoder_bias": EventTermCfg(
      mode="startup",
      func=dr.encoder_bias,
      params={
        "asset_cfg": SceneEntityCfg("robot"),
        "bias_range": (-0.015, 0.015),
      },
    ),
    "base_com": EventTermCfg(
      mode="startup",
      func=dr.body_com_offset,
      params={
        "asset_cfg": SceneEntityCfg("robot", body_names=(_BASE_BODY_NAME,)),
        "operation": "add",
        "ranges": {
          0: (-0.05, 0.05),
          1: (-0.05, 0.05),
          2: (-0.05, 0.05),
        },
      },
    ),
  }

  if JLJWHOLE_ENABLE_MASS_RANDOMIZATION and not play:
    events["body_mass"] = EventTermCfg(
      mode="startup",
      func=dr.pseudo_inertia,
      params={
        "asset_cfg": SceneEntityCfg("robot", body_names=(".*",)),
        "alpha_range": _body_mass_scale_range_to_alpha_range(
          JLJWHOLE_MASS_SCALE_RANGE
        ),
      },
    )

  if JLJWHOLE_ENABLE_FOOT_HARDNESS_RANDOMIZATION and not play:
    events["foot_hardness"] = EventTermCfg(
      mode="startup",
      func=mdp.foot_contact_solref_timeconst,
      params={
        "asset_cfg": SceneEntityCfg(
          "robot", geom_names=JLJWHOLE_FOOT_COLLISION_NAMES
        ),
        "ranges": JLJWHOLE_FOOT_SOLREF_TIMECONST_RANGE,
        "shared_random": True,
      },
    )

  if JLJWHOLE_ENABLE_JOINT_FRICTION_RANDOMIZATION and not play:
    events["joint_friction"] = EventTermCfg(
      mode="startup",
      func=dr.joint_friction,
      params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
        "operation": "abs",
        "ranges": JLJWHOLE_JOINT_FRICTION_RANGE,
        "shared_random": False,
      },
    )

  rewards = {
    "track_linear_velocity": RewardTermCfg(
      func=mdp.track_linear_velocity,
      weight=1.0,
      params={"command_name": "twist", "std": math.sqrt(0.25)},
    ),
    "track_angular_velocity": RewardTermCfg(
      func=mdp.track_angular_velocity,
      weight=1.0,
      params={"command_name": "twist", "std": math.sqrt(0.5)},
    ),
    "body_orientation_l2": RewardTermCfg(
      func=mdp.body_orientation_l2,
      weight=-1.0,
      params={
        "asset_cfg": SceneEntityCfg("robot", body_names=(_BASE_BODY_NAME,))
      },
    ),
    "pose": RewardTermCfg(
      func=mdp.variable_posture,
      weight=1.0,
      params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
        "command_name": "twist",
        "std_standing": {".*": 0.05},
        "std_walking": {
          r".*hip_pitch.*": 0.5,
          r".*hip_roll.*": 0.15,
          r".*hip_yaw.*": 0.15,
          r".*knee.*": 0.5,
          r".*ankle_roll.*": 0.1,
          r".*ankle_pitch.*": 0.15,
          r".*waist_yaw.*": 0.15,
          r".*waist_roll.*": 0.1,
          r".*waist_pitch.*": 0.1,
          r".*shoulder_pitch.*": 0.15,
          r".*shoulder_roll.*": 0.1,
          r".*shoulder_yaw.*": 0.1,
          r".*elbow.*": 0.1,
          r".*wrist.*": 0.1,
        },
        "std_running": {
          r".*hip_pitch.*": 0.5,
          r".*hip_roll.*": 0.25,
          r".*hip_yaw.*": 0.25,
          r".*knee.*": 0.5,
          r".*ankle_roll.*": 0.1,
          r".*ankle_pitch.*": 0.25,
          r".*waist_yaw.*": 0.25,
          r".*waist_roll.*": 0.1,
          r".*waist_pitch.*": 0.1,
          r".*shoulder_pitch.*": 0.25,
          r".*shoulder_roll.*": 0.1,
          r".*shoulder_yaw.*": 0.1,
          r".*elbow.*": 0.1,
          r".*wrist.*": 0.1,
        },
        "walking_threshold": 0.1,
        "running_threshold": 1.5,
      },
    ),
    "body_ang_vel": RewardTermCfg(
      func=mdp.body_angular_velocity_penalty,
      weight=-0.05,
      params={
        "asset_cfg": SceneEntityCfg("robot", body_names=(_BASE_BODY_NAME,))
      },
    ),
    "torso_ang_vel": RewardTermCfg(
      func=mdp.body_angular_velocity_penalty,
      weight=JLJWHOLE_TORSO_ANG_VEL_WEIGHT,
      params={
        "asset_cfg": SceneEntityCfg("robot", body_names=(_TORSO_BODY_NAME,))
      },
    ),
    "waist_joint_pos_l2": RewardTermCfg(
      func=mdp.joint_pos_deviation_l2,
      weight=JLJWHOLE_WAIST_JOINT_POS_L2_WEIGHT,
      params={
        "asset_cfg": SceneEntityCfg(
          "robot", joint_names=JLJWHOLE_WAIST_JOINT_NAMES
        )
      },
    ),
    "arm_joint_pos_l2": RewardTermCfg(
      func=mdp.joint_pos_deviation_l2,
      weight=JLJWHOLE_ARM_JOINT_POS_L2_WEIGHT,
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=JLJWHOLE_ARM_JOINT_NAMES)},
    ),
    "angular_momentum": RewardTermCfg(
      func=mdp.angular_momentum_penalty,
      weight=-0.025,
      params={"sensor_name": "robot/root_angmom"},
    ),
    "is_terminated": RewardTermCfg(func=mdp.is_terminated, weight=-200.0),
    "joint_acc_l2": RewardTermCfg(func=mdp.joint_acc_l2, weight=-2.5e-7),
    "joint_pos_limits": RewardTermCfg(func=mdp.joint_pos_limits, weight=-10.0),
    "action_rate_l2": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.05),
    "foot_gait": RewardTermCfg(
      func=mdp.feet_gait,
      weight=0.5,
      params={
        "period": 0.6,
        "offset": [0.0, 0.5],
        "threshold": 0.56,
        "command_threshold": 0.1,
        "command_name": "twist",
        "sensor_name": "feet_ground_contact",
      },
    ),
    "foot_clearance": RewardTermCfg(
      func=mdp.feet_clearance,
      weight=-1.0,
      params={
        "target_height": 0.10,
        "command_name": "twist",
        "command_threshold": 0.1,
        "asset_cfg": SceneEntityCfg("robot", site_names=_FOOT_SITE_NAMES),
      },
    ),
    "foot_slip": RewardTermCfg(
      func=mdp.feet_slip,
      weight=-0.25,
      params={
        "sensor_name": "feet_ground_contact",
        "command_name": "twist",
        "command_threshold": 0.1,
        "asset_cfg": SceneEntityCfg("robot", site_names=_FOOT_SITE_NAMES),
      },
    ),
    "soft_landing": RewardTermCfg(
      func=mdp.soft_landing,
      weight=-0.8e-3,
      params={
        "sensor_name": "feet_ground_contact",
        "command_name": "twist",
        "command_threshold": 0.1,
      },
    ),
    "stand_still": RewardTermCfg(
      func=mdp.stand_still,
      weight=-1.0,
      params={
        "command_name": "twist",
        "command_threshold": 0.1,
        "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
      },
    ),
    "self_collisions": RewardTermCfg(
      func=mdp.self_collision_cost,
      weight=-1.0,
      params={"sensor_name": self_collision_cfg.name, "force_threshold": 10.0},
    ),
  }

  terminations = {
    "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
    "fell_over": TerminationTermCfg(
      func=mdp.bad_orientation,
      params={"limit_angle": math.radians(70.0)},
    ),
  }

  curriculum = {
    "command_vel": CurriculumTermCfg(
      func=mdp.commands_vel,
      params={
        "command_name": "twist",
        "velocity_stages": [
          {
            "step": 0,
            "lin_vel_x": (-0.5, 1.0),
            "lin_vel_y": (-0.5, 0.5),
            "ang_vel_z": (-1.0, 1.0),
          },
          {
            "step": 5000 * 24,
            "lin_vel_x": (-1.0, 2.0),
            "lin_vel_y": (-1.0, 1.0),
            "ang_vel_z": None,
          },
        ],
      },
    ),
  }

  metrics = {
    "mean_action_acc": MetricsTermCfg(func=mdp.mean_action_acc),
  }

  cfg = ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      terrain=TerrainEntityCfg(
        terrain_type="plane",
        terrain_generator=None,
        max_init_terrain_level=5,
      ),
      sensors=(feet_ground_cfg, self_collision_cfg),
      entities={
        "robot": get_jljwholebody_robot_cfg(
          enable_control_delay=(
            JLJWHOLE_ENABLE_CONTROL_DELAY_RANDOMIZATION and not play
          ),
          control_delay_min_lag=JLJWHOLE_CONTROL_DELAY_LAG_RANGE[0],
          control_delay_max_lag=JLJWHOLE_CONTROL_DELAY_LAG_RANGE[1],
        )
      },
      num_envs=1,
      extent=2.0,
    ),
    observations=observations,
    actions=actions,
    commands=commands,
    events=events,
    rewards=rewards,
    terminations=terminations,
    curriculum=curriculum,
    metrics=metrics,
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.ASSET_BODY,
      entity_name="robot",
      body_name=_BASE_BODY_NAME,
      distance=3.0,
      elevation=-5.0,
      azimuth=90.0,
    ),
    sim=SimulationCfg(
      nconmax=None,
      njmax=300,
      contact_sensor_maxmatch=64,
      mujoco=MujocoCfg(
        timestep=0.005,
        iterations=10,
        ls_iterations=20,
        ccd_iterations=50,
      ),
    ),
    decimation=4,
    episode_length_s=20.0,
  )

  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)
  twist_cmd.viz.z_offset = 1.15

  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = True
    cfg.events.pop("push_robot", None)
    cfg.curriculum = {}
    cfg.events["randomize_terrain"] = EventTermCfg(
      func=mdp.randomize_terrain,
      mode="reset",
      params={},
    )
    twist_cmd.ranges.lin_vel_x = (-0.5, 1.0)
    twist_cmd.ranges.lin_vel_y = (-0.5, 0.5)
    twist_cmd.ranges.ang_vel_z = (-0.5, 0.5)

  return cfg
