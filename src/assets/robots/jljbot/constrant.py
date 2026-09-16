"""LowBody robot constants."""

import tempfile
import xml.etree.ElementTree as ET
from dataclasses import replace
from pathlib import Path
from typing import Any

import mujoco

from mjlab.actuator import DcMotorActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg
from src import SRC_PATH

try:
  from mjlab.actuator import DelayedActuatorCfg
except ImportError:
  DelayedActuatorCfg = None

##
# MJCF and assets.
##

LOWBODY_XML: Path = (
  SRC_PATH / "assets" / "robots" / "jljbot" / "lowbodycapsulev3.xml"
)
LOWBODY_ASSET_DIR = LOWBODY_XML.parent / "lowbodycapsulev3_assets"
assert LOWBODY_XML.exists()
assert LOWBODY_ASSET_DIR.exists()

LOWBODY_IMU_SITE = "imu_in_base_link"


def _prepare_lowbody_xml(xml_path: Path = LOWBODY_XML) -> str:
  """Return MJCF XML adjusted for mjlab-owned actuator config."""
  tree = ET.parse(xml_path)
  root = tree.getroot()

  compiler = root.find("compiler")
  if compiler is not None:
    compiler.set("meshdir", str(LOWBODY_ASSET_DIR))

  for actuator in root.findall("actuator"):
    root.remove(actuator)

  return ET.tostring(root, encoding="unicode")


def get_spec(xml_path: Path = LOWBODY_XML) -> mujoco.MjSpec:
  xml = _prepare_lowbody_xml(xml_path)
  with tempfile.NamedTemporaryFile("w", suffix=".xml", encoding="utf-8") as mjcf:
    mjcf.write(xml)
    mjcf.flush()
    return mujoco.MjSpec.from_file(mjcf.name)


##
# Actuator config.
##


LOWBODY_CONTROL_DELAY_MIN_LAG = 0
LOWBODY_CONTROL_DELAY_MAX_LAG = 2
LOWBODY_CONTROL_DELAY_TARGET = "position"


def _make_lowbody_actuator(
  joint_name: str,
  *,
  stiffness: float,
  damping: float,
  effort_limit: float,
  armature: float,
  velocity_limit: float,
) -> DcMotorActuatorCfg:
  """Create a single-joint LowBody DC motor actuator config."""
  return DcMotorActuatorCfg(
    target_names_expr=(joint_name,),
    stiffness=stiffness,
    damping=damping,
    effort_limit=effort_limit,
    saturation_effort=effort_limit,
    armature=armature,
    velocity_limit=velocity_limit,
  )


LOWBODY_ACTUATOR_LEFT_HIP_PITCH = _make_lowbody_actuator(
  "left_hip_pitch_joint",
  stiffness=100.0,
  damping=3.3,
  effort_limit=120.0,
  armature=0.055,
  velocity_limit=25.0,
)
LOWBODY_ACTUATOR_LEFT_HIP_ROLL = _make_lowbody_actuator(
  "left_hip_roll_joint",
  stiffness=100.0,
  damping=3.3,
  effort_limit=120.0,
  armature=0.055,
  velocity_limit=25.0,
)
LOWBODY_ACTUATOR_LEFT_HIP_YAW = _make_lowbody_actuator(
  "left_hip_yaw_joint",
  stiffness=100.0,
  damping=3.3,
  effort_limit=120.0,
  armature=0.055,
  velocity_limit=25.0,
)
LOWBODY_ACTUATOR_LEFT_KNEE = _make_lowbody_actuator(
  "left_knee_joint",
  stiffness=150.0,
  damping=5.0,
  effort_limit=120.0,
  armature=0.055,
  velocity_limit=25.0,
)
LOWBODY_ACTUATOR_LEFT_ANKLE_PITCH = _make_lowbody_actuator(
  "left_ankle_pitch_joint",
  stiffness=40.0,
  damping=2.0,
  effort_limit=27.0,
  armature=0.01,
  velocity_limit=8.0,
)
LOWBODY_ACTUATOR_LEFT_ANKLE_ROLL = _make_lowbody_actuator(
  "left_ankle_roll_joint",
  stiffness=40.0,
  damping=2.0,
  effort_limit=27.0,
  armature=0.01,
  velocity_limit=8.0,
)
LOWBODY_ACTUATOR_RIGHT_HIP_PITCH = _make_lowbody_actuator(
  "right_hip_pitch_joint",
  stiffness=100.0,
  damping=3.3,
  effort_limit=120.0,
  armature=0.055,
  velocity_limit=25.0,
)
LOWBODY_ACTUATOR_RIGHT_HIP_ROLL = _make_lowbody_actuator(
  "right_hip_roll_joint",
  stiffness=100.0,
  damping=3.3,
  effort_limit=120.0, 
  armature=0.055,
  velocity_limit=25.0,
)
LOWBODY_ACTUATOR_RIGHT_HIP_YAW = _make_lowbody_actuator(
  "right_hip_yaw_joint",
  stiffness=100.0,
  damping=3.3,
  effort_limit=120.0,
  armature=0.055,
  velocity_limit=25.0,
)
LOWBODY_ACTUATOR_RIGHT_KNEE = _make_lowbody_actuator(
  "right_knee_joint",
  stiffness=150.0,
  damping=5.0,
  effort_limit=120.0,
  armature=0.055,
  velocity_limit=25.0,
)
LOWBODY_ACTUATOR_RIGHT_ANKLE_PITCH = _make_lowbody_actuator(
  "right_ankle_pitch_joint",
  stiffness=40.0,
  damping=2.0,
  effort_limit=27.0,
  armature=0.01,
  velocity_limit=8.0,
)
LOWBODY_ACTUATOR_RIGHT_ANKLE_ROLL = _make_lowbody_actuator(
  "right_ankle_roll_joint",
  stiffness=40.0,
  damping=2.0,
  effort_limit=27.0,
  armature=0.01,
  velocity_limit=8.0,
)

LOWBODY_ACTUATORS: tuple[DcMotorActuatorCfg, ...] = (
  LOWBODY_ACTUATOR_LEFT_HIP_PITCH,
  LOWBODY_ACTUATOR_LEFT_HIP_ROLL,
  LOWBODY_ACTUATOR_LEFT_HIP_YAW,
  LOWBODY_ACTUATOR_LEFT_KNEE,
  LOWBODY_ACTUATOR_LEFT_ANKLE_PITCH,
  LOWBODY_ACTUATOR_LEFT_ANKLE_ROLL,
  LOWBODY_ACTUATOR_RIGHT_HIP_PITCH,
  LOWBODY_ACTUATOR_RIGHT_HIP_ROLL,
  LOWBODY_ACTUATOR_RIGHT_HIP_YAW,
  LOWBODY_ACTUATOR_RIGHT_KNEE,
  LOWBODY_ACTUATOR_RIGHT_ANKLE_PITCH,
  LOWBODY_ACTUATOR_RIGHT_ANKLE_ROLL,
)

LOWBODY_ACTION_SCALE_GAIN = 0.25


def _compute_lowbody_action_scale(
  actuators: tuple[DcMotorActuatorCfg, ...],
) -> dict[str, float]:
  """Compute joint-position action scales from nominal PD actuator settings."""
  action_scale: dict[str, float] = {}
  for actuator in actuators:
    (joint_name,) = actuator.target_names_expr
    action_scale[joint_name] = (
      LOWBODY_ACTION_SCALE_GAIN * actuator.effort_limit / actuator.stiffness
    )
  return action_scale


def _with_control_delay(
  actuator: DcMotorActuatorCfg,
  *,
  min_lag: int,
  max_lag: int,
) -> Any:
  """Wrap one LowBody actuator with command delay."""
  if DelayedActuatorCfg is None:
    return replace(actuator, delay_min_lag=min_lag, delay_max_lag=max_lag)

  return DelayedActuatorCfg(
    base_cfg=actuator,
    delay_target=LOWBODY_CONTROL_DELAY_TARGET,
    delay_min_lag=min_lag,
    delay_max_lag=max_lag,
  )


def _make_lowbody_actuators(
  enable_control_delay: bool,
  control_delay_min_lag: int,
  control_delay_max_lag: int,
) -> tuple[Any, ...]:
  """Return LowBody actuator configs with optional control delay."""
  if not enable_control_delay:
    return LOWBODY_ACTUATORS

  return tuple(
    _with_control_delay(
      actuator,
      min_lag=control_delay_min_lag,
      max_lag=control_delay_max_lag,
    )
    for actuator in LOWBODY_ACTUATORS
  )


##
# Keyframe config.
##

INIT_STATE = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 1.05),
  joint_pos={
    "left_hip_pitch_joint": -0.2,
    "right_hip_pitch_joint": 0.2,
    "left_hip_yaw_joint": 0.07,
    "right_hip_yaw_joint": -0.07,
    "left_knee_joint": -0.30,
    "right_knee_joint": 0.30,
    "left_ankle_pitch_joint": -0.15,
    "right_ankle_pitch_joint": -0.15,
  },
  joint_vel={".*": 0.0},
)

LOWBODY_JOINT_SDK_NAMES: tuple[str, ...] = (
  "left_hip_pitch_joint",
  "left_hip_roll_joint",
  "left_hip_yaw_joint",
  "left_knee_joint",
  "left_ankle_pitch_joint",
  "left_ankle_roll_joint",
  "right_hip_pitch_joint",
  "right_hip_roll_joint",
  "right_hip_yaw_joint",
  "right_knee_joint",
  "right_ankle_pitch_joint",
  "right_ankle_roll_joint",
)

##
# Collision config.
##

LOWBODY_FOOT_COLLISION_NAMES: tuple[str, ...] = tuple(
  f"{side}_foot{idx}_collision" for side in ("left", "right") for idx in range(1, 6)
)
FOOT_COLLISION_REGEX = r"^(left|right)_foot[1-5]_collision$"
LOWBODY_FOOT_FRICTION = (0.6,)
LOWBODY_FOOT_SOLREF = (0.01, 1.0)
LOWBODY_FOOT_SOLIMP = (0.9, 0.95, 0.001, 0.5, 2.0)

FULL_COLLISION = CollisionCfg(
  geom_names_expr=(".*_collision",),
  condim={FOOT_COLLISION_REGEX: 3, ".*_collision": 1},
  priority={FOOT_COLLISION_REGEX: 1},
  friction={FOOT_COLLISION_REGEX: LOWBODY_FOOT_FRICTION},
  solref={FOOT_COLLISION_REGEX: LOWBODY_FOOT_SOLREF},
  solimp={FOOT_COLLISION_REGEX: LOWBODY_FOOT_SOLIMP},
  disable_other_geoms=False,
)

FULL_COLLISION_WITHOUT_SELF = CollisionCfg(
  geom_names_expr=(".*_collision",),
  contype=0,
  conaffinity=1,
  condim={FOOT_COLLISION_REGEX: 3, ".*_collision": 1},
  priority={FOOT_COLLISION_REGEX: 1},
  friction={FOOT_COLLISION_REGEX: LOWBODY_FOOT_FRICTION},
  solref={FOOT_COLLISION_REGEX: LOWBODY_FOOT_SOLREF},
  solimp={FOOT_COLLISION_REGEX: LOWBODY_FOOT_SOLIMP},
  disable_other_geoms=False,
)

FEET_ONLY_COLLISION = CollisionCfg(
  geom_names_expr=(FOOT_COLLISION_REGEX,),
  contype=0,
  conaffinity=1,
  condim=3,
  priority=1,
  friction=LOWBODY_FOOT_FRICTION,
  solref=LOWBODY_FOOT_SOLREF,
  solimp=LOWBODY_FOOT_SOLIMP,
)

##
# Final config.
##

LOWBODY_ARTICULATION = EntityArticulationInfoCfg(
  actuators=LOWBODY_ACTUATORS,
  soft_joint_pos_limit_factor=0.9,
)


def _get_lowbody_articulation(
  enable_control_delay: bool = False,
  control_delay_min_lag: int = LOWBODY_CONTROL_DELAY_MIN_LAG,
  control_delay_max_lag: int = LOWBODY_CONTROL_DELAY_MAX_LAG,
) -> EntityArticulationInfoCfg:
  """Get LowBody articulation config."""
  if not enable_control_delay:
    return LOWBODY_ARTICULATION

  return EntityArticulationInfoCfg(
    actuators=_make_lowbody_actuators(
      enable_control_delay=True,
      control_delay_min_lag=control_delay_min_lag,
      control_delay_max_lag=control_delay_max_lag,
    ),
    soft_joint_pos_limit_factor=LOWBODY_ARTICULATION.soft_joint_pos_limit_factor,
  )


def get_lowbody_robot_cfg(
  enable_control_delay: bool = False,
  control_delay_min_lag: int = LOWBODY_CONTROL_DELAY_MIN_LAG,
  control_delay_max_lag: int = LOWBODY_CONTROL_DELAY_MAX_LAG,
) -> EntityCfg:
  """Get a fresh LowBody robot configuration instance."""
  return EntityCfg(
    init_state=INIT_STATE,
    collisions=(FULL_COLLISION,),
    spec_fn=get_spec,
    articulation=_get_lowbody_articulation(
      enable_control_delay=enable_control_delay,
      control_delay_min_lag=control_delay_min_lag,
      control_delay_max_lag=control_delay_max_lag,
    ),
  )


LOWBODY_ACTION_SCALE: dict[str, float] = _compute_lowbody_action_scale(
  LOWBODY_ACTUATORS
)


if __name__ == "__main__":
  import mujoco.viewer as viewer

  from mjlab.entity.entity import Entity

  robot = Entity(get_lowbody_robot_cfg())

  viewer.launch(robot.spec.compile())
