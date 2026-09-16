"""Constants and MJCF adapter for the A00 JLJ whole-body robot.

The source model is an exported MJCF with named collision geoms, task sensors,
and dedicated whole-body foot collisions. The adapter resolves its mesh path and
removes legacy MuJoCo position actuators.
"""

from __future__ import annotations

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
except ImportError:  # pragma: no cover - depends on the installed mjlab version.
  DelayedActuatorCfg = None


JLJWHOLE_XML: Path = (
  SRC_PATH
  / "assets"
  / "robots"
  / "jljwhole"
  / "A00机器人全身_mjcf"
  / "A00机器人全身"
  / "A00机器人全身.xml"
)
JLJWHOLE_ASSET_DIR = JLJWHOLE_XML.parent / "meshes"
assert JLJWHOLE_XML.exists()
assert JLJWHOLE_ASSET_DIR.exists()

JLJWHOLE_BASE_BODY = "base_link"
JLJWHOLE_IMU_SITE = "imu_in_base_link"
# Kept as a reference for the hand-maintained XML site.
JLJWHOLE_IMU_SITE_POS = "0 0 -0.006"
JLJWHOLE_FOOT_SITE_NAMES = ("left_foot", "right_foot")
JLJWHOLE_FOOT_COLLISION_NAMES = tuple(
  f"{side}_foot{index}_collision"
  for side in ("left", "right")
  for index in range(1, 6)
)

def _prepare_jljwhole_xml(xml_path: Path = JLJWHOLE_XML) -> str:
  """Return the source MJCF with runtime-owned settings applied."""
  tree = ET.parse(xml_path)
  root = tree.getroot()

  compiler = root.find("compiler")
  if compiler is None:
    compiler = ET.Element("compiler")
    root.insert(0, compiler)
  compiler.set("meshdir", str(JLJWHOLE_ASSET_DIR))

  # Actuators are owned by mjlab so action scaling, limits, and delay behavior
  # remain configurable from Python.
  for actuator in root.findall("actuator"):
    root.remove(actuator)

  return ET.tostring(root, encoding="unicode")


def get_spec(xml_path: Path = JLJWHOLE_XML) -> mujoco.MjSpec:
  """Build the A00 model without its legacy actuators."""
  xml = _prepare_jljwhole_xml(xml_path)
  with tempfile.NamedTemporaryFile("w", suffix=".xml", encoding="utf-8") as mjcf:
    mjcf.write(xml)
    mjcf.flush()
    return mujoco.MjSpec.from_file(mjcf.name)


JLJWHOLE_CONTROL_DELAY_MIN_LAG = 0
JLJWHOLE_CONTROL_DELAY_MAX_LAG = 2
JLJWHOLE_CONTROL_DELAY_TARGET = "position"


def _make_actuator(
  target_names_expr: tuple[str, ...],
  *,
  stiffness: float,
  damping: float,
  effort_limit: float,
  armature: float,
  velocity_limit: float,
) -> DcMotorActuatorCfg:
  return DcMotorActuatorCfg(
    target_names_expr=target_names_expr,
    stiffness=stiffness,
    damping=damping,
    effort_limit=effort_limit,
    saturation_effort=effort_limit,
    armature=armature,
    velocity_limit=velocity_limit,
  )


# Leg and upper-body values are grouped independently and intentionally easy to
# tune by hand.
JLJWHOLE_ACTUATOR_HIP = _make_actuator(
  (
    "left_hip_pitch",
    "right_hip_pitch",
    "left_hip_roll",
    "right_hip_roll",
    "left_hip_yaw",
    "right_hip_yaw",
  ),
  stiffness=200.0,
  damping=3.3,
  effort_limit=120.0,
  armature=0.055,
  velocity_limit=25.0,
)
JLJWHOLE_ACTUATOR_KNEE = _make_actuator(
  ("left_knee", "right_knee"),
  stiffness=250.0,
  damping=5.0,
  effort_limit=120.0,
  armature=0.055,
  velocity_limit=25.0,
)
JLJWHOLE_ACTUATOR_ANKLE = _make_actuator(
  (
    "left_ankle_pitch",
    "right_ankle_pitch",
    "left_ankle_roll",
    "right_ankle_roll",
  ),
  stiffness=100.0,
  damping=3.0,
  effort_limit=27.0,
  armature=0.01,
  velocity_limit=8.0,
)
JLJWHOLE_ACTUATOR_WAIST = _make_actuator(
  ("waist_yaw", "waist_roll", "waist_pitch"),
  stiffness=150.0,
  damping=4.0,
  effort_limit=40.0,
  armature=0.01,
  velocity_limit=15.0,
)
JLJWHOLE_ACTUATOR_ARM = _make_actuator(
  (
    "left_shoulder_pitch",
    "right_shoulder_pitch",
    "left_shoulder_roll",
    "right_shoulder_roll",
    "left_shoulder_yaw",
    "right_shoulder_yaw",
    "left_elbow",
    "right_elbow",
  ),
  stiffness=40.0,
  damping=2.0,
  effort_limit=40.0,
  armature=0.01,
  velocity_limit=20.0,
)
JLJWHOLE_ACTUATOR_WRIST = _make_actuator(
  (
    "left_wrist_yaw",
    "right_wrist_yaw",
    "left_wrist_roll",
    "right_wrist_roll",
    "left_wrist_pitch",
    "right_wrist_pitch",
  ),
  stiffness=20.0,
  damping=1.0,
  effort_limit=20.0,
  armature=0.01,
  velocity_limit=20.0,
)

JLJWHOLE_ACTUATORS: tuple[DcMotorActuatorCfg, ...] = (
  JLJWHOLE_ACTUATOR_HIP,
  JLJWHOLE_ACTUATOR_KNEE,
  JLJWHOLE_ACTUATOR_ANKLE,
  JLJWHOLE_ACTUATOR_WAIST,
  JLJWHOLE_ACTUATOR_ARM,
  JLJWHOLE_ACTUATOR_WRIST,
)


def _with_control_delay(
  actuator: DcMotorActuatorCfg,
  *,
  min_lag: int,
  max_lag: int,
) -> Any:
  if DelayedActuatorCfg is None:
    return replace(actuator, delay_min_lag=min_lag, delay_max_lag=max_lag)
  return DelayedActuatorCfg(
    base_cfg=actuator,
    delay_target=JLJWHOLE_CONTROL_DELAY_TARGET,
    delay_min_lag=min_lag,
    delay_max_lag=max_lag,
  )


def _get_articulation(
  enable_control_delay: bool = False,
  control_delay_min_lag: int = JLJWHOLE_CONTROL_DELAY_MIN_LAG,
  control_delay_max_lag: int = JLJWHOLE_CONTROL_DELAY_MAX_LAG,
) -> EntityArticulationInfoCfg:
  if not enable_control_delay:
    return JLJWHOLE_ARTICULATION
  return EntityArticulationInfoCfg(
    actuators=tuple(
      _with_control_delay(
        actuator,
        min_lag=control_delay_min_lag,
        max_lag=control_delay_max_lag,
      )
      for actuator in JLJWHOLE_ACTUATORS
    ),
    soft_joint_pos_limit_factor=JLJWHOLE_ARTICULATION.soft_joint_pos_limit_factor,
  )


JLJWHOLE_INIT_STATE = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 1.00),
  joint_pos={
    "left_hip_pitch": -0.2,
    "right_hip_pitch": 0.2,
    "left_hip_yaw": 0.07,
    "right_hip_yaw": -0.07,
    "left_knee": -0.30,
    "right_knee": 0.30,
    "left_ankle_pitch": -0.15,
    "right_ankle_pitch": -0.15,
    "left_elbow":-0.3,
    "right_elbow":-0.3,
    "left_shoulder_pitch":0.4,
    "right_shoulder_pitch":-0.4,
  },
  joint_vel={".*": 0.0},
)

JLJWHOLE_ARTICULATION = EntityArticulationInfoCfg(
  actuators=JLJWHOLE_ACTUATORS,
  soft_joint_pos_limit_factor=0.9,
)

JLJWHOLE_COLLISION_REGEX = r".*_collision.*"
JLJWHOLE_FOOT_COLLISION_REGEX = r"^(left|right)_foot[1-5]_collision$"
JLJWHOLE_FOOT_FRICTION = (0.6,)
JLJWHOLE_FOOT_SOLREF = (0.01, 1.0)
JLJWHOLE_FOOT_SOLIMP = (0.9, 0.95, 0.001, 0.5, 2.0)

JLJWHOLE_FULL_COLLISION = CollisionCfg(
  geom_names_expr=(JLJWHOLE_COLLISION_REGEX,),
  condim={JLJWHOLE_FOOT_COLLISION_REGEX: 3, JLJWHOLE_COLLISION_REGEX: 1},
  priority={JLJWHOLE_FOOT_COLLISION_REGEX: 1},
  friction={JLJWHOLE_FOOT_COLLISION_REGEX: JLJWHOLE_FOOT_FRICTION},
  solref={JLJWHOLE_FOOT_COLLISION_REGEX: JLJWHOLE_FOOT_SOLREF},
  solimp={JLJWHOLE_FOOT_COLLISION_REGEX: JLJWHOLE_FOOT_SOLIMP},
  disable_other_geoms=False,
)


def get_jljwholebody_robot_cfg(
  enable_control_delay: bool = False,
  control_delay_min_lag: int = JLJWHOLE_CONTROL_DELAY_MIN_LAG,
  control_delay_max_lag: int = JLJWHOLE_CONTROL_DELAY_MAX_LAG,
) -> EntityCfg:
  """Return a fresh A00 whole-body entity configuration."""
  return EntityCfg(
    init_state=JLJWHOLE_INIT_STATE,
    collisions=(JLJWHOLE_FULL_COLLISION,),
    spec_fn=get_spec,
    articulation=_get_articulation(
      enable_control_delay=enable_control_delay,
      control_delay_min_lag=control_delay_min_lag,
      control_delay_max_lag=control_delay_max_lag,
    ),
  )


# Short alias matching the robot package name.
get_jljwhole_robot_cfg = get_jljwholebody_robot_cfg


JLJWHOLE_ACTION_SCALE: dict[str, float] = {
  joint_name: 0.25 * actuator.effort_limit / actuator.stiffness
  for actuator in JLJWHOLE_ACTUATORS
  for joint_name in actuator.target_names_expr
}


if __name__ == "__main__":
  import mujoco.viewer as viewer

  from mjlab.entity.entity import Entity

  viewer.launch(Entity(get_jljwholebody_robot_cfg()).spec.compile())
