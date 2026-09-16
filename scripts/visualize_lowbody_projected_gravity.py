"""Interactive rigid-body viewer for checking LowBody projected gravity.

The LowBody joints are initialized from the training configuration and then
held fixed. Only the free base pose is changed by keyboard input, so this
viewer isolates the projected-gravity frame convention from robot dynamics.
"""

from __future__ import annotations

import argparse
import math
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import glfw
import mujoco
import mujoco.viewer
import numpy as np

from src.assets.robots.jljbot.constrant import (
  INIT_STATE,
  LOWBODY_IMU_SITE,
  LOWBODY_XML,
)


GRAVITY_W = np.array([0.0, 0.0, -1.0], dtype=np.float64)
BASE_BODY_NAME = "base_link"
GYRO_SENSOR_NAME = "imu_ang_vel"


@dataclass
class ViewerState:
  roll: float = 0.0
  pitch: float = 0.0
  yaw: float = 0.0
  step_deg: float = 2.0
  show_axes: bool = True
  show_gravity: bool = True
  reset_requested: bool = False


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
    description="Inspect LowBody projected_gravity_b while rotating the rigid robot."
  )
  parser.add_argument(
    "--step-deg",
    type=float,
    default=2.0,
    help="Initial keyboard rotation step in degrees.",
  )
  parser.add_argument(
    "--base-height",
    type=float,
    default=float(INIT_STATE.pos[2]),
    help="World Z position of the LowBody base.",
  )
  parser.add_argument(
    "--real-gravity",
    type=float,
    nargs=3,
    metavar=("GX", "GY", "GZ"),
    default=None,
    help="Optional real-robot projected gravity value for live comparison.",
  )
  parser.add_argument(
    "--real-ang-vel",
    type=float,
    nargs=3,
    metavar=("WX", "WY", "WZ"),
    default=None,
    help="Optional real-robot IMU angular velocity in rad/s for live comparison.",
  )
  parser.add_argument(
    "--no-arrows",
    action="store_true",
    help="Hide the base/IMU-frame and world-gravity arrows.",
  )
  return parser.parse_args()


def make_model() -> mujoco.MjModel:
  model = mujoco.MjModel.from_xml_path(str(Path(LOWBODY_XML).resolve()))
  model.opt.gravity[:] = GRAVITY_W * 9.81
  return model


def set_initial_pose(
  model: mujoco.MjModel,
  data: mujoco.MjData,
  state: ViewerState,
  base_height: float,
) -> tuple[int, int]:
  """Set the training LowBody joint pose and return base body/joint ids."""
  data.qpos[:] = model.qpos0
  data.qvel[:] = 0.0
  data.ctrl[:] = 0.0

  base_body_id = mujoco.mj_name2id(
    model, mujoco.mjtObj.mjOBJ_BODY, BASE_BODY_NAME
  )
  if base_body_id < 0:
    raise RuntimeError(f"Body '{BASE_BODY_NAME}' was not found in {LOWBODY_XML}")

  base_joint_id = int(model.body_jntadr[base_body_id])
  if model.jnt_type[base_joint_id] != mujoco.mjtJoint.mjJNT_FREE:
    raise RuntimeError(f"Body '{BASE_BODY_NAME}' does not have a free joint")

  root_qpos_addr = int(model.jnt_qposadr[base_joint_id])
  data.qpos[root_qpos_addr : root_qpos_addr + 3] = (
    float(INIT_STATE.pos[0]),
    float(INIT_STATE.pos[1]),
    base_height,
  )

  joint_values = dict(INIT_STATE.joint_pos)
  for joint_name, joint_value in joint_values.items():
    joint_id = mujoco.mj_name2id(
      model, mujoco.mjtObj.mjOBJ_JOINT, joint_name
    )
    if joint_id < 0:
      raise RuntimeError(f"Joint '{joint_name}' was not found in {LOWBODY_XML}")
    if model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_HINGE:
      raise RuntimeError(f"Joint '{joint_name}' is not a hinge joint")
    data.qpos[int(model.jnt_qposadr[joint_id])] = float(joint_value)

  state.roll = 0.0
  state.pitch = 0.0
  state.yaw = 0.0
  data.qvel[:] = 0.0
  update_base_orientation(model, data, state, base_joint_id)
  return base_body_id, base_joint_id


def find_imu_ids(model: mujoco.MjModel) -> tuple[int, int]:
  imu_site_id = mujoco.mj_name2id(
    model, mujoco.mjtObj.mjOBJ_SITE, LOWBODY_IMU_SITE
  )
  if imu_site_id < 0:
    raise RuntimeError(
      f"Site '{LOWBODY_IMU_SITE}' was not found in {LOWBODY_XML}"
    )

  gyro_sensor_id = mujoco.mj_name2id(
    model, mujoco.mjtObj.mjOBJ_SENSOR, GYRO_SENSOR_NAME
  )
  if gyro_sensor_id < 0:
    raise RuntimeError(
      f"Sensor '{GYRO_SENSOR_NAME}' was not found in {LOWBODY_XML}"
    )
  if model.sensor_dim[gyro_sensor_id] != 3:
    raise RuntimeError(
      f"Sensor '{GYRO_SENSOR_NAME}' must have dimension 3, got "
      f"{model.sensor_dim[gyro_sensor_id]}"
    )
  return imu_site_id, gyro_sensor_id


def update_base_orientation(
  model: mujoco.MjModel,
  data: mujoco.MjData,
  state: ViewerState,
  base_joint_id: int,
) -> None:
  quat = np.zeros(4, dtype=np.float64)
  euler_xyz = np.array([state.roll, state.pitch, state.yaw], dtype=np.float64)
  mujoco.mju_euler2Quat(quat, euler_xyz, "XYZ")
  root_qpos_addr = int(model.jnt_qposadr[base_joint_id])
  data.qpos[root_qpos_addr + 3 : root_qpos_addr + 7] = quat
  data.qvel[:] = 0.0
  mujoco.mj_forward(model, data)


def projected_gravity(
  data: mujoco.MjData,
  base_body_id: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  """Return root-frame gravity, root rotation, and root position."""
  rotation_w_from_b = np.asarray(data.xmat[base_body_id], dtype=np.float64).reshape(
    3, 3
  )
  gravity_b = rotation_w_from_b.T @ GRAVITY_W
  base_pos_w = np.asarray(data.xpos[base_body_id], dtype=np.float64).copy()
  return gravity_b, rotation_w_from_b, base_pos_w


def imu_measurements(
  model: mujoco.MjModel,
  data: mujoco.MjData,
  imu_site_id: int,
  gyro_sensor_id: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  """Return the IMU-frame gravity, site pose, and MuJoCo gyro output."""
  rotation_w_from_imu = np.asarray(
    data.site_xmat[imu_site_id], dtype=np.float64
  ).reshape(3, 3)
  gravity_imu = rotation_w_from_imu.T @ GRAVITY_W
  imu_pos_w = np.asarray(data.site_xpos[imu_site_id], dtype=np.float64).copy()
  sensor_addr = int(model.sensor_adr[gyro_sensor_id])
  gyro = np.asarray(
    data.sensordata[
      sensor_addr : sensor_addr + int(model.sensor_dim[gyro_sensor_id])
    ],
    dtype=np.float64,
  ).copy()
  return gravity_imu, imu_pos_w, gyro


def normalize(vector: np.ndarray) -> np.ndarray:
  norm = float(np.linalg.norm(vector))
  if norm < 1e-9:
    return np.zeros_like(vector)
  return vector / norm


def format_vector(vector: np.ndarray) -> str:
  return "[" + ", ".join(f"{value:+.5f}" for value in vector) + "]"


def set_arrow(
  geom: mujoco.MjvGeom,
  from_pos: np.ndarray,
  to_pos: np.ndarray,
  color: tuple[float, float, float, float],
  width: float = 0.008,
) -> None:
  mujoco.mjv_initGeom(
    geom,
    type=mujoco.mjtGeom.mjGEOM_ARROW.value,
    size=np.zeros(3, dtype=np.float64),
    pos=np.zeros(3, dtype=np.float64),
    mat=np.eye(3, dtype=np.float64).reshape(-1),
    rgba=np.asarray(color, dtype=np.float32),
  )
  mujoco.mjv_connector(
    geom,
    type=mujoco.mjtGeom.mjGEOM_ARROW.value,
    width=width,
    from_=from_pos,
    to=to_pos,
  )
  geom.category = mujoco.mjtCatBit.mjCAT_DECOR


def update_debug_arrows(
  viewer: mujoco.viewer.Handle,
  base_pos_w: np.ndarray,
  rotation_w_from_b: np.ndarray,
  imu_pos_w: np.ndarray,
  rotation_w_from_imu: np.ndarray,
  show_axes: bool,
  show_gravity: bool,
) -> None:
  scene = viewer.user_scn
  scene.ngeom = 0
  arrow_length = 0.32

  if show_axes:
    axis_colors = (
      (0.95, 0.15, 0.15, 1.0),
      (0.15, 0.9, 0.2, 1.0),
      (0.15, 0.35, 1.0, 1.0),
    )
    for axis_index, color in enumerate(axis_colors):
      geom = scene.geoms[scene.ngeom]
      set_arrow(
        geom,
        base_pos_w,
        base_pos_w + arrow_length * rotation_w_from_b[:, axis_index],
        color,
      )
      scene.ngeom += 1

    # The IMU site has the same axes as base_link in the LowBody XML.
    for axis_index, color in enumerate(axis_colors):
      geom = scene.geoms[scene.ngeom]
      set_arrow(
        geom,
        imu_pos_w,
        imu_pos_w + 0.22 * rotation_w_from_imu[:, axis_index],
        color,
        width=0.004,
      )
      scene.ngeom += 1

  if show_gravity:
    geom = scene.geoms[scene.ngeom]
    gravity_start = base_pos_w + np.array([0.0, 0.0, 0.12])
    set_arrow(
      geom,
      gravity_start,
      gravity_start + 0.55 * GRAVITY_W,
      (1.0, 0.75, 0.05, 1.0),
      width=0.012,
    )
    scene.ngeom += 1


def make_overlay(
  state: ViewerState,
  gravity_b: np.ndarray,
  gravity_imu: np.ndarray,
  imu_pos_b: np.ndarray,
  gyro: np.ndarray,
  real_gravity: np.ndarray | None,
  real_ang_vel: np.ndarray | None,
) -> tuple[int, int, str, str]:
  angles = np.degrees([state.roll, state.pitch, state.yaw])
  gravity_delta = gravity_imu - gravity_b
  left = (
    "LowBody projected_gravity_b\n"
    "--------------------------\n"
    "source     base_link root frame\n"
    f"gravity_b  {format_vector(gravity_b)}\n"
    f"norm       {np.linalg.norm(gravity_b):.5f}\n"
    "\n"
    "Training gyro\n"
    "-------------\n"
    f"sensor     {GYRO_SENSOR_NAME}\n"
    f"site       {LOWBODY_IMU_SITE}\n"
    f"site pos_b {format_vector(imu_pos_b)} m\n"
    f"gyro       {format_vector(gyro)} rad/s\n"
    "state      qvel=0 (static)\n"
    "\n"
    "IMU gravity check\n"
    "-----------------\n"
    f"gravity_imu {format_vector(gravity_imu)}\n"
    f"imu-root   {format_vector(gravity_delta)}\n"
    f"roll       {angles[0]:+7.2f} deg\n"
    f"pitch      {angles[1]:+7.2f} deg\n"
    f"yaw        {angles[2]:+7.2f} deg\n"
    f"step       {state.step_deg:.1f} deg"
  )
  right = (
    "Controls\n"
    "--------\n"
    "Up/Down: pitch +/-\n"
    "Left/Right: roll +/-\n"
    "A/D: yaw +/-\n"
    "R/Home: reset\n"
    "+/-: step size\n"
    "X: toggle base axes\n"
    "G: toggle gravity arrow"
  )

  if real_gravity is not None:
    delta = gravity_b - real_gravity
    angle = math.degrees(
      math.acos(
        float(
          np.clip(
            np.dot(normalize(gravity_b), normalize(real_gravity)),
            -1.0,
            1.0,
          )
        )
      )
    )
    right += (
      "\n\nReal comparison\n"
      "---------------\n"
      f"real       {format_vector(real_gravity)}\n"
      f"delta      {format_vector(delta)}\n"
      f"angle      {angle:.3f} deg"
    )

  if real_ang_vel is not None:
    delta = gyro - real_ang_vel
    right += (
      "\n\nReal gyro comparison\n"
      "--------------------\n"
      f"real       {format_vector(real_ang_vel)} rad/s\n"
      f"delta      {format_vector(delta)} rad/s"
    )

  return (
    mujoco.mjtFontScale.mjFONTSCALE_150.value,
    mujoco.mjtGridPos.mjGRID_TOPLEFT.value,
    left,
    right,
  )


def key_callback_factory(state: ViewerState, state_lock: threading.Lock):
  def key_callback(key: int) -> None:
    with state_lock:
      step = math.radians(state.step_deg)
      if key in (glfw.KEY_UP, glfw.KEY_I):
        state.pitch += step
      elif key in (glfw.KEY_DOWN, glfw.KEY_K):
        state.pitch -= step
      elif key in (glfw.KEY_LEFT, glfw.KEY_J):
        state.roll += step
      elif key in (glfw.KEY_RIGHT, glfw.KEY_L):
        state.roll -= step
      elif key in (glfw.KEY_A, glfw.KEY_U):
        state.yaw += step
      elif key in (glfw.KEY_D, glfw.KEY_O):
        state.yaw -= step
      elif key in (glfw.KEY_R, glfw.KEY_HOME):
        state.reset_requested = True
      elif key in (glfw.KEY_EQUAL, glfw.KEY_KP_ADD):
        state.step_deg = min(45.0, state.step_deg + 1.0)
      elif key in (glfw.KEY_MINUS, glfw.KEY_KP_SUBTRACT):
        state.step_deg = max(0.1, state.step_deg - 1.0)
      elif key == glfw.KEY_X:
        state.show_axes = not state.show_axes
      elif key == glfw.KEY_G:
        state.show_gravity = not state.show_gravity

  return key_callback


def run(args: argparse.Namespace) -> None:
  model = make_model()
  data = mujoco.MjData(model)
  state = ViewerState(step_deg=args.step_deg)
  state_lock = threading.Lock()
  base_body_id, base_joint_id = set_initial_pose(
    model,
    data,
    state,
    base_height=args.base_height,
  )
  imu_site_id, gyro_sensor_id = find_imu_ids(model)

  real_gravity = (
    np.asarray(args.real_gravity, dtype=np.float64)
    if args.real_gravity is not None
    else None
  )
  if real_gravity is not None:
    real_gravity = normalize(real_gravity)
  real_ang_vel = (
    np.asarray(args.real_ang_vel, dtype=np.float64)
    if args.real_ang_vel is not None
    else None
  )

  with mujoco.viewer.launch_passive(
    model,
    data,
    key_callback=key_callback_factory(state, state_lock),
    show_left_ui=True,
    show_right_ui=False,
  ) as viewer:
    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    viewer.cam.trackbodyid = base_body_id
    viewer.cam.distance = 2.2
    viewer.cam.azimuth = 135.0
    viewer.cam.elevation = -18.0

    while viewer.is_running():
      with state_lock:
        reset_requested = state.reset_requested
        state.reset_requested = False
        state_snapshot = ViewerState(
          roll=state.roll,
          pitch=state.pitch,
          yaw=state.yaw,
          step_deg=state.step_deg,
          show_axes=state.show_axes,
          show_gravity=state.show_gravity,
        )

      if reset_requested:
        state_snapshot.roll = 0.0
        state_snapshot.pitch = 0.0
        state_snapshot.yaw = 0.0
        with state_lock:
          state.roll = 0.0
          state.pitch = 0.0
          state.yaw = 0.0

      with viewer.lock():
        update_base_orientation(model, data, state_snapshot, base_joint_id)
        gravity_b, rotation_w_from_b, base_pos_w = projected_gravity(
          data, base_body_id
        )
        rotation_w_from_imu = np.asarray(
          data.site_xmat[imu_site_id], dtype=np.float64
        ).reshape(3, 3)
        gravity_imu, imu_pos_w, gyro = imu_measurements(
          model,
          data,
          imu_site_id,
          gyro_sensor_id,
        )
        imu_pos_b = rotation_w_from_b.T @ (imu_pos_w - base_pos_w)
        if not args.no_arrows:
          update_debug_arrows(
            viewer,
            base_pos_w,
            rotation_w_from_b,
            imu_pos_w,
            rotation_w_from_imu,
            state_snapshot.show_axes,
            state_snapshot.show_gravity,
          )
        viewer.set_texts(
          make_overlay(
            state_snapshot,
            gravity_b,
            gravity_imu,
            imu_pos_b,
            gyro,
            real_gravity,
            real_ang_vel,
          )
        )
        viewer.sync(state_only=True)

      time.sleep(1.0 / 60.0)


def main() -> None:
  run(parse_args())


if __name__ == "__main__":
  main()
