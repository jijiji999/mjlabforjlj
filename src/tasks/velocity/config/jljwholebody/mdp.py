"""JLJ A00 whole-body MDP helpers for velocity tracking."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, TypedDict, cast

import numpy as np
import torch

from mjlab.entity import Entity
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.managers.event_manager import requires_model_fields
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import BuiltinSensor, ContactSensor, RayCastSensor
from mjlab.utils.lab_api.math import (
  matrix_from_quat,
  quat_apply,
  quat_apply_inverse,
  quat_from_euler_xyz,
  quat_mul,
  sample_uniform,
  wrap_to_pi,
)
from mjlab.utils.lab_api.string import resolve_matching_names_values

if TYPE_CHECKING:
  import viser

  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.viewer.debug_visualizer import DebugVisualizer

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")
_SE3_KEYS = ("x", "y", "z", "roll", "pitch", "yaw")


def _sample_se3_range(
  range_dict: dict[str, tuple[float, float]] | None,
  shape: tuple[int, ...],
  device: str,
) -> torch.Tensor:
  range_dict = range_dict or {}
  range_list = [range_dict.get(key, (0.0, 0.0)) for key in _SE3_KEYS]
  ranges = torch.tensor(range_list, device=device)
  return sample_uniform(ranges[:, 0], ranges[:, 1], shape, device=device)


def resolve_env_ids(
  env: ManagerBasedRlEnv, env_ids: torch.Tensor | None
) -> torch.Tensor:
  if env_ids is None:
    return torch.arange(env.num_envs, device=env.device, dtype=torch.int)
  return env_ids


def randomize_terrain(env: ManagerBasedRlEnv, env_ids: torch.Tensor | None) -> None:
  env_ids = resolve_env_ids(env, env_ids)
  terrain = env.scene.terrain
  if terrain is not None:
    terrain.randomize_env_origins(env_ids)


@requires_model_fields("geom_solref")
def foot_contact_solref_timeconst(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  ranges: tuple[float, float],
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  shared_random: bool = True,
) -> None:
  """Randomize foot contact time constant in ``geom_solref[:, 0]``."""
  env_ids = resolve_env_ids(env, env_ids)
  asset: Entity = env.scene[asset_cfg.name]
  geom_ids = asset.indexing.geom_ids[asset_cfg.geom_ids]

  if shared_random:
    samples = sample_uniform(*ranges, (len(env_ids), 1), env.device)
    samples = samples.expand(len(env_ids), len(geom_ids))
  else:
    samples = sample_uniform(*ranges, (len(env_ids), len(geom_ids)), env.device)

  env_grid, geom_grid = torch.meshgrid(env_ids, geom_ids, indexing="ij")
  env.sim.model.geom_solref[env_grid, geom_grid, 0] = samples


def reset_root_state_uniform(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  pose_range: dict[str, tuple[float, float]],
  velocity_range: dict[str, tuple[float, float]] | None = None,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> None:
  env_ids = resolve_env_ids(env, env_ids)
  asset: Entity = env.scene[asset_cfg.name]
  pose_samples = _sample_se3_range(pose_range, (len(env_ids), 6), env.device)

  if asset.is_fixed_base:
    if not asset.is_mocap:
      raise ValueError(
        f"Cannot reset root state for fixed-base non-mocap entity '{asset_cfg.name}'."
      )
    default_root_state = asset.data.default_root_state
    assert default_root_state is not None
    root_states = default_root_state[env_ids].clone()
    positions = (
      root_states[:, 0:3] + pose_samples[:, 0:3] + env.scene.env_origins[env_ids]
    )
    orientations_delta = quat_from_euler_xyz(
      pose_samples[:, 3], pose_samples[:, 4], pose_samples[:, 5]
    )
    orientations = quat_mul(root_states[:, 3:7], orientations_delta)
    asset.write_mocap_pose_to_sim(
      torch.cat([positions, orientations], dim=-1), env_ids=env_ids
    )
    return

  default_root_state = asset.data.default_root_state
  assert default_root_state is not None
  root_states = default_root_state[env_ids].clone()

  positions = (
    root_states[:, 0:3] + pose_samples[:, 0:3] + env.scene.env_origins[env_ids]
  )
  orientations_delta = quat_from_euler_xyz(
    pose_samples[:, 3], pose_samples[:, 4], pose_samples[:, 5]
  )
  orientations = quat_mul(root_states[:, 3:7], orientations_delta)
  vel_samples = _sample_se3_range(velocity_range, (len(env_ids), 6), env.device)
  velocities = root_states[:, 7:13] + vel_samples

  asset.write_root_link_pose_to_sim(
    torch.cat([positions, orientations], dim=-1), env_ids=env_ids
  )
  asset.write_root_link_velocity_to_sim(velocities, env_ids=env_ids)


def reset_joints_by_offset(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  position_range: tuple[float, float],
  velocity_range: tuple[float, float],
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> None:
  env_ids = resolve_env_ids(env, env_ids)
  asset: Entity = env.scene[asset_cfg.name]
  default_joint_pos = asset.data.default_joint_pos
  assert default_joint_pos is not None
  default_joint_vel = asset.data.default_joint_vel
  assert default_joint_vel is not None
  soft_joint_pos_limits = asset.data.soft_joint_pos_limits
  assert soft_joint_pos_limits is not None

  joint_pos = default_joint_pos[env_ids][:, asset_cfg.joint_ids].clone()
  joint_pos += sample_uniform(*position_range, joint_pos.shape, env.device)
  joint_pos_limits = soft_joint_pos_limits[env_ids][:, asset_cfg.joint_ids]
  joint_pos = joint_pos.clamp_(joint_pos_limits[..., 0], joint_pos_limits[..., 1])

  joint_vel = default_joint_vel[env_ids][:, asset_cfg.joint_ids].clone()
  joint_vel += sample_uniform(*velocity_range, joint_vel.shape, env.device)

  joint_ids = asset_cfg.joint_ids
  if isinstance(joint_ids, list):
    joint_ids = torch.tensor(joint_ids, device=env.device)

  asset.write_joint_state_to_sim(
    joint_pos.view(len(env_ids), -1),
    joint_vel.view(len(env_ids), -1),
    env_ids=env_ids,
    joint_ids=joint_ids,
  )


def push_by_setting_velocity(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  velocity_range: dict[str, tuple[float, float]],
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> None:
  env_ids = resolve_env_ids(env, env_ids)
  asset: Entity = env.scene[asset_cfg.name]
  vel_w = asset.data.root_link_vel_w[env_ids]
  vel_w += _sample_se3_range(velocity_range, vel_w.shape, env.device)
  asset.write_root_link_velocity_to_sim(vel_w, env_ids=env_ids)


def builtin_sensor(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  sensor = env.scene[sensor_name]
  assert isinstance(sensor, BuiltinSensor)
  return sensor.data


def projected_gravity(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  return asset.data.projected_gravity_b


def generated_commands(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  return command


def joint_pos_rel(
  env: ManagerBasedRlEnv,
  biased: bool = False,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  default_joint_pos = asset.data.default_joint_pos
  assert default_joint_pos is not None
  joint_pos = asset.data.joint_pos_biased if biased else asset.data.joint_pos
  return joint_pos[:, asset_cfg.joint_ids] - default_joint_pos[:, asset_cfg.joint_ids]


def joint_pos_deviation_l2(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """Penalize selected joints deviating from their default positions."""
  asset: Entity = env.scene[asset_cfg.name]
  default_joint_pos = asset.data.default_joint_pos
  assert default_joint_pos is not None
  joint_pos_error = (
    asset.data.joint_pos[:, asset_cfg.joint_ids]
    - default_joint_pos[:, asset_cfg.joint_ids]
  )
  return torch.sum(torch.square(joint_pos_error), dim=1)


def joint_vel_rel(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  default_joint_vel = asset.data.default_joint_vel
  assert default_joint_vel is not None
  return asset.data.joint_vel[:, asset_cfg.joint_ids] - default_joint_vel[
    :, asset_cfg.joint_ids
  ]


def last_action(env: ManagerBasedRlEnv, action_name: str | None = None) -> torch.Tensor:
  if action_name is None:
    return env.action_manager.action
  return env.action_manager.get_term(action_name).raw_action


def height_scan(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  offset: float = 0.0,
  miss_value: float | None = None,
) -> torch.Tensor:
  sensor: RayCastSensor = env.scene[sensor_name]
  if miss_value is None:
    miss_value = sensor.cfg.max_distance

  data = sensor.data
  num_frames, num_rays = sensor.num_frames, sensor.num_rays_per_frame
  batch_size = data.distances.shape[0]
  frame_z = data.frame_pos_w[:, :, 2:3]
  hit_z = data.hit_pos_w[..., 2].view(batch_size, num_frames, num_rays)
  heights = (frame_z - hit_z - offset).view(batch_size, num_frames * num_rays)
  miss_mask = data.distances < 0
  return torch.where(miss_mask, torch.full_like(heights, miss_value), heights)


def foot_height(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  return asset.data.site_pos_w[:, asset_cfg.site_ids, 2]


def foot_air_time(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  current_air_time = sensor.data.current_air_time
  assert current_air_time is not None
  return current_air_time


def foot_contact(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  assert sensor.data.found is not None
  return (sensor.data.found > 0).float()


def foot_contact_forces(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  assert sensor.data.force is not None
  forces_flat = sensor.data.force.flatten(start_dim=1)
  return torch.sign(forces_flat) * torch.log1p(torch.abs(forces_flat))


def phase(env: ManagerBasedRlEnv, period: float, command_name: str) -> torch.Tensor:
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  global_phase = (env.episode_length_buf * env.step_dt) % period / period
  phase_obs = torch.zeros(env.num_envs, 2, device=env.device)
  phase_obs[:, 0] = torch.sin(global_phase * torch.pi * 2.0)
  phase_obs[:, 1] = torch.cos(global_phase * torch.pi * 2.0)
  stand_mask = torch.linalg.norm(command, dim=1) < 0.1
  return torch.where(stand_mask.unsqueeze(1), torch.zeros_like(phase_obs), phase_obs)


def mean_action_acc(env: ManagerBasedRlEnv) -> torch.Tensor:
  action_acc = (
    env.action_manager.action
    - 2 * env.action_manager.prev_action
    + env.action_manager.prev_prev_action
  )
  return torch.mean(torch.abs(action_acc), dim=-1)


def track_linear_velocity(
  env: ManagerBasedRlEnv,
  std: float,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  actual = asset.data.root_link_lin_vel_b
  xy_error = torch.sum(torch.square(command[:, :2] - actual[:, :2]), dim=1)
  z_error = torch.square(actual[:, 2])
  lin_vel_error = xy_error + (2 * z_error)
  return torch.exp(-lin_vel_error / std**2)


def track_angular_velocity(
  env: ManagerBasedRlEnv,
  std: float,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  actual = asset.data.root_link_ang_vel_b
  z_error = torch.square(command[:, 2] - actual[:, 2])
  xy_error = torch.sum(torch.square(actual[:, :2]), dim=1)
  ang_vel_error = z_error + (0.05 * xy_error)
  return torch.exp(-ang_vel_error / std**2)


def body_orientation_l2(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  if asset_cfg.body_ids:
    body_quat_w = asset.data.body_link_quat_w[:, asset_cfg.body_ids, :].squeeze(1)
    gravity_w = asset.data.gravity_vec_w
    projected_gravity_b = quat_apply_inverse(body_quat_w, gravity_w)
    return torch.sum(torch.square(projected_gravity_b[:, :2]), dim=1)
  return torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)


def self_collision_cost(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  force_threshold: float = 10.0,
) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  data = sensor.data
  if data.force_history is not None:
    force_mag = torch.norm(data.force_history, dim=-1)
    hit = (force_mag > force_threshold).any(dim=1)
    return hit.sum(dim=-1).float()
  assert data.found is not None
  return data.found.squeeze(-1)


def body_angular_velocity_penalty(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  ang_vel = asset.data.body_link_ang_vel_w[:, asset_cfg.body_ids, :].squeeze(1)
  return torch.sum(torch.square(ang_vel[:, :2]), dim=1)


def angular_momentum_penalty(
  env: ManagerBasedRlEnv,
  sensor_name: str,
) -> torch.Tensor:
  angmom_sensor: BuiltinSensor = env.scene[sensor_name]
  angmom = angmom_sensor.data
  angmom_magnitude_sq = torch.sum(torch.square(angmom), dim=-1)
  env.extras["log"]["Metrics/angular_momentum_mean"] = torch.mean(
    torch.sqrt(angmom_magnitude_sq)
  )
  return angmom_magnitude_sq


def is_terminated(env: ManagerBasedRlEnv) -> torch.Tensor:
  return env.termination_manager.terminated.float()


def joint_acc_l2(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  return torch.sum(torch.square(asset.data.joint_acc[:, asset_cfg.joint_ids]), dim=1)


def joint_pos_limits(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  soft_joint_pos_limits = asset.data.soft_joint_pos_limits
  assert soft_joint_pos_limits is not None
  out_of_limits = -(
    asset.data.joint_pos[:, asset_cfg.joint_ids]
    - soft_joint_pos_limits[:, asset_cfg.joint_ids, 0]
  ).clip(max=0.0)
  out_of_limits += (
    asset.data.joint_pos[:, asset_cfg.joint_ids]
    - soft_joint_pos_limits[:, asset_cfg.joint_ids, 1]
  ).clip(min=0.0)
  return torch.sum(out_of_limits, dim=1)


def action_rate_l2(env: ManagerBasedRlEnv) -> torch.Tensor:
  return torch.sum(
    torch.square(env.action_manager.action - env.action_manager.prev_action), dim=1
  )


def feet_air_time(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  threshold: float = 0.4,
  command_name: str | None = None,
  command_threshold: float = 0.1,
) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  air_time = sensor.data.current_air_time
  contact_time = sensor.data.current_contact_time
  assert air_time is not None
  assert contact_time is not None
  in_contact = contact_time > 0.0
  in_mode_time = torch.where(in_contact, contact_time, air_time)
  single_stance = torch.mean(in_contact.float(), dim=1) == 0.5
  mode_time = torch.min(
    torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1
  )[0]
  reward = torch.clamp(threshold - torch.abs(mode_time - threshold), min=0.0)
  if command_name is not None:
    command = env.command_manager.get_command(command_name)
    if command is not None:
      total_command = torch.norm(command[:, :2], dim=1) + torch.abs(command[:, 2])
      reward *= (total_command > command_threshold).float()
  return reward


def feet_clearance(
  env: ManagerBasedRlEnv,
  target_height: float,
  command_name: str | None = None,
  command_threshold: float = 0.1,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  foot_z = asset.data.site_pos_w[:, asset_cfg.site_ids, 2]
  foot_vel_xy = asset.data.site_lin_vel_w[:, asset_cfg.site_ids, :2]
  vel_norm = torch.norm(foot_vel_xy, dim=-1)
  cost = torch.sum(torch.abs(foot_z - target_height) * vel_norm, dim=1)
  if command_name is not None:
    command = env.command_manager.get_command(command_name)
    if command is not None:
      total_command = torch.norm(command[:, :2], dim=1) + torch.abs(command[:, 2])
      cost *= (total_command > command_threshold).float()
  return cost


def feet_gait(
  env: ManagerBasedRlEnv,
  period: float,
  offset: list[float],
  threshold: float,
  command_threshold: float,
  command_name: str,
  sensor_name: str,
) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  current_contact_time = sensor.data.current_contact_time
  assert current_contact_time is not None
  is_contact = current_contact_time > 0.0
  global_phase = ((env.episode_length_buf * env.step_dt) / period).unsqueeze(1)
  offsets = torch.as_tensor(offset, device=env.device, dtype=global_phase.dtype).view(
    1, -1
  )
  is_stance = ((global_phase + offsets) % 1.0) < threshold
  reward = (is_stance == is_contact).float().mean(dim=1)
  command = env.command_manager.get_command(command_name)
  if command is not None:
    total_command = torch.norm(command[:, :2], dim=1) + torch.abs(command[:, 2])
    reward *= (total_command > command_threshold).float()
  return reward


class feet_swing_height:
  """Penalize deviation from target swing height, evaluated at landing."""

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    self.site_names = cfg.params["asset_cfg"].site_names
    self.peak_heights = torch.zeros(
      (env.num_envs, len(self.site_names)), device=env.device, dtype=torch.float32
    )
    self.step_dt = env.step_dt

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    sensor_name: str,
    target_height: float,
    command_name: str,
    command_threshold: float,
    asset_cfg: SceneEntityCfg,
  ) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene[sensor_name]
    command = env.command_manager.get_command(command_name)
    assert command is not None
    foot_heights = asset.data.site_pos_w[:, asset_cfg.site_ids, 2]
    assert contact_sensor.data.found is not None
    in_air = contact_sensor.data.found == 0
    self.peak_heights = torch.where(
      in_air,
      torch.maximum(self.peak_heights, foot_heights),
      self.peak_heights,
    )
    first_contact = contact_sensor.compute_first_contact(dt=self.step_dt)
    total_command = torch.norm(command[:, :2], dim=1) + torch.abs(command[:, 2])
    active = (total_command > command_threshold).float()
    error = self.peak_heights / target_height - 1.0
    cost = torch.sum(torch.square(error) * first_contact.float(), dim=1) * active
    num_landings = torch.sum(first_contact.float())
    peak_heights_at_landing = self.peak_heights * first_contact.float()
    env.extras["log"]["Metrics/peak_height_mean"] = torch.sum(
      peak_heights_at_landing
    ) / torch.clamp(num_landings, min=1)
    self.peak_heights = torch.where(
      first_contact,
      torch.zeros_like(self.peak_heights),
      self.peak_heights,
    )
    return cost


def feet_slip(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  command_name: str,
  command_threshold: float = 0.01,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  contact_sensor: ContactSensor = env.scene[sensor_name]
  command = env.command_manager.get_command(command_name)
  assert command is not None
  total_command = torch.norm(command[:, :2], dim=1) + torch.abs(command[:, 2])
  active = (total_command > command_threshold).float()
  assert contact_sensor.data.found is not None
  in_contact = (contact_sensor.data.found > 0).float()
  foot_vel_xy = asset.data.site_lin_vel_w[:, asset_cfg.site_ids, :2]
  vel_xy_norm = torch.norm(foot_vel_xy, dim=-1)
  cost = torch.sum(torch.square(vel_xy_norm) * in_contact, dim=1) * active
  num_in_contact = torch.sum(in_contact)
  env.extras["log"]["Metrics/slip_velocity_mean"] = torch.sum(
    vel_xy_norm * in_contact
  ) / torch.clamp(num_in_contact, min=1)
  return cost


def soft_landing(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  command_name: str | None = None,
  command_threshold: float = 0.05,
) -> torch.Tensor:
  contact_sensor: ContactSensor = env.scene[sensor_name]
  assert contact_sensor.data.force is not None
  force_magnitude = torch.norm(contact_sensor.data.force, dim=-1)
  first_contact = contact_sensor.compute_first_contact(dt=env.step_dt)
  landing_impact = force_magnitude * first_contact.float()
  cost = torch.sum(landing_impact, dim=1)
  num_landings = torch.sum(first_contact.float())
  env.extras["log"]["Metrics/landing_force_mean"] = torch.sum(
    landing_impact
  ) / torch.clamp(num_landings, min=1)
  if command_name is not None:
    command = env.command_manager.get_command(command_name)
    if command is not None:
      total_command = torch.norm(command[:, :2], dim=1) + torch.abs(command[:, 2])
      cost *= (total_command > command_threshold).float()
  return cost


class variable_posture:
  """Reward default-posture proximity with speed-dependent tolerance."""

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    asset: Entity = env.scene[cfg.params["asset_cfg"].name]
    default_joint_pos = asset.data.default_joint_pos
    assert default_joint_pos is not None
    self.default_joint_pos = default_joint_pos
    _, joint_names = asset.find_joints(cfg.params["asset_cfg"].joint_names)

    _, _, std_standing = resolve_matching_names_values(
      data=cfg.params["std_standing"],
      list_of_strings=joint_names,
    )
    _, _, std_walking = resolve_matching_names_values(
      data=cfg.params["std_walking"],
      list_of_strings=joint_names,
    )
    _, _, std_running = resolve_matching_names_values(
      data=cfg.params["std_running"],
      list_of_strings=joint_names,
    )
    self.std_standing = torch.tensor(
      std_standing, device=env.device, dtype=torch.float32
    )
    self.std_walking = torch.tensor(std_walking, device=env.device, dtype=torch.float32)
    self.std_running = torch.tensor(std_running, device=env.device, dtype=torch.float32)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    std_standing,
    std_walking,
    std_running,
    asset_cfg: SceneEntityCfg,
    command_name: str,
    walking_threshold: float = 0.5,
    running_threshold: float = 1.5,
  ) -> torch.Tensor:
    del std_standing, std_walking, std_running
    asset: Entity = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    assert command is not None
    total_speed = torch.norm(command[:, :2], dim=1) + torch.abs(command[:, 2])
    standing_mask = (total_speed < walking_threshold).float()
    walking_mask = (
      (total_speed >= walking_threshold) & (total_speed < running_threshold)
    ).float()
    running_mask = (total_speed >= running_threshold).float()
    std = (
      self.std_standing * standing_mask.unsqueeze(1)
      + self.std_walking * walking_mask.unsqueeze(1)
      + self.std_running * running_mask.unsqueeze(1)
    )
    current_joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    desired_joint_pos = self.default_joint_pos[:, asset_cfg.joint_ids]
    error_squared = torch.square(current_joint_pos - desired_joint_pos)
    return torch.exp(-torch.mean(error_squared / (std**2), dim=1))


def stand_still(
  env: ManagerBasedRlEnv,
  command_name: str,
  command_threshold: float = 0.1,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  default_joint_pos = asset.data.default_joint_pos
  assert default_joint_pos is not None
  diff_angle = asset.data.joint_pos[:, asset_cfg.joint_ids] - default_joint_pos[
    :, asset_cfg.joint_ids
  ]
  cost = torch.sum(torch.square(diff_angle), dim=1)
  command = env.command_manager.get_command(command_name)
  if command is not None:
    total_command = torch.norm(command[:, :2], dim=1) + torch.abs(command[:, 2])
    cost *= (total_command <= command_threshold).float()
  return cost


def time_out(env: ManagerBasedRlEnv) -> torch.Tensor:
  return env.episode_length_buf >= env.max_episode_length


def bad_orientation(
  env: ManagerBasedRlEnv,
  limit_angle: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  return torch.acos(-asset.data.projected_gravity_b[:, 2]).abs() > limit_angle


def illegal_contact(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  force_threshold: float = 10.0,
) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  data = sensor.data
  if data.force_history is not None:
    force_mag = torch.norm(data.force_history, dim=-1)
    return (force_mag > force_threshold).any(dim=-1).any(dim=-1)
  assert data.found is not None
  return torch.any(data.found, dim=-1)


class VelocityStage(TypedDict):
  step: int
  lin_vel_x: tuple[float, float] | None
  lin_vel_y: tuple[float, float] | None
  ang_vel_z: tuple[float, float] | None


def commands_vel(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  command_name: str,
  velocity_stages: list[VelocityStage],
) -> dict[str, torch.Tensor]:
  del env_ids
  command_term = env.command_manager.get_term(command_name)
  assert command_term is not None
  cfg = cast(UniformVelocityCommandCfg, command_term.cfg)
  for stage in velocity_stages:
    if env.common_step_counter > stage["step"]:
      if stage["lin_vel_x"] is not None:
        cfg.ranges.lin_vel_x = stage["lin_vel_x"]
      if stage["lin_vel_y"] is not None:
        cfg.ranges.lin_vel_y = stage["lin_vel_y"]
      if stage["ang_vel_z"] is not None:
        cfg.ranges.ang_vel_z = stage["ang_vel_z"]
  return {}


class UniformVelocityCommand(CommandTerm):
  cfg: UniformVelocityCommandCfg

  def __init__(self, cfg: UniformVelocityCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)

    if self.cfg.heading_command and self.cfg.ranges.heading is None:
      raise ValueError("heading_command=True but ranges.heading is set to None.")
    if self.cfg.ranges.heading and not self.cfg.heading_command:
      raise ValueError("ranges.heading is set but heading_command=False.")

    self.robot: Entity = env.scene[cfg.entity_name]

    self.vel_command_b = torch.zeros(self.num_envs, 3, device=self.device)
    self.heading_target = torch.zeros(self.num_envs, device=self.device)
    self.heading_error = torch.zeros(self.num_envs, device=self.device)
    self.is_heading_env = torch.zeros(
      self.num_envs, dtype=torch.bool, device=self.device
    )
    self.is_standing_env = torch.zeros_like(self.is_heading_env)

    self.metrics["error_vel_xy"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_vel_yaw"] = torch.zeros(self.num_envs, device=self.device)

    self._joystick_enabled: viser.GuiCheckboxHandle | None = None
    self._joystick_sliders: list[viser.GuiSliderHandle] = []
    self._joystick_get_env_idx: Callable[[], int] | None = None

  @property
  def command(self) -> torch.Tensor:
    return self.vel_command_b

  def _update_metrics(self) -> None:
    max_command_time = self.cfg.resampling_time_range[1]
    max_command_step = max_command_time / self._env.step_dt
    self.metrics["error_vel_xy"] += (
      torch.norm(
        self.vel_command_b[:, :2] - self.robot.data.root_link_lin_vel_b[:, :2], dim=-1
      )
      / max_command_step
    )
    self.metrics["error_vel_yaw"] += (
      torch.abs(self.vel_command_b[:, 2] - self.robot.data.root_link_ang_vel_b[:, 2])
      / max_command_step
    )

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    r = torch.empty(len(env_ids), device=self.device)
    self.vel_command_b[env_ids, 0] = r.uniform_(*self.cfg.ranges.lin_vel_x)
    self.vel_command_b[env_ids, 1] = r.uniform_(*self.cfg.ranges.lin_vel_y)
    self.vel_command_b[env_ids, 2] = r.uniform_(*self.cfg.ranges.ang_vel_z)
    active = torch.norm(self.vel_command_b[env_ids, :], dim=1) > 0.1
    self.vel_command_b[env_ids, :] *= active.unsqueeze(1)
    if self.cfg.heading_command:
      assert self.cfg.ranges.heading is not None
      self.heading_target[env_ids] = r.uniform_(*self.cfg.ranges.heading)
      self.is_heading_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_heading_envs
    self.is_standing_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_standing_envs

    init_vel_mask = r.uniform_(0.0, 1.0) < self.cfg.init_velocity_prob
    init_vel_env_ids = env_ids[init_vel_mask]
    if len(init_vel_env_ids) > 0:
      root_pos = self.robot.data.root_link_pos_w[init_vel_env_ids]
      root_quat = self.robot.data.root_link_quat_w[init_vel_env_ids]
      lin_vel_b = self.robot.data.root_link_lin_vel_b[init_vel_env_ids]
      lin_vel_b[:, :2] = self.vel_command_b[init_vel_env_ids, :2]
      root_lin_vel_w = quat_apply(root_quat, lin_vel_b)
      root_ang_vel_b = self.robot.data.root_link_ang_vel_b[init_vel_env_ids]
      root_ang_vel_b[:, 2] = self.vel_command_b[init_vel_env_ids, 2]
      root_state = torch.cat(
        [root_pos, root_quat, root_lin_vel_w, root_ang_vel_b], dim=-1
      )
      self.robot.write_root_state_to_sim(root_state, init_vel_env_ids)

  def _update_command(self) -> None:
    if self.cfg.heading_command:
      self.heading_error = wrap_to_pi(self.heading_target - self.robot.data.heading_w)
      env_ids = self.is_heading_env.nonzero(as_tuple=False).flatten()
      self.vel_command_b[env_ids, 2] = torch.clip(
        self.cfg.heading_control_stiffness * self.heading_error[env_ids],
        min=self.cfg.ranges.ang_vel_z[0],
        max=self.cfg.ranges.ang_vel_z[1],
      )
    standing_env_ids = self.is_standing_env.nonzero(as_tuple=False).flatten()
    self.vel_command_b[standing_env_ids, :] = 0.0

  def create_gui(
    self,
    name: str,
    server: "viser.ViserServer",
    get_env_idx: Callable[[], int],
  ) -> None:
    from viser import Icon

    ranges = self.cfg.ranges
    axes = [
      ("lin_vel_x", ranges.lin_vel_x[1]),
      ("lin_vel_y", ranges.lin_vel_y[1]),
      ("ang_vel_z", ranges.ang_vel_z[1]),
    ]
    sliders: list = []

    with server.gui.add_folder(name.capitalize()):
      enabled = server.gui.add_checkbox("Enable", initial_value=False)

      for label, max_val in axes:
        max_input = server.gui.add_slider(
          f"Max {label}",
          initial_value=max_val,
          step=0.1,
          min=0.1,
          max=10.0,
        )
        slider = server.gui.add_slider(
          label,
          min=-max_val,
          max=max_val,
          step=0.05,
          initial_value=0.0,
        )

        @max_input.on_update
        def _(_ev, _s=slider, _m=max_input) -> None:
          _s.min = -_m.value
          _s.max = _m.value

        sliders.append(slider)

      zero_btn = server.gui.add_button("Zero", icon=Icon.SQUARE_X)

      @zero_btn.on_click
      def _(_) -> None:
        for slider in sliders:
          slider.value = 0.0

    self._joystick_enabled = enabled
    self._joystick_sliders = sliders
    self._joystick_get_env_idx = get_env_idx

  def compute(self, dt: float) -> None:
    super().compute(dt)
    if self._joystick_enabled is not None and self._joystick_enabled.value:
      assert self._joystick_get_env_idx is not None
      idx = self._joystick_get_env_idx()
      for i, slider in enumerate(self._joystick_sliders):
        self.vel_command_b[idx, i] = slider.value

  def _debug_vis_impl(self, visualizer: "DebugVisualizer") -> None:
    env_indices = visualizer.get_env_indices(self.num_envs)
    if not env_indices:
      return

    cmds = self.command.cpu().numpy()
    base_pos_ws = self.robot.data.root_link_pos_w.cpu().numpy()
    base_mat_ws = matrix_from_quat(self.robot.data.root_link_quat_w).cpu().numpy()
    lin_vel_bs = self.robot.data.root_link_lin_vel_b.cpu().numpy()
    ang_vel_bs = self.robot.data.root_link_ang_vel_b.cpu().numpy()
    scale = self.cfg.viz.scale
    z_offset = self.cfg.viz.z_offset

    for batch in env_indices:
      base_pos_w = base_pos_ws[batch]
      if np.linalg.norm(base_pos_w) < 1e-6:
        continue
      base_mat_w = base_mat_ws[batch]

      def local_to_world(vec: np.ndarray) -> np.ndarray:
        return base_pos_w + base_mat_w @ vec

      cmd = cmds[batch]
      lin_vel_b = lin_vel_bs[batch]
      ang_vel_b = ang_vel_bs[batch]
      origin = np.array([0, 0, z_offset]) * scale
      visualizer.add_arrow(
        local_to_world(origin),
        local_to_world(origin + np.array([cmd[0], cmd[1], 0]) * scale),
        color=(0.2, 0.2, 0.6, 0.6),
        width=0.015,
      )
      visualizer.add_arrow(
        local_to_world(origin),
        local_to_world(origin + np.array([0, 0, cmd[2]]) * scale),
        color=(0.2, 0.6, 0.2, 0.6),
        width=0.015,
      )
      visualizer.add_arrow(
        local_to_world(origin),
        local_to_world(origin + np.array([lin_vel_b[0], lin_vel_b[1], 0]) * scale),
        color=(0.0, 0.6, 1.0, 0.7),
        width=0.015,
      )
      visualizer.add_arrow(
        local_to_world(origin),
        local_to_world(origin + np.array([0, 0, ang_vel_b[2]]) * scale),
        color=(0.0, 1.0, 0.4, 0.7),
        width=0.015,
      )


@dataclass(kw_only=True)
class UniformVelocityCommandCfg(CommandTermCfg):
  entity_name: str
  heading_command: bool = False
  heading_control_stiffness: float = 1.0
  rel_standing_envs: float = 0.0
  rel_heading_envs: float = 1.0
  init_velocity_prob: float = 0.0

  @dataclass
  class Ranges:
    lin_vel_x: tuple[float, float]
    lin_vel_y: tuple[float, float]
    ang_vel_z: tuple[float, float]
    heading: tuple[float, float] | None = None

  ranges: Ranges

  @dataclass
  class VizCfg:
    z_offset: float = 0.2
    scale: float = 0.5

  viz: VizCfg = field(default_factory=VizCfg)

  def build(self, env: ManagerBasedRlEnv) -> UniformVelocityCommand:
    return UniformVelocityCommand(self, env)

  def __post_init__(self):
    if self.heading_command and self.ranges.heading is None:
      raise ValueError(
        "The velocity command has heading commands active (heading_command=True) but "
        "the `ranges.heading` parameter is set to None."
      )
