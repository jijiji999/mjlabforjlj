"""Task-private MDP terms for G1 lower-body locomotion."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.string import resolve_matching_names_values

if TYPE_CHECKING:
  import viser

  from mjlab.envs import ManagerBasedRlEnv


class UniformHeightCommand(CommandTerm):
  """Sample an absolute pelvis-height command on flat ground."""

  cfg: "UniformHeightCommandCfg"

  def __init__(self, cfg: "UniformHeightCommandCfg", env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    self._robot: Entity = env.scene[cfg.entity_name]
    self._sampled_height = torch.full(
      (self.num_envs, 1), cfg.default_height, device=self.device
    )
    self._height_command = self._sampled_height.clone()
    self.metrics["error_height"] = torch.zeros(self.num_envs, device=self.device)

    self._joystick_enabled: viser.GuiCheckboxHandle | None = None
    self._height_slider: viser.GuiSliderHandle | None = None
    self._joystick_get_env_idx: Callable[[], int] | None = None

  @property
  def command(self) -> torch.Tensor:
    return self._height_command

  def _update_metrics(self) -> None:
    max_command_steps = self.cfg.resampling_time_range[1] / self._env.step_dt
    height_error = torch.abs(
      self._height_command[:, 0] - self._robot.data.root_link_pos_w[:, 2]
    )
    self.metrics["error_height"] += height_error / max_command_steps

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    samples = torch.empty(len(env_ids), device=self.device)
    self._sampled_height[env_ids, 0] = samples.uniform_(*self.cfg.height_range)
    self._height_command[env_ids] = self._sampled_height[env_ids]

  def _update_command(self) -> None:
    self._height_command[:] = self._sampled_height
    if self.cfg.moving_command_name is not None:
      velocity_command = self._env.command_manager.get_command(
        self.cfg.moving_command_name
      )
      standing = torch.linalg.vector_norm(velocity_command, dim=1) <= 0.1
      self._height_command[standing, 0] = self.cfg.default_height

  def create_gui(
    self,
    name: str,
    server: "viser.ViserServer",
    get_env_idx: Callable[[], int],
  ) -> None:
    """Add an interactive height slider to the play viewer."""
    with server.gui.add_folder(name.capitalize()):
      enabled = server.gui.add_checkbox("Enable", initial_value=False)
      slider = server.gui.add_slider(
        "pelvis_height",
        min=self.cfg.height_range[0],
        max=self.cfg.height_range[1],
        step=0.01,
        initial_value=self.cfg.default_height,
      )
    self._joystick_enabled = enabled
    self._height_slider = slider
    self._joystick_get_env_idx = get_env_idx

  def compute(self, dt: float) -> None:
    super().compute(dt)
    if self._joystick_enabled is not None and self._joystick_enabled.value:
      assert self._height_slider is not None
      assert self._joystick_get_env_idx is not None
      env_idx = self._joystick_get_env_idx()
      self._sampled_height[env_idx, 0] = self._height_slider.value
      self._update_command()


@dataclass(kw_only=True)
class UniformHeightCommandCfg(CommandTermCfg):
  """Configuration for :class:`UniformHeightCommand`."""

  entity_name: str
  height_range: tuple[float, float]
  default_height: float
  moving_command_name: str | None = None

  def __post_init__(self) -> None:
    height_min, height_max = self.height_range
    if height_min > height_max:
      raise ValueError("height_range minimum must be <= maximum.")
    if not height_min <= self.default_height <= height_max:
      raise ValueError("default_height must be inside height_range.")

  def build(self, env: ManagerBasedRlEnv) -> UniformHeightCommand:
    return UniformHeightCommand(self, env)


def velocity_height_commands(
  env: ManagerBasedRlEnv,
  velocity_command_name: str,
  height_command_name: str,
) -> torch.Tensor:
  """Concatenate the 3-D velocity command and 1-D height command."""
  velocity = env.command_manager.get_command(velocity_command_name)
  height = env.command_manager.get_command(height_command_name)
  return torch.cat((velocity, height), dim=1)


def track_pelvis_height(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """Reward tracking absolute pelvis height above flat ground."""
  if std <= 0.0:
    raise ValueError("Height tracking std must be positive.")
  asset: Entity = env.scene[asset_cfg.name]
  height_command = env.command_manager.get_command(command_name)[:, 0]
  height_error = height_command - asset.data.root_link_pos_w[:, 2]
  return torch.exp(-torch.square(height_error) / std**2)


class SmoothRandomJointPositionTargets:
  """Drive non-policy joints through smooth random targets around zero.

  Each environment follows independent cubic smooth-step segments.  Segment
  endpoints and durations are resampled independently, which produces a
  continuous, bounded target signal without exposing these joints as policy
  actions.
  """

  def __init__(self, cfg: EventTermCfg, env: ManagerBasedRlEnv):
    asset_cfg = cfg.params["asset_cfg"]
    assert isinstance(asset_cfg, SceneEntityCfg)
    assert isinstance(asset_cfg.joint_ids, list)

    self._asset: Entity = env.scene[asset_cfg.name]
    self._device = env.device
    self._joint_ids = torch.tensor(
      asset_cfg.joint_ids, device=env.device, dtype=torch.long
    )
    joint_names = [self._asset.joint_names[i] for i in asset_cfg.joint_ids]

    amplitude = cfg.params["amplitude"]
    if isinstance(amplitude, dict):
      _, _, amplitude_values = resolve_matching_names_values(amplitude, joint_names)
      amplitude_tensor = torch.tensor(
        amplitude_values, device=env.device, dtype=torch.float32
      )
    else:
      amplitude_tensor = torch.full(
        (len(joint_names),), float(amplitude), device=env.device
      )
    if torch.any(amplitude_tensor < 0.0):
      raise ValueError("Random joint target amplitudes must be non-negative.")
    self._amplitude = amplitude_tensor.unsqueeze(0)

    self._duration_range = cfg.params["duration_range_s"]
    duration_min, duration_max = self._duration_range
    if duration_min <= 0.0 or duration_min > duration_max:
      raise ValueError(
        "duration_range_s must contain positive, increasing bounds."
      )

    shape = (env.num_envs, len(joint_names))
    self._segment_start = torch.zeros(shape, device=env.device)
    self._segment_goal = torch.zeros_like(self._segment_start)
    self._elapsed = torch.zeros((env.num_envs, 1), device=env.device)
    self._duration = torch.ones_like(self._elapsed)

    # Keep targets valid even if a caller steps before the first explicit reset.
    self.reset()

  def _sample_goals(self, count: int) -> torch.Tensor:
    samples = torch.rand((count, self._joint_ids.numel()), device=self._device)
    return (2.0 * samples - 1.0) * self._amplitude

  def _sample_durations(self, count: int) -> torch.Tensor:
    duration_min, duration_max = self._duration_range
    return (
      torch.rand((count, 1), device=self._device)
      * (duration_max - duration_min)
      + duration_min
    )

  def _write_targets(
    self,
    targets: torch.Tensor,
    env_ids: torch.Tensor | slice | None = None,
  ) -> None:
    encoder_bias = self._asset.data.encoder_bias
    if env_ids is None:
      env_ids = slice(None)
    biased_targets = targets - encoder_bias[env_ids][:, self._joint_ids]
    if isinstance(env_ids, torch.Tensor):
      # Entity.set_joint_position_target uses paired advanced indexing when
      # both axes are tensors.  Use an explicit outer product for reset subsets.
      self._asset.data.joint_pos_target[
        env_ids[:, None], self._joint_ids[None, :]
      ] = biased_targets
    else:
      self._asset.set_joint_position_target(
        biased_targets, joint_ids=self._joint_ids, env_ids=env_ids
      )

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    count = self._segment_start[env_ids].shape[0]
    self._segment_start[env_ids] = 0.0
    self._segment_goal[env_ids] = self._sample_goals(count)
    self._elapsed[env_ids] = 0.0
    self._duration[env_ids] = self._sample_durations(count)
    self._write_targets(torch.zeros_like(self._segment_start[env_ids]), env_ids)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    amplitude: float | dict[str, float],
    duration_range_s: tuple[float, float],
  ) -> None:
    del env_ids, asset_cfg, amplitude, duration_range_s  # Cached at construction.

    self._elapsed += env.step_dt
    phase = torch.clamp(self._elapsed / self._duration, 0.0, 1.0)
    blend = phase * phase * (3.0 - 2.0 * phase)
    targets = self._segment_start + blend * (
      self._segment_goal - self._segment_start
    )
    self._write_targets(targets)

    finished = (self._elapsed[:, 0] >= self._duration[:, 0]).nonzero().flatten()
    if len(finished) > 0:
      self._segment_start[finished] = self._segment_goal[finished]
      self._segment_goal[finished] = self._sample_goals(len(finished))
      self._elapsed[finished] = 0.0
      self._duration[finished] = self._sample_durations(len(finished))
