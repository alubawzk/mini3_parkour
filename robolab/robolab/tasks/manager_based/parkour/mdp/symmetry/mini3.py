# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# Copyright (c) 2025-2026, The RoboLab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Left-right symmetry functions for the Mini3 21-DoF robot in AMP/parkour training.

Mirrors the structure of :mod:`...symmetry.rpo`, but the joint left/right pairing and
sign-flip set are built **programmatically from ``robot.joint_names``** so they stay
correct regardless of the IsaacSim joint ordering (which is *not* the URDF order).

Mirror convention (standard sagittal-plane reflection):
  * a ``left_X`` joint swaps with the matching ``right_X`` joint;
  * joints acting about the roll (forward) or yaw (up) axis are negated after the swap;
  * pitch joints (about the left-right axis) are *not* negated.
So the negate set is exactly the joints whose name contains ``roll`` or ``yaw``
(e.g. ``waist_yaw`` — which has no left/right pair — is still negated).

The observation groups use ``concatenate_terms=False`` with ``history_length`` flattened
into the feature dim, so each group is a nested ``TensorDict`` and the joint terms have
shape ``(B, history * num_joints)`` — handled by reshaping to ``(..., -1, num_joints)``.
"""

from __future__ import annotations

import torch
from tensordict import TensorDict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

__all__ = ["compute_symmetric_states"]

# Lazily-built joint maps (depend on the running articulation's joint order).
_LEFT: list[int] | None = None
_RIGHT: list[int] | None = None
_NEGATE: list[int] | None = None
_NUM_JOINTS: int | None = None


def _ensure_joint_maps(env: ManagerBasedRLEnv) -> None:
    """Build left/right pairing and the roll/yaw negate set from ``robot.joint_names``."""
    global _LEFT, _RIGHT, _NEGATE, _NUM_JOINTS
    if _LEFT is not None:
        return
    joint_names = env.unwrapped.scene["robot"].joint_names
    name_to_idx = {name: i for i, name in enumerate(joint_names)}

    left, right = [], []
    for name, idx in name_to_idx.items():
        if name.startswith("left_"):
            mirror = "right_" + name[len("left_") :]
            if mirror not in name_to_idx:
                raise ValueError(f"No right-side counterpart for joint '{name}' when building Mini3 symmetry maps.")
            left.append(idx)
            right.append(name_to_idx[mirror])

    negate = [idx for name, idx in name_to_idx.items() if ("roll" in name) or ("yaw" in name)]

    _LEFT, _RIGHT, _NEGATE, _NUM_JOINTS = left, right, negate, len(joint_names)


@torch.no_grad()
def compute_symmetric_states(
    env: ManagerBasedRLEnv,
    obs: TensorDict | None = None,
    actions: torch.Tensor | None = None,
):
    """Augment observations and actions with the left-right symmetry transformation."""
    _ensure_joint_maps(env)

    if obs is not None:
        batch_size = obs.batch_size[0]
        # 2 symmetries (original + left-right): double the batch.
        obs_aug = obs.repeat(2)
        obs_aug["policy"][:batch_size] = obs["policy"][:]
        obs_aug["policy"][batch_size : 2 * batch_size] = _transform_policy_obs_left_right(obs["policy"])
        obs_aug["critic"][:batch_size] = obs["critic"][:]
        obs_aug["critic"][batch_size : 2 * batch_size] = _transform_critic_obs_left_right(env, obs["critic"])
    else:
        obs_aug = None

    if actions is not None:
        batch_size = actions.shape[0]
        actions_aug = torch.zeros(batch_size * 2, actions.shape[1], device=actions.device)
        actions_aug[:batch_size] = actions[:]
        # actions are single-step (B, num_joints): swap directly, no history reshape.
        actions_aug[batch_size : 2 * batch_size] = _switch_joints_left_right(actions)
    else:
        actions_aug = None

    return obs_aug, actions_aug


# ---------------------------------------------------------------------------
# Observation transformers
# ---------------------------------------------------------------------------


def _transform_policy_obs_left_right(obs: TensorDict) -> TensorDict:
    """Left-right mirror for the policy observation group (concatenate_terms=False)."""
    obs = obs.clone()
    obs["base_ang_vel"] = _apply_xyz_sign(obs["base_ang_vel"], [-1, 1, -1])
    obs["projected_gravity"] = _apply_xyz_sign(obs["projected_gravity"], [1, -1, 1])
    obs["velocity_commands"] = _apply_xyz_sign(obs["velocity_commands"], [1, -1, -1])
    obs["joint_pos"] = _switch_joints_left_right_flat(obs["joint_pos"])
    obs["joint_vel"] = _switch_joints_left_right_flat(obs["joint_vel"])
    obs["actions"] = _switch_joints_left_right_flat(obs["actions"])
    obs["depth_image"] = _transform_depth_obs_left_right(obs["depth_image"])
    return obs


def _transform_critic_obs_left_right(env: ManagerBasedRLEnv, obs: TensorDict) -> TensorDict:
    """Left-right mirror for the critic observation group (concatenate_terms=False)."""
    obs = obs.clone()
    obs["base_lin_vel"] = _apply_xyz_sign(obs["base_lin_vel"], [1, -1, 1])
    obs["base_ang_vel"] = _apply_xyz_sign(obs["base_ang_vel"], [-1, 1, -1])
    obs["projected_gravity"] = _apply_xyz_sign(obs["projected_gravity"], [1, -1, 1])
    obs["velocity_commands"] = _apply_xyz_sign(obs["velocity_commands"], [1, -1, -1])
    obs["joint_pos"] = _switch_joints_left_right_flat(obs["joint_pos"])
    obs["joint_vel"] = _switch_joints_left_right_flat(obs["joint_vel"])
    obs["actions"] = _switch_joints_left_right_flat(obs["actions"])
    obs["depth_image"] = _transform_depth_obs_left_right(obs["depth_image"])
    if "height_scan" in obs.keys():
        obs["height_scan"] = _transform_height_scan_left_right(env, obs["height_scan"])
    return obs


def _transform_depth_obs_left_right(obs: torch.Tensor) -> torch.Tensor:
    """Mirror a depth image left-right (flip the width axis)."""
    return torch.flip(obs, dims=(-1,))


def _height_scan_left_right_dims(env: ManagerBasedRLEnv) -> tuple[int, int, int]:
    """(history_length, ny, nx) for the height-scan grid, from the running env cfg."""
    cfg = getattr(env, "unwrapped", env).cfg
    pat = cfg.scene.height_scanner.pattern_cfg
    if pat.ordering != "xy":
        raise NotImplementedError(
            "height_scan L-R symmetry only supports GridPatternCfg ordering 'xy';"
            f" got {pat.ordering!r}."
        )
    hist = cfg.observations.critic.height_scan.history_length
    res = float(pat.resolution)
    s0, s1 = float(pat.size[0]), float(pat.size[1])
    # Match isaaclab.sensors.ray_caster.patterns.grid_pattern (same arange endpoints / step).
    nx = int(torch.arange(-s0 / 2, s0 / 2 + 1.0e-9, res).numel())
    ny = int(torch.arange(-s1 / 2, s1 / 2 + 1.0e-9, res).numel())
    return hist, ny, nx


def _transform_height_scan_left_right(env: ManagerBasedRLEnv, hs: torch.Tensor) -> torch.Tensor:
    """Mirror the lateral (y) grid rows of the flattened height-scan history."""
    hist, ny, nx = _height_scan_left_right_dims(env)
    out = hs.view(hs.shape[0], hist, ny, nx).flip(dims=[2])
    return out.reshape(hs.shape)


# ---------------------------------------------------------------------------
# Joint swap helpers
# ---------------------------------------------------------------------------


def _apply_xyz_sign(obs: torch.Tensor, signs: list[int]) -> torch.Tensor:
    obs_shape = obs.shape
    obs = obs.reshape(*obs_shape[:-1], -1, 3)
    obs = obs * torch.tensor(signs, device=obs.device, dtype=obs.dtype)
    return obs.reshape(obs_shape)


def _switch_joints_left_right_flat(joint_data: torch.Tensor) -> torch.Tensor:
    """Swap on a tensor whose last dim is ``history * num_joints``."""
    shape = joint_data.shape
    joint_data = joint_data.reshape(*shape[:-1], -1, _NUM_JOINTS)
    joint_data = _switch_joints_left_right(joint_data)
    return joint_data.reshape(shape)


def _switch_joints_left_right(joint_data: torch.Tensor) -> torch.Tensor:
    """Swap left/right joint values (last dim == num_joints) and negate roll/yaw joints."""
    out = joint_data.clone()
    out[..., _LEFT] = joint_data[..., _RIGHT]
    out[..., _RIGHT] = joint_data[..., _LEFT]
    out[..., _NEGATE] = -out[..., _NEGATE]
    return out
