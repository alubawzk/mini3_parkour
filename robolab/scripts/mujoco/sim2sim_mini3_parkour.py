# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# Copyright (c) 2025-2026, The RoboLab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# MuJoCo sim2sim entry point for MINI3 parkour policies exported as separate
# ONNX graphs (depth encoder + actor). The shared depth/ONNX/viewer loop is
# reused from sim2sim_rpo_parkour.py; this file provides MINI3-specific robot
# parameters, sensor names, joint ordering, and default scenes.

from __future__ import annotations

import argparse
import os
import sys
import types
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

REPO_ROOT = Path(__file__).resolve().parents[3]
ROBOLAB_SRC = REPO_ROOT / "robolab"
if str(ROBOLAB_SRC) not in sys.path:
    sys.path.insert(0, str(ROBOLAB_SRC))

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

try:
    import tqdm  # noqa: F401
except ImportError:
    tqdm_stub = types.ModuleType("tqdm")
    tqdm_stub.tqdm = lambda iterable, *args, **kwargs: iterable
    sys.modules["tqdm"] = tqdm_stub

try:
    from pynput import keyboard as _keyboard_probe  # noqa: F401
except Exception:
    pynput_stub = types.ModuleType("pynput")
    keyboard_stub = types.ModuleType("pynput.keyboard")

    class _NoopListener:
        def __init__(self, *args, **kwargs):
            pass

        def start(self):
            return self

        def stop(self):
            pass

    keyboard_stub.Listener = _NoopListener
    pynput_stub.keyboard = keyboard_stub
    sys.modules["pynput"] = pynput_stub
    sys.modules["pynput.keyboard"] = keyboard_stub

import sim2sim_rpo_parkour as parkour_sim
from robolab.assets import ISAAC_DATA_DIR


# MINI3 MJCF actuator/qpos order (after the free joint), from mini3.xml.
MJCF_JOINT_ORDER: tuple[str, ...] = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_pitch_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_pitch_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_pitch_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_pitch_joint",
)

# Isaac action/observation order for MINI3, following the same converted-articulation
# ordering pattern as RPO but with MINI3 joint names and 21 DoF.
POLICY_JOINT_ORDER: tuple[str, ...] = (
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "waist_yaw_joint",
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint",
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "left_shoulder_roll_joint",
    "right_shoulder_roll_joint",
    "left_knee_pitch_joint",
    "right_knee_pitch_joint",
    "left_shoulder_yaw_joint",
    "right_shoulder_yaw_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "left_elbow_pitch_joint",
    "right_elbow_pitch_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
)

USD2URDF = [MJCF_JOINT_ORDER.index(name) for name in POLICY_JOINT_ORDER]

# MINI3 MJCF sensors. framequat is wxyz; frameangvel is world-frame, so it is
# rotated into base frame below to match Isaac's base_ang_vel observation term.
_ORIENTATION_SENSOR = "base_link_site_quat"
_ANGULAR_VELOCITY_SENSOR = "base_link_site_angvel"


def get_obs(data, model):
    """MINI3 articulation observation from MuJoCo, aligned with parkour policy terms."""
    del model
    q = data.qpos.astype(np.double)
    dq = data.qvel.astype(np.double)

    quat_wxyz = data.sensor(_ORIENTATION_SENSOR).data.astype(np.double)
    quat_xyzw = quat_wxyz[[1, 2, 3, 0]]
    rot = R.from_quat(quat_xyzw)

    v = rot.apply(data.qvel[:3], inverse=True).astype(np.double)
    omega_world = data.sensor(_ANGULAR_VELOCITY_SENSOR).data.astype(np.double)
    omega = rot.apply(omega_world, inverse=True).astype(np.double)
    gvec = rot.apply(np.array([0.0, 0.0, -1.0]), inverse=True).astype(np.double)
    return q, dq, quat_xyzw, v, omega, gvec


def latest_mini3_export_dir() -> Path:
    """Return the newest local MINI3 export with actor + depth encoder, or a stable placeholder."""
    root = REPO_ROOT / "logs" / "rsl_rl" / "mini3_parkour"
    candidates = sorted(
        p.parent
        for p in root.glob("*/exported/actor.onnx")
        if (p.parent / "0-depth_encoder.onnx").is_file()
    )
    if candidates:
        return candidates[-1]
    return root / "2026-06-22_18-25-14_Revise_HorizontalScale_AddVelTrack" / "exported"


def configure_shared_loop_for_mini3() -> None:
    """Patch the shared RPO parkour loop with MINI3-specific observation hooks."""
    parkour_sim.get_obs = get_obs
    parkour_sim.cmd.hold_vx = 0.8
    parkour_sim.cmd.hold_vy = 0.0
    parkour_sim.cmd.hold_dyaw = 1.0
    parkour_sim.cmd.ramp_vx_per_s = 2.0
    parkour_sim.cmd.ramp_vy_per_s = 0.0
    parkour_sim.cmd.ramp_dyaw_per_s = 3.0
    parkour_sim.cmd.reset()
    parkour_sim.cmd.camera_follow = True
    parkour_sim.cmd.camera_mode = parkour_sim.CameraMode.ORBIT


def parse_args() -> argparse.Namespace:
    default_export = latest_mini3_export_dir()
    parser = argparse.ArgumentParser(description="MINI3 parkour sim2sim (depth_encoder.onnx + actor.onnx).")
    parser.add_argument(
        "--depth_encoder",
        type=str,
        default=str(default_export / "0-depth_encoder.onnx"),
        help="Path to depth encoder ONNX.",
    )
    parser.add_argument(
        "--actor",
        type=str,
        default=str(default_export / "actor.onnx"),
        help="Path to actor ONNX (includes obs normalizer if exported with normalization).",
    )
    parser.add_argument(
        "--mujoco_xml",
        type=str,
        default=None,
        help="MJCF path (absolute or relative); if set, overrides --scene.",
    )
    parser.add_argument(
        "--scene",
        type=str,
        choices=("stairs", "plane"),
        default="stairs",
        help="Scene: stairs=mini3_stairs.xml; plane=flat scene.xml.",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        default=False,
        help="Run without GUI; record simulation_parkour.mp4 (depth preview and MuJoCo viewer disabled).",
    )
    parser.add_argument(
        "--no_depth_vis",
        action="store_true",
        default=False,
        help="Do not open OpenCV depth preview (default: one window, metric + encoder side by side).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_shared_loop_for_mini3()

    mjcf_dir = Path(ISAAC_DATA_DIR) / "robots" / "roboparty" / "mini3" / "mjcf"
    scene_xml = {
        "stairs": mjcf_dir / "mini3_stairs.xml",
        "plane": mjcf_dir / "scene.xml",
    }
    xml_path = Path(args.mujoco_xml) if args.mujoco_xml else scene_xml[args.scene]

    class Sim2simCfg:
        class sim_config:
            mujoco_model_path = str(xml_path)
            sim_duration = 1_000_000.0
            dt = 0.002
            decimation = 10
            depth_camera_body = "waist_yaw_link"

        class robot_config:
            kps = np.array(
                [
                    70.0,
                    55.0,
                    25.0,
                    70.0,
                    50.0,
                    45.0,
                    70.0,
                    55.0,
                    25.0,
                    70.0,
                    50.0,
                    45.0,
                    65.0,
                    30.0,
                    25.0,
                    30.0,
                    20.0,
                    30.0,
                    25.0,
                    30.0,
                    20.0,
                ],
                dtype=np.double,
            )
            kds = np.array(
                [
                    4.5,
                    2.8,
                    1.1,
                    4.5,
                    1.0,
                    1.0,
                    4.5,
                    2.8,
                    1.1,
                    4.5,
                    1.0,
                    1.0,
                    3.0,
                    1.0,
                    2.0,
                    1.0,
                    1.0,
                    1.0,
                    2.0,
                    1.0,
                    1.0,
                ],
                dtype=np.double,
            )
            default_pos = np.array(
                [
                    -0.4,
                    0.0,
                    0.0,
                    0.8,
                    -0.4,
                    0.0,
                    -0.4,
                    0.0,
                    0.0,
                    0.8,
                    -0.4,
                    0.0,
                    0.0,
                    0.0,
                    0.25,
                    0.0,
                    1.0,
                    0.0,
                    -0.25,
                    0.0,
                    1.0,
                ],
                dtype=np.double,
            )
            tau_limit = np.array(
                [
                    27.0,
                    27.0,
                    27.0,
                    27.0,
                    27.0,
                    27.0,
                    27.0,
                    27.0,
                    27.0,
                    27.0,
                    27.0,
                    27.0,
                    27.0,
                    12.5,
                    12.5,
                    12.5,
                    12.5,
                    12.5,
                    12.5,
                    12.5,
                    12.5,
                ],
                dtype=np.double,
            )
            num_actions = 21
            action_scale = 0.25
            usd2urdf = USD2URDF

    enc_sess, act_sess = parkour_sim.build_onnx_sessions(
        args.depth_encoder,
        args.actor,
        providers=parkour_sim._SIM2SIM_PERF_ONNX_PROVIDERS,
    )
    parkour_sim.run_mujoco_onnx(
        enc_sess,
        act_sess,
        Sim2simCfg(),
        headless=args.headless,
        debug_obs=parkour_sim._SIM2SIM_PERF_DEBUG_OBS,
        show_depth_vis=not args.no_depth_vis,
        depth_vis_scale=max(1, parkour_sim._SIM2SIM_PERF_DEPTH_VIS_SCALE),
        realtime_sync=parkour_sim._SIM2SIM_PERF_REALTIME_SYNC,
        quiet=parkour_sim._SIM2SIM_PERF_QUIET,
        depth_vis_every_step=parkour_sim._SIM2SIM_PERF_DEPTH_VIS_EVERY_STEP,
        depth_vis_policy_stride=max(1, parkour_sim._SIM2SIM_PERF_DEPTH_VIS_POLICY_STRIDE),
        viewer_sync_every=parkour_sim._SIM2SIM_PERF_VIEWER_SYNC_EVERY,
        viewer_fallback_width=max(320, parkour_sim._SIM2SIM_PERF_VIEWER_FALLBACK_W),
        viewer_fallback_height=max(240, parkour_sim._SIM2SIM_PERF_VIEWER_FALLBACK_H),
    )


if __name__ == "__main__":
    main()
