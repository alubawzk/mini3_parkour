# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# Copyright (c) 2025-2026, The RoboLab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
"""Script to play parkour checkpoints (RSL-RL, AMP). ONNX export uses real observations."""

import argparse
import os
import sys
import time

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Play parkour RL agent (RSL-RL / AMP).")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during play.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument(
    "--exportonnx",
    action="store_true",
    default=False,
    help="Export EncoderActorCritic policy as separate ONNX files (depth encoder(s) + actor).",
)
parser.add_argument(
    "--draw_camera_fov",
    action="store_true",
    default=False,
    help="Visualize the depth camera FOV: ground ray-hit points + frustum edges (needs a GUI viewport).",
)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
if args_cli.video:
    args_cli.enable_cameras = True

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Check for installed RSL-RL version."""

import importlib.metadata as metadata

from packaging import version

installed_version = metadata.version("rsl-rl-lib")

"""Rest everything follows."""

import gymnasium as gym
import torch

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import isaaclab_tasks  # noqa: F401
import robolab.tasks  # noqa: F401

from rsl_rl.runners import AMPRunner, DistillationRunner, OnPolicyRunner


# Color/size constants for the FOV overlay.
_FOV_POINT_COLOR = (0.1, 0.9, 0.1, 1.0)   # ground ray-hit points: green
_FOV_POINT_SIZE = 6.0
_FOV_EDGE_COLOR = (1.0, 0.55, 0.0, 1.0)   # apex->corner frustum edges: orange
_FOV_QUAD_COLOR = (1.0, 0.9, 0.0, 1.0)    # ground footprint quad: yellow
_FOV_LINE_WIDTH = 2.0


def draw_camera_fov(env, draw, sensor_name="camera", point_stride=3):
    """Draw the depth camera's FOV for every environment using Isaac Sim debug-draw.

    Renders three things from the camera's ray-cast geometry (clean, pre-noise):
      * the ground projection of (a subsample of) the camera rays -- ``ray_hits_w``;
      * the 4 frustum edges from the camera origin to the FOV corners;
      * the ground footprint quad connecting those 4 corner hits.

    The rays are ordered row-major as ``image_shape = (height, width)``; misses come
    back non-finite and are skipped so dangling lines/points are not drawn.
    """
    cam = env.unwrapped.scene.sensors[sensor_name]
    hits = cam.ray_hits_w          # (B, N, 3) world-space ground hits
    starts = cam._ray_starts_w     # (B, N, 3) world-space ray origins (camera center)
    height, width = cam.image_shape
    num_envs = hits.shape[0]

    hits_g = hits.view(num_envs, height, width, 3)
    starts_g = starts.view(num_envs, height, width, 3)
    # corner indices in (row, col): TL, TR, BR, BL
    corner_rc = [(0, 0), (0, width - 1), (height - 1, width - 1), (height - 1, 0)]

    points, point_colors, point_sizes = [], [], []
    line_a, line_b, line_colors, line_widths = [], [], [], []

    for b in range(num_envs):
        # ground projection points (subsampled grid)
        pts = hits_g[b, ::point_stride, ::point_stride].reshape(-1, 3)
        pts = pts[torch.isfinite(pts).all(dim=1)]
        for p in pts.tolist():
            points.append((p[0], p[1], p[2]))
            point_colors.append(_FOV_POINT_COLOR)
            point_sizes.append(_FOV_POINT_SIZE)

        # corner apex/hit pairs, skipping any corner whose ray missed
        corners = []
        for r, c in corner_rc:
            apex = starts_g[b, r, c]
            hit = hits_g[b, r, c]
            corners.append((apex, hit, bool(torch.isfinite(hit).all())))

        # frustum edges: apex -> corner hit
        for apex, hit, ok in corners:
            if not ok:
                continue
            line_a.append(tuple(apex.tolist()))
            line_b.append(tuple(hit.tolist()))
            line_colors.append(_FOV_EDGE_COLOR)
            line_widths.append(_FOV_LINE_WIDTH)

        # ground footprint quad: corner hit -> next corner hit
        for i in range(4):
            _, hit_i, ok_i = corners[i]
            _, hit_j, ok_j = corners[(i + 1) % 4]
            if not (ok_i and ok_j):
                continue
            line_a.append(tuple(hit_i.tolist()))
            line_b.append(tuple(hit_j.tolist()))
            line_colors.append(_FOV_QUAD_COLOR)
            line_widths.append(_FOV_LINE_WIDTH)

    draw.clear_points()
    draw.clear_lines()
    if points:
        draw.draw_points(points, point_colors, point_sizes)
    if line_a:
        draw.draw_lines(line_a, line_b, line_colors, line_widths)


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    # handle deprecated configurations
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)

    # rsl-rl 3.x 的 PPO/PPOAMP 不支持 Isaac Lab 2.3 为 5.0 预留的字段
    if version.parse(installed_version) < version.parse("5.0.0"):
        for key in ("optimizer", "share_cnn_encoders"):
            if hasattr(agent_cfg.algorithm, key):
                delattr(agent_cfg.algorithm, key)

    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)
    env_cfg.log_dir = log_dir

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during play.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "AMPRunner":
        runner = AMPRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path, map_location=agent_cfg.device)

    torch_policy = runner.get_inference_policy(device=env.unwrapped.device)

    try:
        policy_nn = runner.alg.policy
    except AttributeError:
        policy_nn = runner.alg.actor_critic

    export_model_dir = os.path.join(log_dir, "exported")
    # export policy to onnx (separate encoder + actor graphs when policy implements export_as_onnx)
    if agent_cfg.load_run is not None and args_cli.exportonnx:
        assert env.unwrapped.num_envs == 1, "Exporting to ONNX is only supported for single environment."
        if not hasattr(policy_nn, "export_as_onnx"):
            raise AttributeError(
                "export_as_onnx is missing on the policy module; use EncoderActorCritic / EncoderMoEActorCritic "
                "for parkour ONNX export."
            )
        os.makedirs(export_model_dir, exist_ok=True)
        obs = env.get_observations()
        policy_nn.export_as_onnx(obs, export_model_dir)
    else:
        obs = env.get_observations()

    policy = torch_policy
    dt = env.unwrapped.step_dt

    # optional depth-camera FOV overlay (ground hit points + frustum edges)
    draw = None
    if args_cli.draw_camera_fov:
        try:
            from isaacsim.util.debug_draw import _debug_draw

            draw = _debug_draw.acquire_debug_draw_interface()
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] --draw_camera_fov requested but debug-draw is unavailable: {exc}")

    timestep = 0
    while simulation_app.is_running():
        start_time = time.time()
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)
            policy_nn.reset(dones)
        if draw is not None:
            draw_camera_fov(env, draw)
        if args_cli.video:
            timestep += 1
            if timestep == args_cli.video_length:
                break

        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
