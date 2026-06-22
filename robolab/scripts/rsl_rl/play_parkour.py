# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# Copyright (c) 2025-2026, The RoboLab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
"""Script to play parkour checkpoints (RSL-RL, AMP). ONNX export uses real observations."""

import argparse
import copy
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
parser.add_argument(
    "--keyboard",
    action="store_true",
    default=False,
    help="Control the base_velocity command with the keyboard. Requires a GUI window.",
)
parser.add_argument("--keyboard_lin_step", type=float, default=0.1, help="Keyboard linear velocity increment in m/s.")
parser.add_argument("--keyboard_ang_step", type=float, default=0.1, help="Keyboard yaw velocity increment in rad/s.")
parser.add_argument("--keyboard_max_lin_vel", type=float, default=1.0, help="Keyboard linear velocity clamp in m/s.")
parser.add_argument("--keyboard_max_ang_vel", type=float, default=1.5, help="Keyboard yaw velocity clamp in rad/s.")
parser.add_argument(
    "--terrain_type",
    type=str,
    default=None,
    help="Use only one configured terrain type during play, e.g. pyramid_stairs_32_random.",
)
parser.add_argument(
    "--terrain_difficulty",
    type=float,
    default=None,
    help="Fixed terrain difficulty in [0, 1] when --terrain_type is set.",
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


class ParkourKeyboardController:
    """Keyboard override for the manager-based parkour base_velocity command."""

    def __init__(
        self,
        env,
        command_name: str = "base_velocity",
        lin_vel_step: float = 0.1,
        ang_vel_step: float = 0.1,
        max_lin_vel: float = 1.0,
        max_ang_vel: float = 1.5,
    ):
        self.env = env.unwrapped
        self.command_name = command_name
        self.lin_vel_step = lin_vel_step
        self.ang_vel_step = ang_vel_step
        self.max_lin_vel = max_lin_vel
        self.max_ang_vel = max_ang_vel
        self.command = torch.zeros(3, device=self.env.device)

        self._term = self.env.command_manager.get_term(command_name)
        self._original_update_command = None

        import carb
        import omni.appwindow

        self._carb = carb
        self._appwindow = omni.appwindow.get_default_app_window()
        if self._appwindow is None:
            raise RuntimeError("--keyboard requires an Isaac Sim GUI window. Remove --headless.")
        self._input = carb.input.acquire_input_interface()
        self._keyboard = self._appwindow.get_keyboard()
        self._keyboard_sub = self._input.subscribe_to_keyboard_events(self._keyboard, self._on_keyboard_event)
        self._original_update_command = self._term._update_command
        self._term._update_command = self._manual_update_command

        print("[Keyboard] base_velocity control enabled")
        print("[Keyboard] W/S: vx +/- | A/D: yaw +/- | Q/E: vy +/- | X or Space: stop | R: reset envs")
        self.apply()

    def close(self):
        if getattr(self, "_keyboard_sub", None) is not None:
            self._input.unsubscribe_from_keyboard_events(self._keyboard, self._keyboard_sub)
            self._keyboard_sub = None
        if getattr(self, "_term", None) is not None and getattr(self, "_original_update_command", None) is not None:
            self._term._update_command = self._original_update_command

    def apply(self):
        self._term.vel_command_b[:] = self.command
        if hasattr(self._term, "is_standing_env"):
            self._term.is_standing_env[:] = False

    def _manual_update_command(self):
        self.apply()

    def _on_keyboard_event(self, event, *args, **kwargs):
        event_type = self._carb.input.KeyboardEventType
        if event.type not in (event_type.KEY_PRESS, event_type.KEY_REPEAT):
            return True

        key = event.input.name.upper()
        changed = False
        if key == "W":
            self.command[0] += self.lin_vel_step
            changed = True
        elif key == "S":
            self.command[0] -= self.lin_vel_step
            changed = True
        elif key == "Q":
            self.command[1] += self.lin_vel_step
            changed = True
        elif key == "E":
            self.command[1] -= self.lin_vel_step
            changed = True
        elif key == "A":
            self.command[2] += self.ang_vel_step
            changed = True
        elif key == "D":
            self.command[2] -= self.ang_vel_step
            changed = True
        elif key in ("X", "SPACE", "SPACEBAR"):
            self.command.zero_()
            changed = True
        elif key == "R":
            self.env.episode_length_buf[:] = self.env.max_episode_length
            print("[Keyboard] reset requested")

        if changed:
            self.command[0:2].clamp_(min=-self.max_lin_vel, max=self.max_lin_vel)
            self.command[2].clamp_(min=-self.max_ang_vel, max=self.max_ang_vel)
            self.apply()
            vx, vy, wz = self.command.tolist()
            print(f"[Keyboard] command vx={vx:.2f} m/s, vy={vy:.2f} m/s, wz={wz:.2f} rad/s")
        return True


def select_play_terrain(env_cfg, terrain_type: str | None, terrain_difficulty: float | None = None):
    if terrain_type is None:
        return

    terrain_generator = env_cfg.scene.terrain.terrain_generator
    if terrain_generator is None:
        raise RuntimeError("--terrain_type requires a generated terrain config.")

    available_types = list(terrain_generator.sub_terrains.keys())
    if terrain_type not in terrain_generator.sub_terrains:
        raise ValueError(f"Unknown terrain_type '{terrain_type}'. Available terrain types: {available_types}")

    terrain_generator.sub_terrains = {terrain_type: copy.deepcopy(terrain_generator.sub_terrains[terrain_type])}
    terrain_generator.sub_terrains[terrain_type].proportion = 1.0
    terrain_generator.num_cols = 1

    if terrain_difficulty is not None:
        if not 0.0 <= terrain_difficulty <= 1.0:
            raise ValueError("--terrain_difficulty must be in [0, 1].")
        # Use random generation with a degenerate range to get one fixed difficulty.
        terrain_generator.curriculum = False
        terrain_generator.difficulty_range = (terrain_difficulty, terrain_difficulty)

    if env_cfg.commands.base_velocity.velocity_ranges is not None:
        env_cfg.commands.base_velocity.velocity_ranges = {
            terrain_type: env_cfg.commands.base_velocity.velocity_ranges.get(
                terrain_type, {"lin_vel_x": (0.0, 0.0), "lin_vel_y": (0.0, 0.0), "ang_vel_z": (0.0, 0.0)}
            )
        }
    if env_cfg.commands.base_velocity.random_velocity_terrain is not None:
        env_cfg.commands.base_velocity.random_velocity_terrain = [
            name for name in env_cfg.commands.base_velocity.random_velocity_terrain if name == terrain_type
        ]

    print(
        f"[INFO] Play terrain override: terrain_type={terrain_type}, "
        f"num_rows={terrain_generator.num_rows}, num_cols={terrain_generator.num_cols}, "
        f"difficulty_range={terrain_generator.difficulty_range}, curriculum={terrain_generator.curriculum}"
    )


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

    select_play_terrain(env_cfg, args_cli.terrain_type, args_cli.terrain_difficulty)

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

    keyboard_controller = None
    if args_cli.keyboard:
        if getattr(args_cli, "headless", False):
            raise RuntimeError("--keyboard requires a GUI window. Remove --headless.")
        keyboard_controller = ParkourKeyboardController(
            env,
            lin_vel_step=args_cli.keyboard_lin_step,
            ang_vel_step=args_cli.keyboard_ang_step,
            max_lin_vel=args_cli.keyboard_max_lin_vel,
            max_ang_vel=args_cli.keyboard_max_ang_vel,
        )
        obs = env.get_observations()

    timestep = 0
    try:
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
    finally:
        if keyboard_controller is not None:
            keyboard_controller.close()

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
