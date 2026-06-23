import copy
import os

from isaaclab.utils import configclass

from robolab import ROBOLAB_ROOT_DIR
from robolab.assets.robots.roboparty import MINI3_CFG, MINI3_LINKS
from robolab.sensors import get_link_prim_targets
from robolab.tasks.manager_based.parkour.parkour_env_cfg import ROUGH_TERRAINS_CFG, ParkourEnvCfg

MINI3_CFG.init_state.pos = (0.0, 0.0, 0.5)
AMP_NUM_STEPS = 3


def _scale_terrains_for_mini3(base_cfg):
    """Return a copy of ``base_cfg`` with absolute (metric) terrain dimensions scaled
    down for MINI3, which is ~half the size of RPO (init height 0.43 vs 0.85 m).

    The shared ``ROUGH_TERRAINS_CFG`` is tuned for RPO; reused as-is, every metric
    obstacle (step height, gap width, surface roughness) is ~2x harder relative to
    MINI3's body. Only height/length quantities that set robot-relative difficulty are
    changed here. Left untouched on purpose:
      * ``slope_range``      -- dimensionless ratio, size-independent.
      * ``platform_width`` / ``border_width`` / ``size`` -- scene layout, not difficulty.
      * ``gap_depth``        -- "hole you must not step in"; keep RPO depth.
      * ``step_width``       -- stair tread depth; shrinking it makes stairs *denser*
                                (harder), so keep it well above the ~0.12 m foot
                                sampling range (LEG_VOLUME_POINTS_GRID).
    """

    def _half(noise_scale):
        # noise_scale is a float (nested perlin_cfg) or a [min, max] list (plane terrain);
        # multiply by 0.5 so an intentionally-flat 0.0 stays 0.0.
        if isinstance(noise_scale, (list, tuple)):
            return type(noise_scale)(v * 0.5 for v in noise_scale)
        return noise_scale * 0.5

    cfg = copy.deepcopy(base_cfg)
    # Keep the RPO terrain resolution while debugging MINI3 reset stability. 0.025
    # quadruples the heightfield cells and makes virtual edge extraction much slower.
    cfg.horizontal_scale = 0.05
    # vertical_scale stays 0.005: 0.03-0.10 m steps -> 6-20 height units, already fine.
    for sub in cfg.sub_terrains.values():
        # Stair tread height 0.05-0.20 m (up to half MINI3's leg) -> 0.03-0.10 m.
        if hasattr(sub, "step_height_range"):
            sub.step_height_range = (0.03, 0.10)
        # Gap to stride over 0.1-0.40 m -> 0.05-0.20 m.
        if hasattr(sub, "gap_distance_range"):
            sub.gap_distance_range = (0.05, 0.20)
        # Surface roughness amplitude on the flat/rough plane terrains ([0.0, 0.1] list).
        if hasattr(sub, "noise_scale"):
            sub.noise_scale = _half(sub.noise_scale)
        # Roughness baked on top of stairs/gaps (the slope's intentional 0.0 stays 0.0).
        if getattr(sub, "perlin_cfg", None) is not None:
            sub.perlin_cfg.noise_scale = _half(sub.perlin_cfg.noise_scale)
    return cfg


MINI3_ROUGH_TERRAINS_CFG = _scale_terrains_for_mini3(ROUGH_TERRAINS_CFG)

# Play variant: same scaled terrain but without walls so the robot is easy to observe.
ROUGH_TERRAINS_CFG_PLAY = copy.deepcopy(MINI3_ROUGH_TERRAINS_CFG)
for sub_terrain_name, sub_terrain_cfg in ROUGH_TERRAINS_CFG_PLAY.sub_terrains.items():
    sub_terrain_cfg.wall_prob = [0.0, 0.0, 0.0, 0.0]


@configclass
class MINI3ParkourRoughEnvCfg(ParkourEnvCfg):
    def __post_init__(self):
        ### post init of parent
        super().__post_init__()
        ### Scene
        self.scene.terrain.terrain_generator = MINI3_ROUGH_TERRAINS_CFG
        self.scene.robot = MINI3_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.camera.mesh_prim_paths.extend(get_link_prim_targets(MINI3_LINKS))

        # The base ParkourEnvCfg/SceneCfg is written for the RPO robot, whose body/joint
        # naming differs from MINI3. Remap every RPO-specific name to its MINI3 equivalent
        # so sensors, rewards, terminations and events resolve to real bodies/joints:
        #   *_knee_link    -> *_knee_pitch_link
        #   *_elbow_yaw_link -> *_elbow_pitch_link
        #   torso_link     -> waist_yaw_link      (MINI3 has no torso_link)
        #   torso_joint / *_arm_*_joint -> waist_yaw_joint / *_shoulder_*_joint
        # Scene sensors
        self.scene.knee_volume_points.prim_path = "{ENV_REGEX_NS}/Robot/.*_knee_pitch_link"
        # NOTE: the camera/height-scanner offsets are tuned for the RPO torso_link mount;
        # mounting on waist_yaw_link may need offset re-calibration for MINI3.
        self.scene.camera.prim_path = "{ENV_REGEX_NS}/Robot/waist_yaw_link"
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/waist_yaw_link"
        self.scene.terrain.max_init_terrain_level = 3
        # Discriminator key bodies (order preserved to match the reference motion layout)
        self.observations.disc.key_body_pos_b.params["asset_cfg"].body_names = [
            "left_ankle_roll_link",
            "right_ankle_roll_link",
            "left_knee_pitch_link",
            "right_knee_pitch_link",
            "left_elbow_pitch_link",
            "right_elbow_pitch_link",
        ]

        ### Rewards
        self.rewards.rewards.joint_deviation_upper_body.params["asset_cfg"].joint_names = [
            ".*_shoulder_.*_joint",
            ".*_elbow_.*_joint",
            "waist_yaw_joint",
        ]
        self.rewards.rewards.freeze_upper_torso.params["asset_cfg"].joint_names = ["waist_yaw_joint"]
        self.rewards.rewards.pelvis_orientation_l2.params["asset_cfg"].body_names = "base_link"
        # self.rewards.rewards.termination_penalty.weight  = -500.0
        self.rewards.rewards.track_lin_vel_xy_exp.weight =  15.0
        self.rewards.rewards.track_ang_vel_z_exp.weight  =  15.0
        self.rewards.rewards.feet_stumble.params["sensor_cfg"].body_names = [
            ".*_ankle_roll_link",
            # ".*_knee_pitch_link",
        ]
        self.rewards.rewards.feet_at_plane.params["height_offset"] = 0.028

        ### Terminations
        # self.terminations.terrain_out_bound = None
        # self.terminations.base_contact = None
        self.terminations.base_contact.params["sensor_cfg"].body_names = ["waist_yaw_link"]
        # self.terminations.bad_orientation = None
        # self.terminations.root_height = None
        self.terminations.root_height.params["minimum_height"] = 0.12

        ### Events
        self.events.add_base_mass = None
        self.events.randomize_rigid_body_com = None
        self.events.reset_robot_joints.params["position_range"] = (-0.05, 0.05)
        self.motion_data.motion_dataset.motion_data_dir = os.path.join(
            ROBOLAB_ROOT_DIR, "data", "motions", "mini3_lab"
        )
        self.motion_data.motion_dataset.motion_data_weights = {
            "mini3_36_01_stageii": 1,
            "mini3_36_11_stageii": 1,
            # "mini3_114_08_stageii": 1,
            # "mini3_114_09_stageii": 1,
            "mini3_A1_-_Stand_stageii": 1,
            "mini3_B9_-_walk_turn_left_90_stageii": 1,
            "mini3_B10_-_walk_turn_left_45_stageii": 1,
            "mini3_B13_-_walk_turn_right_45_stageii": 1,
            "mini3_B14_-_walk_turn_right_135_stageii": 1,
            "mini3_B15_-_walk_turn_around_same_direction_stageii": 1,
            "mini3_0007_Walking001_stageii": 1,
        }
        self.animation.animation.num_steps_to_use = AMP_NUM_STEPS
        self.observations.disc.history_length = AMP_NUM_STEPS


class ShoeConfigMixin:
    def apply_shoe_config(self):
        self.scene.robot = MINI3_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.leg_volume_points.points_generator.z_min = -0.063
        self.scene.leg_volume_points.points_generator.z_max = -0.023
        self.rewards.rewards.feet_at_plane.params["height_offset"] = 0.058



@configclass
class MINI3ParkourRoughEnvCfg_PLAY(MINI3ParkourRoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        self.scene.terrain.terrain_generator = ROUGH_TERRAINS_CFG_PLAY
        # make a smaller scene for play
        self.scene.num_envs = 10
        self.scene.env_spacing = 2.5
        self.episode_length_s = 10
        # self.terminations.root_height = None

        # self.commands.base_velocity.velocity_ranges["pyramid_stairs"] = {"lin_vel_x": (1.0, 1.0), "lin_vel_y": (0.0, 0.0), "ang_vel_z": (-1.0, 1.0)}
        # self.commands.base_velocity.velocity_ranges["pyramid_stairs_high"] = {"lin_vel_x": (1.0, 1.0), "lin_vel_y": (0.0, 0.0), "ang_vel_z": (-1.0, 1.0)}
        # self.commands.base_velocity.velocity_ranges["pyramid_stairs_inv"] = {"lin_vel_x": (1.0, 1.0), "lin_vel_y": (0.0, 0.0), "ang_vel_z": (-1.0, 1.0)}
        # self.commands.base_velocity.velocity_ranges["pyramid_stairs_inv_high"] = {"lin_vel_x": (1.0, 1.0), "lin_vel_y": (0.0, 0.0), "ang_vel_z": (-1.0, 1.0)}
        # self.commands.base_velocity.velocity_ranges["pyramid_stairs_inv_high_ground_aligned"] = {"lin_vel_x": (1.0, 1.0), "lin_vel_y": (0.0, 0.0), "ang_vel_z": (-1.0, 1.0)}
        # self.commands.base_velocity.velocity_ranges["hf_pyramid_slope_inv"] = {"lin_vel_x": (1.0, 1.0), "lin_vel_y": (0.0, 0.0), "ang_vel_z": (-1.0, 1.0)}
        self.commands.base_velocity.resampling_time_range = (8.0, 12.0)
        self.commands.base_velocity.rel_standing_envs = 0.0
        
        # spawn the robot randomly in the grid (instead of their terrain levels)
        # reduce the number of terrains to save memory
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 1
            self.scene.terrain.terrain_generator.num_cols = 1

        self.scene.leg_volume_points.debug_vis = True
        self.scene.knee_volume_points.debug_vis = True
        self.commands.base_velocity.debug_vis = True
        self.events.reset_robot_joints.params = {
            "position_range": (0.0, 0.0),
            "velocity_range": (0.0, 0.0),
        }


@configclass
class MINI3ParkourEnvCfg(MINI3ParkourRoughEnvCfg, ShoeConfigMixin):
    def __post_init__(self):
        super().__post_init__()
        # self.apply_shoe_config()


@configclass
class MINI3ParkourEnvCfg_PLAY(MINI3ParkourRoughEnvCfg_PLAY, ShoeConfigMixin):
    def __post_init__(self):
        super().__post_init__()
        # self.apply_shoe_config()
