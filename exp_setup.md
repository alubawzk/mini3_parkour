```bash
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
```

## Train
```bash
nohup bash -lc 'CUDA_VISIBLE_DEVICES=3 python robolab/scripts/rsl_rl/train.py --task Mini3-AMP --num_envs 8192 --max_iterations 10000 --headless --run_name Mini3_AMP_Baseline >g1_amp_3.log 2>&1' &

nohup env CUDA_VISIBLE_DEVICES=1 python robolab/scripts/rsl_rl/train.py --task Mini3-Flat --headless --logger tensorboard --num_envs 8192 --max_iterations 10000 --checkpoint_root=./logs/rsl_rl --run_name ReviseFootCollis_AddVelTrack_ChangeDefaultPos >my_output1.log 2>&1 &


nohup env CUDA_VISIBLE_DEVICES=2 python robolab/scripts/rsl_rl/train.py --task Mini3-AMP --headless --logger tensorboard --num_envs 8192 --max_iterations 10000 --checkpoint_root=./logs/rsl_rl --run_name Revise_FootCollis_ChangeDefaultPos > my_output2.log 2>&1 &


nohup env CUDA_VISIBLE_DEVICES=3 python robolab/scripts/rsl_rl/train.py --task Mini3-AMP --headless --logger tensorboard --num_envs 8192 --max_iterations 10000 --run_name Add_KeyBodyObs_NoFriction_ReviseDefaultPos >my_output3.log 2>&1 &


nohup env CUDA_VISIBLE_DEVICES=4 python robolab/scripts/rsl_rl/train.py --task Mini3-Flat --headless --logger tensorboard --num_envs 8192 --max_iterations 10000 --run_name Add_Pitch_10Degree_AddCoMDomainRand >my_output4.log 2>&1 &


nohup env CUDA_VISIBLE_DEVICES=5 python robolab/scripts/rsl_rl/train.py --task Mini3-AMP --headless --logger tensorboard --num_envs 8192 --max_iterations 10000 --run_name Add_KeyBodyObs_NoFriction_ReviseDefaultPos >my_output5.log 2>&1 &


nohup env CUDA_VISIBLE_DEVICES=6 python robolab/scripts/rsl_rl/train.py --task Mini3-AMP --headless --logger tensorboard --num_envs 8192 --max_iterations 10000 --run_name Add_KeyBodyObs_NoFriction >my_output6.log 2>&1 &


nohup env CUDA_VISIBLE_DEVICES=7 python robolab/scripts/rsl_rl/train.py --task Mini3-AMP --headless --logger tensorboard --num_envs 8192 --max_iterations 10000 --run_name Add_KeyBodyObs >my_output7.log 2>&1 &



## Mimic
nohup bash -lc 'CUDA_VISIBLE_DEVICES=0 python robolab/scripts/rsl_rl/train.py --task Mini3-BeyondMimic --logger tensorboard --num_envs 8192 --max_iterations 15000 --run_name Charles_DynamicFriction_Resume --headless >my_output_dance_0.log 2>&1; echo $? > my_output_dance_0.exitcode' &


nohup bash -lc 'CUDA_VISIBLE_DEVICES=1 python robolab/scripts/rsl_rl/train.py --task Mini3-BeyondMimic --logger tensorboard --num_envs 8192 --max_iterations 15000 --run_name Charles_DynamicFriction_Resume --headless >my_output_dance_1.log 2>&1; echo $? > my_output_dance_1.exitcode' &


nohup bash -lc 'CUDA_VISIBLE_DEVICES=2 python robolab/scripts/rsl_rl/train.py --task Mini3-BeyondMimic --logger tensorboard --num_envs 8192 --max_iterations 15000 --run_name Charles_DynamicFriction_Resume --headless >my_output_dance_2.log 2>&1; echo $? > my_output_dance_2.exitcode' &


nohup bash -lc 'CUDA_VISIBLE_DEVICES=3 python robolab/scripts/rsl_rl/train.py --task Mini3-BeyondMimic --logger tensorboard --num_envs 8192 --max_iterations 15000 --run_name Charles_DynamicFriction_Resume --headless >my_output_dance_3.log 2>&1; echo $? > my_output_dance_3.exitcode' &


nohup bash -lc 'CUDA_VISIBLE_DEVICES=4 python robolab/scripts/rsl_rl/train.py --task Mini3-BeyondMimic --logger tensorboard --num_envs 8192 --max_iterations 15000 --run_name Charles_DynamicFriction_Resume --headless >my_output_dance_4.log 2>&1; echo $? > my_output_dance_4.exitcode' &


nohup bash -lc 'CUDA_VISIBLE_DEVICES=5 python robolab/scripts/rsl_rl/train.py --task Mini3-BeyondMimic --logger tensorboard --num_envs 8192 --max_iterations 15000 --run_name Charles_DynamicFriction_Resume --headless >my_output_dance_5.log 2>&1; echo $? > my_output_dance_5.exitcode' &


nohup bash -lc 'CUDA_VISIBLE_DEVICES=6 python robolab/scripts/rsl_rl/train.py --task Mini3-BeyondMimic --logger tensorboard --num_envs 8192 --max_iterations 15000 --run_name Charles_DynamicFriction_Resume --headless >my_output_dance_6.log 2>&1; echo $? > my_output_dance_6.exitcode' &


nohup bash -lc 'CUDA_VISIBLE_DEVICES=7 python robolab/scripts/rsl_rl/train.py --task Mini3-BeyondMimic --logger tensorboard --num_envs 8192 --max_iterations 15000 --run_name 97_BackFlip_ReduceTerminationThreshold --headless >my_output_dance_7.log 2>&1; echo $? > my_output_dance_7.exitcode' &
465923

```

## Play
```bash
python robolab/scripts/rsl_rl/play.py --task Mini3-Flat --load_run 2026-05-02_09-58-21_Add_Orientation0.2_AddSmooth_AddDynamicFriction --num_envs 1
python robolab/scripts/rsl_rl/play_bm.py --task Mini3-BeyondMimic --load_run 2026-05-04_00-00-23_Charles_DynamicFriction --num_envs 1 env.use_terrain=false
```


## Train G1-CatTraverse
```bash
python robolab/scripts/rsl_rl/play.py --task G1-CatTraverse --field_path data/TypiObs/narrow0 --load_run 2026-04-13_01-48-27_mini3_12DoF_ReviseReward --num_envs 1

nohup env CUDA_VISIBLE_DEVICES=0 python robolab/scripts/rsl_rl/train.py --task G1-CatTraverse --field_path data/TypiObs/narrow0 --headless --logger tensorboard --num_envs 8192 --max_iterations 10000 --run_name Narrow0_baseline >collis_0.log 2>&1 &


nohup env CUDA_VISIBLE_DEVICES=1 python robolab/scripts/rsl_rl/train.py --task G1-CatTraverse --field_path data/TypiObs/bar0 --headless --logger tensorboard --num_envs 8192 --max_iterations 10000 --run_name Bar0_baseline >collis_1.log 2>&1 &


nohup env CUDA_VISIBLE_DEVICES=2 python robolab/scripts/rsl_rl/train.py --task G1-CatTraverse --field_path data/TypiObs/hole --headless --logger tensorboard --num_envs 8192 --max_iterations 10000 --run_name Hole_baseline >collis_2.log 2>&1 &


nohup env CUDA_VISIBLE_DEVICES=3 python robolab/scripts/rsl_rl/train.py --task G1-CatTraverse --field_path data/TypiObs/ceil0 --headless --logger tensorboard --num_envs 8192 --max_iterations 10000 --run_name Ceil0_baseline >collis_3.log 2>&1 &


```

## Train G1-CatSonic
```bash
nohup bash -lc 'CUDA_VISIBLE_DEVICES=0 python robolab/scripts/rsl_rl/train.py --task G1-CatSonic --field_path data/TypiObs/narrow0 --logger tensorboard --num_envs 8192 --max_iterations 15000 --run_name ReviseCMD_Narrow0 --headless >sonic_0.log 2>&1' &
179619

nohup bash -lc 'CUDA_VISIBLE_DEVICES=1 python robolab/scripts/rsl_rl/train.py --task G1-CatSonic --field_path data/TypiObs/narrow0 --logger tensorboard --num_envs 8192 --max_iterations 15000 --run_name ReviseCMD_Narrow0_AddTorsoUp --headless >sonic_1.log 2>&1' &
187571

nohup bash -lc 'CUDA_VISIBLE_DEVICES=2 python robolab/scripts/rsl_rl/train.py --task G1-CatSonic --field_path data/TypiObs/ceil0 --logger tensorboard --num_envs 8192 --max_iterations 15000 --run_name Ceil0_AllDoF --headless >sonic_2.log 2>&1' &


nohup bash -lc 'CUDA_VISIBLE_DEVICES=3 python robolab/scripts/rsl_rl/train.py --task G1-CatSonic --field_path data/TypiObs/hole --logger tensorboard --num_envs 8192 --max_iterations 15000 --run_name Hole_AllDoF --headless >sonic_3.log 2>&1' &


nohup bash -lc 'CUDA_VISIBLE_DEVICES=4 python robolab/scripts/rsl_rl/train.py --task G1-CatSonic --field_path data/TypiObs/bend --logger tensorboard --num_envs 8192 --max_iterations 15000 --run_name ReviseCMD_Bend --headless >sonic_4.log 2>&1' &
180784

nohup bash -lc 'CUDA_VISIBLE_DEVICES=4 python robolab/scripts/rsl_rl/train.py --task G1-CatSonic --field_path data/TypiObs/hole --logger tensorboard --num_envs 8192 --max_iterations 15000 --run_name Hole_MultiBranch_AllDoF --headless >sonic_4.log 2>&1' &


nohup bash -lc 'CUDA_VISIBLE_DEVICES=5 python robolab/scripts/rsl_rl/train.py --task G1-CatSonic --field_path data/TypiObs/narrow0 --logger tensorboard --num_envs 8192 --max_iterations 15000 --run_name Narrow0_MultiBranch_AllDoF --headless >sonic_5.log 2>&1' &


nohup bash -lc 'CUDA_VISIBLE_DEVICES=6 python robolab/scripts/rsl_rl/train.py --task G1-CatSonic --field_path data/TypiObs/bend --logger tensorboard --num_envs 8192 --max_iterations 15000 --run_name Bend_MultiBranch_AllDoF --headless >sonic_6.log 2>&1' &


nohup bash -lc 'CUDA_VISIBLE_DEVICES=7 python robolab/scripts/rsl_rl/train.py --task G1-CatSonic --field_path data/TypiObs/ceil0 --logger tensorboard --num_envs 8192 --max_iterations 15000 --run_name Ceil0_MultiBranch_AllDoF --headless >sonic_7.log 2>&1' &


```

## Play G1-CatSonic
```bash
python robolab/scripts/rsl_rl/play.py --task G1-CatSonic --field_path data/TypiObs/narrow0 --load_run 2026-05-03_12-01-02_Narrow0_Sonic_ObsWholeLastAction --num_envs 1
python robolab/scripts/rsl_rl/play.py --task G1-CatSonic --field_path data/TypiObs/ceil0 --load_run 2026-05-03_12-25-12_Ceil0_Sonic_ObsWholeLastAction_29 --num_envs 1
python robolab/scripts/rsl_rl/play.py --task G1-CatSonic --field_path data/TypiObs/bend --load_run 2026-05-03_12-25-51_Bend_Sonic_ObsWholeLastAction_29 --num_envs 1
python robolab/scripts/rsl_rl/play.py --task G1-CatSonic --field_path data/TypiObs/hole --load_run 2026-05-03_12-02-45_Hole_Sonic_baseline_ObsWholeLastAction --num_envs 1
```

## Train G1-Depth
```bash
nohup bash -lc 'CUDA_VISIBLE_DEVICES=3 python robolab/scripts/rsl_rl/train.py --task G1-CatDepth --num_envs 8192 --max_iterations 10000 --field_path data/TypiObs/bend --headless --run_name Bend_Depth_Baseline env.teacher_checkpoint_path=//mnt/data/mini3_lab/logs/rsl/G1-CatSonic/g1_cat_sonic/2026-05-05_21-16-36_Bend_MultiBranch/model_14999.pt >sonic_3.log 2>&1; echo $? > sonic_3.exitcode' &
```


## Train G1-SonicMB
```bash
nohup bash -lc 'CUDA_VISIBLE_DEVICES=0 GROOT_WHOLEBODY_ROOT=/home/wzk/gear_jonic python robolab/scripts/rsl_rl/train.py --task G1-CatSonicMB --num_envs 8192 --max_iterations 10000 --field_path data/TypiObs/ceil0 --headless --run_name Ceil0_CatSonicMB_Baseline >sonicmb_0.log 2>&1' &
596002

nohup bash -lc 'CUDA_VISIBLE_DEVICES=1 GROOT_WHOLEBODY_ROOT=/home/wzk/gear_jonic python robolab/scripts/rsl_rl/train.py --task G1-CatSonicMB --num_envs 8192 --max_iterations 10000 --field_path data/TypiObs/narrow0 --headless --run_name Narrow0_CatSonicMB_Baseline >sonicmb_1.log 2>&1' &
599379

nohup bash -lc 'CUDA_VISIBLE_DEVICES=2 GROOT_WHOLEBODY_ROOT=/home/wzk/gear_jonic python robolab/scripts/rsl_rl/train.py --task G1-CatSonicMB --num_envs 8192 --max_iterations 10000 --field_path data/TypiObs/bar1 --headless --run_name Bar1_CatSonicMB_Baseline >sonicmb_2.log 2>&1' &
601815

nohup bash -lc 'CUDA_VISIBLE_DEVICES=3 GROOT_WHOLEBODY_ROOT=/home/wzk/gear_jonic python robolab/scripts/rsl_rl/train.py --task G1-CatSonicMB --num_envs 8192 --max_iterations 10000 --field_path data/TypiObs/ceil1 --headless --run_name Ceil1_CatSonicMB_Baseline >sonicmb_3.log 2>&1' &


``` 

## G1 cat-amp
```bash
nohup bash -lc 'CUDA_VISIBLE_DEVICES=0 python robolab/scripts/rsl_rl/train.py --task G1-AMP --num_envs 8192 --max_iterations 10000 --headless --run_name Free_Upright_Walk_w_Style_w_FlatTask >g1_amp_0.log 2>&1' &


nohup bash -lc 'CUDA_VISIBLE_DEVICES=1 python robolab/scripts/rsl_rl/train.py --task G1-AMP --num_envs 8192 --max_iterations 10000 --headless --run_name Free_Upright_Walk_w_Style_w_FlatTask_Revise_YDistReward >g1_amp_1.log 2>&1' &


nohup bash -lc 'CUDA_VISIBLE_DEVICES=2 python robolab/scripts/rsl_rl/train.py --task G1-AMP --num_envs 8192 --max_iterations 10000 --headless --run_name Free_Squat_Walk_w_Style_w_FlatTask_Revise_YDistReward >g1_amp_2.log 2>&1' &


nohup bash -lc 'CUDA_VISIBLE_DEVICES=3 python robolab/scripts/rsl_rl/train.py --task G1-AMP-Crawl --num_envs 8192 --max_iterations 10000 --headless --run_name Revise_ang_vel_xy_l2_and_lin_vel_z_l2_Add_FlatOri >g1_amp_3.log 2>&1' &


nohup bash -lc 'CUDA_VISIBLE_DEVICES=4 python robolab/scripts/rsl_rl/train.py --task G1-AMP-Crawl --num_envs 8192 --max_iterations 10000 --headless --run_name Revise_VelCMDTrackReward_Add_Crawl_mode_weights >g1_amp_4.log 2>&1' &


nohup bash -lc 'CUDA_VISIBLE_DEVICES=5 python robolab/scripts/rsl_rl/train.py --task G1-AMP-Crawl --num_envs 8192 --max_iterations 10000 --headless --run_name Revise_ang_vel_xy_l2_and_lin_vel_z_l2_Add_FlatOri >g1_amp_5.log 2>&1' &


nohup bash -lc 'CUDA_VISIBLE_DEVICES=6 python robolab/scripts/rsl_rl/train.py --task G1-AMP-Crawl --num_envs 8192 --max_iterations 10000 --headless --run_name Revise_VelCMDTrackReward_Add_Crawl_mode_weights_Add_flat_orient >g1_amp_6.log 2>&1' &


nohup bash -lc 'CUDA_VISIBLE_DEVICES=7 python robolab/scripts/rsl_rl/train.py --task G1-AMP-Crawl --num_envs 8192 --max_iterations 10000 --headless --run_name Revise_SplitedCMDSamplingLogitic >g1_amp_7.log 2>&1' &





nohup bash -lc 'CUDA_VISIBLE_DEVICES=0 python robolab/scripts/rsl_rl/train.py --task MINI3-Parkour --headless --logger=tensorboard --num_envs=4096 --max_iterations 10000 --run_name Parkour_baseline_500Hz >mini3_parkour_0.log 2>&1' &
2762727

nohup bash -lc 'CUDA_VISIBLE_DEVICES=1 python robolab/scripts/rsl_rl/train.py --task MINI3-Parkour --headless --logger=tensorboard --num_envs=4096 --max_iterations 10000 --run_name Add_VelCMDTrack >mini3_parkour_1.log 2>&1' &
2798339

nohup bash -lc 'CUDA_VISIBLE_DEVICES=2 python robolab/scripts/rsl_rl/train.py --task MINI3-Parkour --headless --logger=tensorboard --num_envs=4096 --max_iterations 10000 --run_name Redcue_FeetHeightOffset_0.02 >mini3_parkour_2.log 2>&1' &
2801387

nohup bash -lc 'CUDA_VISIBLE_DEVICES=3 python robolab/scripts/rsl_rl/train.py --task RPO-Parkour --headless --logger=tensorboard --num_envs=4096 --max_iterations 10000 --run_name RPO_Parkour_Baseline >pro_parkour_.log 2>&1' &

nohup bash -lc 'CUDA_VISIBLE_DEVICES=4 python robolab/scripts/rsl_rl/train.py --task MINI3-Parkour --headless --logger tensorboard --num_envs 4096 --max_iterations 10000 --run_name Revise_VOLUME_POINTS_GRID >mini3_parkour_4.log 2>&1' &
2830554

nohup bash -lc 'CUDA_VISIBLE_DEVICES=5 python robolab/scripts/rsl_rl/train.py --task MINI3-Parkour --headless --logger tensorboard --num_envs 4096 --max_iterations 10000 --run_name WO_AMP >mini3_parkour_5.log 2>&1' &
2833623


```


## mini3 force
```bash
CUDA_VISIBLE_DEVICES=4 python robolab/scripts/rsl_rl/train.py \
  --task=Mini3-Force --headless --logger=tensorboard \
  --num_envs=8192 --max_iterations=10000 --run_name=force_base_est \
  >force_base_4.log 2>&1 &
```
