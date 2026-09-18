# VLM Agent Policy

VLM robot control for EmbodiedBench, aligned with [Show-Harness](https://github.com/chenanno/Show-Harness).

## Quick Start

```bash
# 1. Start VLM server (二选一)
./scripts/start_qwen3.sh      # Qwen3.8-27B-NVFP4
# or
./scripts/start_cosmos3.sh    # Cosmos3-Nano reasoner

# 2. Run
conda activate pharm_flow
python policies/vlm_agent/run.py --task BananaInBowlTask --visualizer kit
```

## Commands

```bash
# Token mode (default)
python policies/vlm_agent/run.py --task BananaInBowlTask --mode token --visualizer kit

# Tool mode
python policies/vlm_agent/run.py --task BananaInBowlTask --mode tool --visualizer kit

# Multiple envs
python policies/vlm_agent/run.py --task BananaInBowlTask --num-envs 4

# Custom VLM server
python policies/vlm_agent/run.py --task BananaInBowlTask --vlm-url http://10.0.0.1:8000/v1

# Run multiple tasks
python policies/vlm_agent/run.py --task BananaInBowlTask CubeInBowlTask_0

# List available tasks
python scripts/list_tasks.py

# Adaptive sampling (up to 200 episodes)
python policies/vlm_agent/run.py --task BananaInBowlTask --num-episodes-adaptive 200

# Save videos
python policies/vlm_agent/run.py --task BananaInBowlTask --video-mode all

# Calibrate axis signs / travel per step (run after changing robot, task, dt or decimation)
python examples/run_rel_ik_demo.py --task BananaInBowlTask --delta 0.018
```

## Configuration

| Argument | Default | Description |
|----------|---------|-------------|
| `--vlm-url` | `http://localhost:8000/v1` | VLM API endpoint |
| `--model` | auto | Model name (auto-detected) |
| `--mode` | `token` | `token` or `tool` |
| `--task` | None | Task name(s) |
| `--num-envs` | 1 | Number of parallel envs |
| `--num-runs` | 1 | Runs per task |
| `--video-mode` | `all` | `all`, `viewport`, `sensor`, `none` |
| `--step-m` | `0.072` | 每次决策命令的位移量（m） |
| `--control-steps` | `8` | 一次决策拆成几个控制步 |
| `--request-timeout` | `120` | 单次请求超时（秒） |

## Robot Configuration

所有 policy 共用同一个机器人配置：Franka + Robotiq 2F-85（`DroidCfg`）。

```python
# robolab/registrations/droid/auto_env_registrations_rel_ik.py
auto_register_droid_rel_ik_envs(
    task=args_cli.task,
    cameras=WRIST_LEFT,  # 2 cameras: wrist + left shoulder
    # robot_cfg=DroidCfg  # 固定 Franka + Robotiq 2F-85
)
```

其他 policy（Pi0、GR00T、Cosmos3）也用相同的 `DroidCfg`，区别只在 action space：
- Pi0/GR00T: `auto_register_droid_envs` → joint position (8D)
- VLM Agent: `auto_register_droid_rel_ik_envs` → relative IK (7D)

## Token Mode

VLM 每次决策输出一个离散 token，解释器将其展开成 8 个控制步的笛卡尔位移。

```
VLM 观察图像 + 状态 → 输出 "MV_FWD" → 解释器 → 8 × [0.018, 0, 0, 姿态修正×3, gripper]
```

| Token | 含义 | 命令量 | 实际位移 |
|-------|------|--------|---------|
| `MV_FWD` | 前移 | +X 0.072m | ≈20mm |
| `MV_BACK` | 后移 | -X 0.072m | ≈20mm |
| `MV_LEFT` | 左移 | +Y 0.072m | ≈20mm |
| `MV_RIGHT` | 右移 | -Y 0.072m | ≈20mm |
| `MV_UP` | 上升 | +Z 0.072m | ≈20mm |
| `MV_DOWN` | 下降 | -Z 0.072m | ≈20mm |
| `ROTATE_CW` | 顺时针旋转 | Yaw +0.15rad | — |
| `ROTATE_CCW` | 逆时针旋转 | Yaw -0.15rad | — |
| `GRASP` | 闭合夹爪 | gripper=1，保持 10 步 | — |
| `RELEASE` | 打开夹爪 | gripper=0，保持 10 步 | — |
| `DONE` | 任务完成 | 停止调用 VLM | — |
| `STILL` | 不动 | 保持当前 | — |

命令量 ≠ 实际位移：relative IK 只达成命令量的约 28%，所以 0.072m 命令对应约 2cm 物理位移。
命令量除以 `DroidRelIKActionCfg.scale`（0.5）后才写入 action。

模型输出格式：两行，`WRIST: <YES|NO>` 然后 `ACTION: <TOKEN>`。

## Tool Mode

VLM 通过 function calling 输出结构化工具调用：

| 工具 | 参数 | 说明 |
|------|------|------|
| `move_by` | `deltas: {x, y, z, yaw}`, `note` | 按基座坐标系位移（m / rad），超出上限自动裁剪 |
| `grasp` | 无 | 闭合夹爪 |
| `release` | 无 | 打开夹爪 |
| `done` | `summary` | 任务完成 |
| `give_up` | `reason` | 无法完成 |

## Specs

- **Robot**: Franka + Robotiq 2F-85 (relative IK, 7-DoF Cartesian delta)
- **Cameras**: 2 (wrist + left shoulder), 640×360 PNG input（1280×720 等比缩小，无裁剪无黑边），prompt 中分别标注为 `AgentView` / `WristView`
- **决策频率**: 每 8 个控制步一次（15Hz 控制，90s 任务约 170 次调用）
- **Tests**: `python -m pytest tests/test_vlm_client.py`

各项数值的来源和标定过程见 `token_controller.py` 与 `kinematics.py` 的 docstring。

## 注意

- `DONE` 不会提前结束 episode（`robolab/eval/episode.py` 没有 client 端终止钩子），只是停止调用 VLM。
- `MOVE_VECTORS` 的轴向符号按 RoboLab Franka 约定（+X 远离基座、机器人左 = +Y、+Z 上），换机器人或任务后用上面的 `run_rel_ik_demo.py` 确认，符号错了日志上看不出来。
