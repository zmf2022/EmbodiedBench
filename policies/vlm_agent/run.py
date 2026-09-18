# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Evaluate a VLM as a RoboLab policy across registered tasks."""

import argparse
import logging
import sys
import traceback

import cv2  # noqa: F401 -- must import this before isaaclab. Do not remove
from isaaclab.app import AppLauncher

POLICY = "vlm_agent"

parser = argparse.ArgumentParser(description="Evaluate a VLM policy backend.")
parser.add_argument("--vlm-url", "--vlm_url", default="http://localhost:8000/v1",
                    help="OpenAI-compatible chat-completions base URL (default: http://localhost:8000/v1).")
parser.add_argument("--model", default=None,
                    help="Model name (auto-detected from the server's /models if omitted).")
parser.add_argument("--mode", choices=["token", "tool"], default="token",
                    help=("Control mode: 'token' for discrete action units, 'tool' for "
                          "function calling (default: token)."))
parser.add_argument("--request-timeout", "--request_timeout", type=float, default=120.0,
                    help="Per-request timeout in seconds (default: 120).")
parser.add_argument("--step-m", "--step_m", type=float, default=None,
                    help=("COMMANDED translation per decision, in metres (default: 0.072, "
                          "which measures ~2 cm of physical travel — relative IK achieves "
                          "only ~28%% of what it is asked for). Re-measure with "
                          "examples/run_rel_ik_demo.py after changing the robot, task, "
                          "dt or decimation."))
parser.add_argument("--control-steps", "--control_steps", type=int, default=None,
                    help=("Control steps one decision is split over (default: 8). Total "
                          "travel is set by --step-m and is insensitive to this, so raising "
                          "it shrinks the orientation transient at no cost in distance."))
parser.add_argument("--ik-scale", "--ik_scale", type=float, default=None,
                    help=("Override the action config's translation scale. Leave unset: it "
                          "is read off DroidRelIKActionCfg so the two cannot drift."))
parser.add_argument("--enable-verbose", "--enable_verbose", action="store_true",
                    help="Verbose output (default: False).")

from robolab.eval.runner import add_common_eval_args, run_evaluation  # noqa: E402

add_common_eval_args(parser)
AppLauncher.add_app_launcher_args(parser)

args_cli, _ = parser.parse_known_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import robolab.constants  # noqa: E402
from robolab.registrations.droid.auto_env_registrations_rel_ik import auto_register_droid_rel_ik_envs  # noqa: E402
from robolab.registrations.droid.camera_presets import WRIST_LEFT  # noqa: E402
from robolab.robots.droid import DroidRelIKActionCfg  # noqa: E402

from policies.vlm_agent.client import VLMInferenceClient  # noqa: E402
from policies.vlm_agent.token_controller import CONTROL_STEPS, STEP_M, TokenController  # noqa: E402
from policies.vlm_agent.tool_controller import ToolController  # noqa: E402
from policies.vlm_agent.vlm_backend import VLMBackend  # noqa: E402

robolab.constants.ENABLE_SUBTASK_PROGRESS_CHECKING = args_cli.enable_subtask
robolab.constants.VERBOSE = args_cli.enable_verbose

# The repo configures no logging handler, so the client's per-decision trace would be
# dropped (root has no handler and Python's lastResort only emits WARNING and above).
# Attach one to this policy's logger only: raising the ROOT level to INFO would also
# unmute Isaac Sim, which tests/conftest.py deliberately pins to "warn".
_handler = logging.StreamHandler()
_handler.setFormatter(logging.Formatter("%(message)s"))
_vlm_log = logging.getLogger("policies.vlm_agent")
_vlm_log.addHandler(_handler)
_vlm_log.setLevel(logging.INFO)
_vlm_log.propagate = False

auto_register_droid_rel_ik_envs(
    task_dirs=args_cli.task_dirs,
    task=args_cli.task,
    cameras=WRIST_LEFT,  # wrist + over-the-shoulder left; the prompt names both
)

# Read the translation scale off the action config instead of hardcoding it. Every metric
# displacement the policy commands is divided by this, so a silent change here would
# quietly rescale every motion.
IK_SCALE = args_cli.ik_scale if args_cli.ik_scale is not None else float(DroidRelIKActionCfg().arm_action.scale)


def make_client(args: argparse.Namespace) -> VLMInferenceClient:
    backend = VLMBackend(base_url=args.vlm_url, model=args.model, timeout=args.request_timeout)
    print(f"\033[96m[VLMAgent] Waiting for the VLM server at {args.vlm_url}...\033[0m")
    backend.wait_for_server(timeout=120.0)

    controller_cls = ToolController if args.mode == "tool" else TokenController
    controller = controller_cls(
        ik_scale=IK_SCALE,
        step_m=args.step_m if args.step_m is not None else STEP_M,
        control_steps=args.control_steps if args.control_steps is not None else CONTROL_STEPS,
    )
    print(
        f"\033[96m[VLMAgent] model={backend.model} mode={args.mode} "
        f"ik_scale={controller.ik_scale} step_m={controller.step_m} "
        f"control_steps={controller.control_steps} "
        f"(~{controller.step_m * 0.28 * 1000:.0f} mm physical per decision)\033[0m"
    )
    return VLMInferenceClient(backend=backend, controller=controller, mode=args.mode)


def main() -> None:
    run_evaluation(args_cli, policy=POLICY, client_factory=make_client)
    simulation_app.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\033[96m[RoboLab] Terminated with error: {e}\033[0m")
        traceback.print_exc()
        simulation_app.close()
        sys.exit(1)
