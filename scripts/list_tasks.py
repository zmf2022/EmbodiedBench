# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""List available tasks for VLM Agent evaluation."""

import os
import sys

TASK_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "robolab", "tasks", "benchmark")


def main():
    task_dir = os.path.abspath(TASK_DIR)
    if not os.path.isdir(task_dir):
        print(f"Task directory not found: {task_dir}")
        sys.exit(1)

    tasks = sorted(
        f.replace(".py", "")
        for f in os.listdir(task_dir)
        if f.endswith(".py") and not f.startswith("__")
    )

    print(f"Available tasks ({len(tasks)}):\n")
    for task in tasks:
        print(f"  {task}")

    print(f"\nUsage: python policies/vlm_agent/run.py --task {tasks[0]}")


if __name__ == "__main__":
    main()
