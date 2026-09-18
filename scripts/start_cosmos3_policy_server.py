"""Launch the Cosmos3 RoboLab policy server with content guardrails disabled.

Why: the server unconditionally builds text+video guardrail runners at startup,
which downloads ``nvidia/Cosmos-Guardrail1`` (+Qwen3Guard/RetinaFace weights)
from Hugging Face. That repo is gated and unreachable from here, so startup
crashes before serving. For local simulation eval, content safety filtering is
irrelevant — the empty runner is a no-op by design
(``GuardrailRunner.run_safety_check`` returns safe, ``postprocess`` is identity).

Usage (cosmos3 env, from the EmbodiedBench root):
    conda run --no-capture-output -n cosmos3 python -u scripts/start_cosmos3_policy_server.py \
      --checkpoint-path /mnt/datadisk/models/Cosmos3-Nano-Policy-DROID \
      --port 8000

``--no-capture-output`` is required: plain ``conda run`` pipes the child's
stdout/stderr and only flushes them when it exits, so a long-running server
looks completely silent (``python -u`` does not help — the buffering is in
conda, not Python).

Then run the client:
    conda activate pharm_flow
    python policies/cosmos3/run.py --task BananaInBowlTask --remote-port 8000
"""

import sys
from pathlib import Path

# Shims live in-repo (scripts/_shims) so no third-party tree or conda env is
# modified: the policy server prefers `openpi_server.*`, which we vendor there.
sys.path.insert(0, str(Path(__file__).resolve().parent / "_shims"))

from cosmos_framework.auxiliary.guardrail.common import presets
from cosmos_framework.auxiliary.guardrail.common.core import GuardrailRunner


def _noop_text_runner(offload_model_to_cpu: bool = False) -> GuardrailRunner:
    return GuardrailRunner(safety_models=[], postprocessors=[])


def _noop_video_runner(offload_model_to_cpu: bool = False) -> GuardrailRunner:
    return GuardrailRunner(safety_models=[], postprocessors=[])


presets.create_text_guardrail_runner = _noop_text_runner
presets.create_video_guardrail_runner = _noop_video_runner

# Foreground logs: cosmos only adds file sinks at startup, so re-add stdout here.
from cosmos_framework.utils.log import init_loguru_stdout  # noqa: E402

init_loguru_stdout()

# ... and keep it: _init_log_console() calls logger.remove() twice when
# verbose=False, wiping every stdout sink. Force verbose so the second
# remove() is skipped and foreground logs survive.
from cosmos_framework.inference.common import init as _init_mod  # noqa: E402

_orig_init_log_console = _init_mod._init_log_console


def _init_log_console_fg(*args, **kwargs):
    kwargs["verbose"] = True
    return _orig_init_log_console(*args, **kwargs)


_init_mod._init_log_console = _init_log_console_fg

from cosmos_framework.scripts import action_policy_server_robolab as _server  # noqa: E402


if __name__ == "__main__":
    _server.main()
