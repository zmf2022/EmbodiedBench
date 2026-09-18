# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""VLM inference client for RoboLab's relative-IK Droid envs.

``infer`` is overridden rather than the default extract/pack/query/unpack-chunk flow, and
that is the central design point, not an optimisation. The base class caches a
``(horizon, action_dim)`` chunk and replays it; **this policy cannot use a precomputed
chunk**, because two of the seven action slots depend on the pose at the instant each
control step is issued:

* the rotation slots carry the correction back to the held-orientation reference, which is
  a function of the CURRENT orientation — replaying a stale one keeps commanding a rotation
  the arm has already made (see :func:`kinematics.hold_orientation_rotvec`);
* the gripper slot carries the latched COMMANDED state, which a later decision may change
  mid-plan.

So one VLM decision expands into a *plan* of N control steps, and each step's action is
materialised from the plan plus the live observation. The base's chunk helpers are left
untouched and unused; everything else (``begin_episode``, ``reset``, ``infer_batch``,
``visualize``) still works as documented.

Querying the model once per decision instead of once per env step is also what makes the
policy affordable: a 90 s task is 1350 control steps, which is ~170 VLM calls at the
default 8 steps per decision rather than 1350.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from robolab.eval.base_client import InferenceClient

from robolab.core.observations.observation_utils import (
    _label_panel,
    unpack_image_obs,
    unpack_proprio_obs,
)

from policies.vlm_agent.token_controller import (
    EMPTY_GRASP_WIDTH_M,
    GRIPPER_OPEN,
    RELEASE_UNIT,
    STILL_UNIT,
    CAMERA_ROTATIONS,
    Decision,
    camera_label,
    camera_role,
    gripper_width_m,
    preprocess_for_vlm,
    rotate_image,
    fingertip_position,
    gripper_tilt_deg,
    hold_orientation_rotvec,
)

logger = logging.getLogger(__name__)

# How many past decisions to show the model. Enough to notice it is oscillating between
# two opposite units, short enough not to crowd the images out of the context.
HISTORY_LEN = 6

# Consecutive per-env query/parse failures tolerated before giving up. A single malformed
# reply must not kill a sweep that has already banked episodes, but a server that is down
# should not burn a whole 1350-step episode issuing hold actions either.
MAX_CONSECUTIVE_FAILURES = 10

# Control steps to let the scene settle before the FIRST model call of an episode
# (Show-Harness's ``num_steps_wait``). The reset drops objects into place, so deciding on
# frame 0 means deciding on a scene that is still moving.
SETTLE_STEPS = 8


# Key ``unpack_image_obs`` adds for the tiled, camera-labelled frame — a view for the
# human, not a camera for the policy.
COMBINED_KEY = "combined_image"


def extract_views(raw_obs: dict, env_id: int) -> list[tuple[str, np.ndarray]]:
    """Return ``[(label, HWC uint8 image)]``.

    External views come first and the wrist view last. The order is deliberate and stable:
    the prompt's direction rules refer to the views by label, and a stable order keeps the
    image/label pairing readable for models that attend to position as well as to the
    interleaved text.
    """
    if "image_obs" not in raw_obs:
        return []
    unpacked = unpack_image_obs(raw_obs, env_id=env_id)
    unpacked.pop(COMBINED_KEY, None)  # the raw full-resolution tile; the preview builds its own
    views = sorted(unpacked.items(), key=lambda kv: (camera_role(kv[0]) == "wrist", kv[0]))
    return [
        (camera_label(key), rotate_image(image, CAMERA_ROTATIONS.get(camera_role(key), 0)))
        for key, image in views
    ]


def extract_state(raw_obs: dict, env_id: int) -> dict:
    """Per-env proprioception, plus the derived quantities the prompt reports.

    ``gripper_pos`` is deliberately surfaced as ``gripper_measured`` and never used as a
    command: it is the *measured* normalised ``finger_joint`` angle, which sits mid-range
    whenever the fingers are loaded against an object. The commanded state is owned by the
    client (see ``VLMInferenceClient._gripper_cmd``).
    """
    if "proprio_obs" not in raw_obs:
        return {}
    state: dict[str, Any] = dict(unpack_proprio_obs(raw_obs, env_id=env_id))

    if "arm_joint_pos" in state:
        state["joint_pos"] = np.asarray(state["arm_joint_pos"], dtype=np.float64).ravel()
    if "gripper_pos" in state:
        state["gripper_measured"] = float(np.asarray(state["gripper_pos"]).ravel()[0])
    if "ee_pos" in state:
        state["ee_pos"] = np.asarray(state["ee_pos"], dtype=np.float64).ravel()[:3]
    if "ee_quat" in state:
        state["ee_quat"] = np.asarray(state["ee_quat"], dtype=np.float64).ravel()[:4]
        state["tilt_deg"] = gripper_tilt_deg(state["ee_quat"])
        if "ee_pos" in state:
            state["tip_pos"] = fingertip_position(state["ee_pos"], state["ee_quat"])
    return state


@dataclass
class _EnvState:
    """Everything the controller latches per env, reset at each episode boundary."""

    plan: Decision | None = None
    steps_left: int = 0
    gripper_cmd: float = GRIPPER_OPEN
    quat_ref: np.ndarray | None = None
    recent: list[str] = field(default_factory=list)
    finished: bool = False
    failures: int = 0
    settle_left: int = SETTLE_STEPS


class VLMInferenceClient(InferenceClient):
    """Drives a Franka + Robotiq 2F-85 through a VLM, in token or tool mode."""

    def __init__(self, backend, controller, mode: str = "token") -> None:
        super().__init__()
        self.backend = backend
        self.controller = controller
        self.mode = mode
        self._envs: dict[int, _EnvState] = {}

    # ------------------------------------------------------------------
    # Episode lifecycle
    # ------------------------------------------------------------------

    def begin_episode(self, episode_idx: int) -> None:
        super().begin_episode(episode_idx)
        self._envs.clear()

    def reset(self, *, env_id: int | None = None) -> None:
        if env_id is None:
            self._envs.clear()
        else:
            self._envs.pop(env_id, None)
        super().reset(env_id=env_id)

    def close(self) -> None:
        closer = getattr(self.backend, "close", None)
        if callable(closer):
            closer()

    def _env_state(self, env_id: int) -> _EnvState:
        if env_id not in self._envs:
            self._envs[env_id] = _EnvState()
        return self._envs[env_id]

    # ------------------------------------------------------------------
    # Control loop
    # ------------------------------------------------------------------

    def infer(self, obs, instruction: str, *, env_id: int = 0) -> dict:
        """One control step: continue the current plan, or ask the VLM for the next one."""
        state = self._env_state(env_id)
        extracted = self._extract_observation(obs, env_id=env_id)
        pose = extracted["state"]

        # Latch the orientation to hold on this env's first step of the episode. Everything
        # afterwards is corrected back to it, so a drifting DLS solution cannot redefine
        # "upright" one step at a time.
        if state.quat_ref is None and "ee_quat" in pose:
            state.quat_ref = np.asarray(pose["ee_quat"], dtype=np.float64).copy()
            logger.info("[VLM] env %d holding orientation %s", env_id, np.round(state.quat_ref, 4).tolist())

        if state.settle_left > 0:
            # Hold while the reset settles — no model call, no plan.
            state.settle_left -= 1
        elif state.steps_left <= 0 and not state.finished:
            # An empty closed gripper is reopened reflexively, without spending a call.
            if not self._auto_release(state, pose, env_id=env_id):
                self._plan_next(state, extracted, instruction, env_id=env_id)

        action = self._action_for_step(state, pose)
        return {"action": action, "viz": self._build_visualization(extracted)}

    def _auto_release(self, state: _EnvState, pose: dict, *, env_id: int) -> bool:
        """Reopen a CLOSED gripper whose fingers have collapsed onto each other.

        Show-Harness's ``plugins/auto_release``, which their RoboLab config enables. It asks
        one physical question — is the closed gripper empty? — and answers it from the
        measured width, so a grasp that missed, or one that slipped later, does not leave
        the arm carrying nothing for the rest of the episode. No model call, no image.
        """
        if state.gripper_cmd <= 0.5 or "gripper_measured" not in pose:
            return False
        width = gripper_width_m(pose["gripper_measured"])
        if width >= EMPTY_GRASP_WIDTH_M:
            return False

        logger.info(
            "[VLM] env %d empty grasp (%.4f m < %.4f m); reopening",
            env_id, width, EMPTY_GRASP_WIDTH_M,
        )
        state.plan = self.controller.decide(RELEASE_UNIT)
        state.steps_left = state.plan.steps
        state.gripper_cmd = GRIPPER_OPEN
        state.recent = (state.recent + ["RELEASE(auto)"])[-HISTORY_LEN:]
        return True

    def _plan_next(self, state: _EnvState, extracted: dict, instruction: str, *, env_id: int) -> None:
        """Query the VLM and install the resulting plan, or hold on failure.

        Two-stage on purpose. The first attempt leaves decoding unconstrained so the model
        can reason about the views before committing; only if the reply cannot be parsed do
        we retry with the vocabulary pinned. Pinning it up front is what a naive
        ``guided_choice`` does, and it forbids the reasoning the choice depends on.
        """
        try:
            try:
                decision = self._decide(extracted, instruction, state, constrain=False)
            except ValueError as exc:
                # A parse failure, not a transport one: the model replied, just not in a
                # shape we could read. Worth one constrained retry.
                logger.warning("[VLM] env %d unparsable reply (%s); retrying constrained", env_id, exc)
                decision = self._decide(extracted, instruction, state, constrain=True)
        except Exception:
            state.failures += 1
            logger.exception(
                "[VLM] env %d decision failed (%d/%d consecutive); holding position",
                env_id, state.failures, MAX_CONSECUTIVE_FAILURES,
            )
            if state.failures >= MAX_CONSECUTIVE_FAILURES:
                raise
            # Hold for exactly one step, then query again rather than replaying a stale
            # plan. This has to be a real one-step plan: leaving ``plan`` as None would pin
            # ``steps_left`` (nothing decrements it) and the env would hold for the rest of
            # the episode without ever retrying.
            state.plan = self.controller.decide(STILL_UNIT)
            state.steps_left = 1
            return

        state.failures = 0
        state.plan = decision
        state.steps_left = decision.steps
        state.recent = (state.recent + [decision.unit])[-HISTORY_LEN:]

        if decision.gripper is not None:
            state.gripper_cmd = float(decision.gripper)
        if decision.yaw_rad and state.quat_ref is not None:
            # Rotation moves the REFERENCE; the hold then drives the arm onto it while
            # still correcting roll and pitch, so a yaw command cannot become a tilt.
            state.quat_ref = self.controller.rotate_reference(state.quat_ref, decision)
        if decision.terminal:
            state.finished = True
            logger.info(
                "[VLM] env %d %s — no further model calls this episode. %s",
                env_id, decision.unit, decision.note,
            )

        logger.info(
            "[VLM] env %d | %s steps=%d dxyz=%s yaw=%.3f gripper=%s wrist=%s | %s",
            env_id, decision.unit, decision.steps,
            np.round(decision.move_m * decision.steps, 4).tolist(),
            decision.yaw_rad,
            "CLOSED" if state.gripper_cmd > 0.5 else "OPEN",
            {True: "YES", False: "NO", None: "-"}[decision.wrist],
            decision.note,
        )

    def _decide(self, extracted: dict, instruction: str, state: _EnvState, *, constrain: bool) -> Decision:
        request = self._pack_request(extracted, instruction, state=state, constrain=constrain)
        return self._unpack_response(self._query_server(request))

    def _action_for_step(self, state: _EnvState, pose: dict) -> np.ndarray:
        """Materialise this control step's 7-D action from the plan and the live pose."""
        decision = state.plan
        if decision is None or state.steps_left <= 0 or decision.terminal:
            decision = self.controller.decide(STILL_UNIT)
        else:
            state.steps_left -= 1

        rot = np.zeros(3)
        if state.quat_ref is not None and "ee_quat" in pose:
            rot = hold_orientation_rotvec(state.quat_ref, pose["ee_quat"])

        return self.controller.action_for_step(
            decision, gripper_cmd=state.gripper_cmd, rot_correction=rot
        )

    # ------------------------------------------------------------------
    # Base-class hooks
    # ------------------------------------------------------------------

    def _extract_observation(self, raw_obs, *, env_id: int = 0) -> dict:
        return {
            "views": extract_views(raw_obs, env_id),
            "state": extract_state(raw_obs, env_id),
        }

    def _pack_request(
        self,
        extracted_obs: dict,
        instruction: str,
        *,
        state: _EnvState | None = None,
        constrain: bool = False,
    ) -> dict:
        """Build the chat request. ``state`` carries the commanded gripper and the decision
        history, which are client-owned rather than observable, so they cannot come from the
        obs. ``constrain`` pins the reply to the vocabulary; see :meth:`_plan_next`."""
        gripper_cmd = state.gripper_cmd if state is not None else GRIPPER_OPEN
        recent = state.recent if state is not None else []
        payload = dict(extracted_obs)
        payload["instruction"] = instruction
        messages = self.controller.build_prompt(payload, gripper_cmd=gripper_cmd, recent=recent)

        request: dict = {"messages": messages}
        if self.mode == "tool":
            request["tools"] = self.controller.get_tool_schemas()
        elif constrain:
            request["choices"] = list(self.controller.TOKEN_VOCAB)
        return request

    def _query_server(self, request: dict) -> dict:
        return self.backend.complete(
            request["messages"],
            tools=request.get("tools"),
            choices=request.get("choices"),
        )

    def _unpack_response(self, response: dict) -> Decision:
        """Return a :class:`Decision`, not an action chunk.

        Deliberately narrower than the base annotation: ``infer`` is overridden and the
        ``(horizon, action_dim)`` chunk path is unused, because the action cannot be
        computed until the control step is actually issued (see the module docstring).
        """
        choices = response.get("choices") or []
        if not choices:
            raise RuntimeError(f"VLM response has no choices: {str(response)[:200]}")
        # A reply cut off at ``max_tokens`` is a thinking trace with no answer on the end.
        # ``parse_unit``'s last-mention fallback would still return *something* from it —
        # whichever option the model happened to be weighing when the budget ran out — and
        # that reads as a clean decision in the log. Raise instead, so ``_plan_next`` takes
        # the constrained retry, which pins the reply to one vocabulary token and cannot
        # truncate. Only the server serving a reasoning model without a ``--reasoning-parser``
        # puts the trace in ``content`` at all; see ``scripts/start_cosmos3.sh``.
        if choices[0].get("finish_reason") == "length":
            raise ValueError(
                "reply truncated at max_tokens (thinking trace never reached an ACTION line)"
            )
        return self.controller.decision_from_message(choices[0].get("message") or {})

    def _build_visualization(self, extracted_obs: dict) -> np.ndarray | None:
        """The frames the model is actually given, tiled and labelled.

        Built from the preprocessed 256x256 frames rather than ``unpack_image_obs``'s
        full-resolution tile, so the preview shows the crop, the letterbox and the
        resolution the model sees — and fits on a screen without extra scaling.
        """
        views = extracted_obs.get("views")
        if not views:
            return None
        return np.concatenate(
            [_label_panel(preprocess_for_vlm(image), label) for label, image in views],
            axis=1,
        )
