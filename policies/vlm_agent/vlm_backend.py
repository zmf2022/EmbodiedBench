# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""HTTP transport for an OpenAI-compatible chat-completions server (vLLM, and friends).

Follows the transport conventions of ``policies/molmoact2_droid/client.py``: an injectable
``requests.Session``, a validated timeout, and ``raise_for_status`` for anything that is
not worth retrying.

The one non-obvious part is constrained decoding. Restricting the reply to the action
vocabulary is useful as a *fallback* when a model will not honour the ``ACTION: <UNIT>``
contract, but it must not be the default: with the output pinned to a single vocabulary
token the model cannot emit any reasoning, and choosing a direction from two camera views
is exactly the kind of decision that needs it. The field spelling also moved — vLLM's
legacy top-level ``guided_choice`` became ``structured_outputs: {"choice": [...]}``, and a
non-vLLM server has neither — so :meth:`complete` walks that ladder once and remembers
which rung worked instead of failing on the first 400.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import requests

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT_SECONDS = 120.0

# Thinking, at the highest setting the server will take. vLLM toggles Qwen3-style thinking
# through the chat template, where it is a BOOLEAN with no tiers; the graded knob is
# ``reasoning_effort``, which vLLM accepts for some models and rejects for others, so it is
# negotiated below rather than assumed. Both chat-template spellings are sent because
# different templates read different names; a template using neither ignores them.
THINKING = {"enable_thinking": True, "thinking": True}
REASONING_EFFORT = "xhigh"

# A thinking trace runs well past a few hundred tokens, and a truncated one contains no
# 'ACTION:' line at all — every decision would fail and the arm would just hold. This is the
# budget for trace + answer together.
DEFAULT_MAX_TOKENS = 8192
RETRY_STATUS = (408, 409, 429, 500, 502, 503, 504)

# Constrained-decoding spellings, newest first. ``None`` is the terminal rung: send the
# request unconstrained and let the caller parse the reply.
_CHOICE_DIALECTS: tuple[str | None, ...] = ("structured_outputs", "guided_choice", None)

# "not negotiated yet", distinct from the legitimate ``None`` rung ("this server supports
# no constrained decoding at all").
_UNNEGOTIATED = object()


class VLMBackend:
    """Chat-completions client with retries and constrained-decoding negotiation."""

    def __init__(
        self,
        base_url: str = "http://localhost:8000/v1",
        model: str | None = None,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        max_retries: int = 3,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = 0.0,
        session: requests.Session | None = None,
    ) -> None:
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        if max_retries < 1:
            raise ValueError("max_retries must be >= 1")
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = float(timeout)
        self.max_retries = int(max_retries)
        self.max_tokens = int(max_tokens)
        self.temperature = float(temperature)
        self.session = session or requests.Session()
        # Which constrained-decoding spelling this server accepts; negotiated on first use.
        self._choice_dialect: Any = _UNNEGOTIATED
        # Cleared if the server rejects the field rather than the request.
        self._send_reasoning_effort = True

    # ------------------------------------------------------------------
    # Requests
    # ------------------------------------------------------------------

    def _payload(
        self,
        messages: list[dict],
        *,
        tools: list[dict] | None,
        choices: list[str] | None,
        dialect: str | None,
        max_tokens: int | None,
    ) -> dict:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": int(max_tokens or self.max_tokens),
            "chat_template_kwargs": THINKING,
        }
        if self._send_reasoning_effort:
            payload["reasoning_effort"] = REASONING_EFFORT
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        if choices and dialect == "structured_outputs":
            payload["structured_outputs"] = {"choice": list(choices)}
        elif choices and dialect == "guided_choice":
            payload["guided_choice"] = list(choices)
        return payload

    def complete(
        self,
        messages: list[dict],
        *,
        tools: list[dict] | None = None,
        choices: list[str] | None = None,
        max_tokens: int | None = None,
    ) -> dict:
        """POST one chat completion.

        ``choices`` constrains the reply to that set of strings. It is only honoured when
        the caller asks for it — see the module docstring for why that is not the default.
        """
        dialects: tuple[str | None, ...]
        if not choices:
            dialects = (None,)
        elif self._choice_dialect is _UNNEGOTIATED:
            dialects = _CHOICE_DIALECTS
        else:
            dialects = (self._choice_dialect,)  # type: ignore[assignment]

        last_error: Exception | None = None
        for dialect in dialects:
            payload = self._payload(
                messages, tools=tools, choices=choices, dialect=dialect, max_tokens=max_tokens
            )
            try:
                result = self._post(payload)
            except _UnsupportedField as exc:
                if self._send_reasoning_effort and "reasoning_effort" in str(exc).lower():
                    # Graded effort is not supported here; thinking itself still is.
                    logger.warning("[VLM] server rejected reasoning_effort=%r; dropping it "
                                   "(thinking stays on via chat_template_kwargs)", REASONING_EFFORT)
                    self._send_reasoning_effort = False
                    payload.pop("reasoning_effort", None)
                    result = self._post(payload)
                    if choices:
                        self._choice_dialect = dialect
                    return result
                # This spelling is not known to the server; fall through to the next rung.
                logger.info("[VLM] server rejected %r constrained decoding (%s)", dialect, exc)
                last_error = exc
                continue
            if choices:
                self._choice_dialect = dialect
                if dialect is None:
                    logger.warning(
                        "[VLM] server supports no constrained-decoding field; relying on the "
                        "'ACTION: <UNIT>' output contract alone."
                    )
            return result

        raise RuntimeError(f"VLM rejected every constrained-decoding dialect: {last_error}")

    def _post(self, payload: dict) -> dict:
        url = f"{self.base_url}/chat/completions"
        last: Exception | None = None

        for attempt in range(self.max_retries):
            try:
                resp = self.session.post(url, json=payload, timeout=self.timeout)
                if resp.status_code == 200:
                    return resp.json()
                body = (resp.text or "")[:500]
                if resp.status_code == 400 and _mentions_unknown_field(body):
                    raise _UnsupportedField(body)
                if resp.status_code not in RETRY_STATUS:
                    raise RuntimeError(f"VLM API {resp.status_code}: {body}")
                last = RuntimeError(f"VLM API {resp.status_code}: {body}")
            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as exc:
                last = exc

            if attempt < self.max_retries - 1:
                wait = 2**attempt
                logger.warning("[VLM] %s — retrying in %ds", last, wait)
                time.sleep(wait)

        raise RuntimeError(f"VLM API failed after {self.max_retries} attempts: {last}")

    def close(self) -> None:
        self.session.close()

    # ------------------------------------------------------------------
    # Readiness
    # ------------------------------------------------------------------

    def health_check(self) -> bool:
        try:
            return self.session.get(f"{self.base_url}/models", timeout=5.0).status_code == 200
        except requests.exceptions.RequestException:
            return False

    def wait_for_server(self, timeout: float = 120.0) -> None:
        """Block until the server answers, then auto-detect the model if unset."""
        start = time.time()
        while time.time() - start < timeout:
            if self.health_check():
                if self.model is None:
                    self._detect_model()
                return
            time.sleep(2)
        raise TimeoutError(f"VLM server at {self.base_url} not ready after {timeout}s")

    def _detect_model(self) -> None:
        """Name the first model the server advertises.

        Raises rather than falling back to a placeholder: a wrong model name produces a
        confusing 404 on every step, which is worse than failing at startup.
        """
        resp = self.session.get(f"{self.base_url}/models", timeout=5.0)
        resp.raise_for_status()
        models = (resp.json() or {}).get("data") or []
        if not models:
            raise RuntimeError(
                f"VLM server at {self.base_url} advertises no models; pass --model explicitly."
            )
        self.model = models[0]["id"]
        logger.info("[VLM] auto-detected model: %s", self.model)


class _UnsupportedField(Exception):
    """The server rejected a request field rather than the request itself."""


def _mentions_unknown_field(body: str) -> bool:
    """Whether a 400 body looks like 'you sent a field I do not know'.

    Deliberately conservative: a 400 that is about the *content* of the request (an image
    the model cannot decode, a context overflow) must keep propagating rather than being
    retried with a different spelling.
    """
    lowered = body.lower()
    field_named = any(
        name in lowered
        for name in ("guided_choice", "structured_outputs", "guided_decoding",
                     "reasoning_effort", "chat_template_kwargs")
    )
    complaint = any(
        phrase in lowered
        for phrase in ("extra inputs", "unexpected keyword", "not permitted", "unknown field",
                       "not supported", "unrecognized", "additional properties", "invalid_request")
    )
    return field_named and complaint
