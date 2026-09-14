# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Frontend session tracking for HOT continuation.

The engine-side HOT implementation transfers exact model state between two
requests, so a successor request must contain only the tokens generated after
the previous checkpoint.  Chat clients, however, are used to sending the full
conversation history -- OpenAI's Chat Completions API is stateless.

This module adds an opt-in, API-server-side session cache.  It remembers the
messages that produced the previous checkpoint and trims the next request down
to the messages added after that checkpoint.  A leading assistant message is
dropped as well: its tokens are already part of the engine-side checkpoint, and
the engine expects only the *new* tool/user messages that follow it.

The manager is intentionally a plain in-memory cache.  Multi-worker
deployments need sticky routing or a shared implementation; the default
single-API-worker deployment is covered here.
"""

from __future__ import annotations

import copy
import json
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

from vllm.entrypoints.chat_utils import ChatCompletionMessageParam
from vllm.logger import init_logger

logger = init_logger(__name__)

_DEFAULT_MAX_SESSIONS = 4096


def _message_key(message: Any) -> str:
    """Return a stable, JSON-friendly key for a chat message.

    ``default=str`` keeps this usable for multimodal content objects even when
    they are not directly JSON serializable.
    """
    return json.dumps(
        message,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    )


def _request_context(request: Any) -> tuple[Any, ...]:
    """Return the request fields that make a HOT checkpoint reusable.

    A checkpoint is tied to the template, tools, model/LoRA and cache salt
    that were used to create it.  If any of those change, the frontend must
    fall back to a full request rather than blindly feeding a stale suffix to
    the engine.
    """
    return (
        getattr(request, "model", None),
        getattr(request, "cache_salt", None),
        _message_key(getattr(request, "chat_template", None)),
        _message_key(getattr(request, "chat_template_kwargs", None)),
        _message_key(getattr(request, "tools", None)),
        _message_key(getattr(request, "tool_choice", None)),
    )


def _longest_common_prefix(left: list[Any], right: list[Any]) -> int:
    limit = min(len(left), len(right))
    index = 0
    while index < limit and _message_key(left[index]) == _message_key(right[index]):
        index += 1
    return index


@dataclass
class HotSessionPlan:
    """How one incoming chat request should be sent to the engine."""

    messages: list[ChatCompletionMessageParam]
    continuation_handle: str | None
    used_handle: bool
    original_messages: list[ChatCompletionMessageParam]
    context: tuple[Any, ...]


@dataclass
class _HotSessionState:
    messages: list[ChatCompletionMessageParam]
    continuation_handle: str
    context: tuple[Any, ...]
    last_access: float


class HotFrontendSessionManager:
    """Track ``session_id`` -> previous full message list + HOT handle."""

    def __init__(
        self,
        *,
        ttl_seconds: float = 300.0,
        max_sessions: int = _DEFAULT_MAX_SESSIONS,
    ) -> None:
        self.ttl_seconds = ttl_seconds
        self.max_sessions = max_sessions
        self._sessions: OrderedDict[str, _HotSessionState] = OrderedDict()

    def _prune(self) -> None:
        now = time.monotonic()
        expired = [
            session_id
            for session_id, state in self._sessions.items()
            if now - state.last_access > self.ttl_seconds
        ]
        for session_id in expired:
            self._sessions.pop(session_id, None)
        while len(self._sessions) > self.max_sessions:
            self._sessions.popitem(last=False)

    def prepare(
        self,
        session_id: str,
        request: Any,
    ) -> HotSessionPlan:
        """Decide whether an incoming request can continue from a checkpoint.

        The returned plan always includes a deep copy of the client-provided
        messages in ``original_messages``.  ``messages`` is what should be
        rendered and sent to vLLM: either the original full history or a
        suffix consisting only of new messages.
        """
        self._prune()
        original_messages = copy.deepcopy(list(request.messages))
        context = _request_context(request)

        state = self._sessions.get(session_id)
        if state is None or state.context != context:
            self._sessions.pop(session_id, None)
            return HotSessionPlan(
                messages=original_messages,
                continuation_handle=None,
                used_handle=False,
                original_messages=original_messages,
                context=context,
            )

        prefix_len = _longest_common_prefix(state.messages, original_messages)
        if prefix_len < len(state.messages):
            logger.info(
                "HOT frontend session %s diverged from previous history; "
                "falling back to a full request",
                session_id,
            )
            self._sessions.pop(session_id, None)
            return HotSessionPlan(
                messages=original_messages,
                continuation_handle=None,
                used_handle=False,
                original_messages=original_messages,
                context=context,
            )

        suffix = copy.deepcopy(original_messages[prefix_len:])
        # The checkpoint already contains the previous generated assistant
        # message.  Chat clients normally echo that message back at the start
        # of the new suffix; the engine needs only what comes after it.
        if suffix and suffix[0].get("role") == "assistant":
            suffix = suffix[1:]

        if not suffix:
            logger.info(
                "HOT frontend session %s has no new messages after the "
                "previous assistant turn; falling back to a full request",
                session_id,
            )
            self._sessions.pop(session_id, None)
            return HotSessionPlan(
                messages=original_messages,
                continuation_handle=None,
                used_handle=False,
                original_messages=original_messages,
                context=context,
            )

        logger.debug(
            "HOT frontend session %s: sending %d suffix message(s) with handle",
            session_id,
            len(suffix),
        )
        return HotSessionPlan(
            messages=suffix,
            continuation_handle=state.continuation_handle,
            used_handle=True,
            original_messages=original_messages,
            context=context,
        )

    def record_response(
        self,
        session_id: str,
        original_messages: list[ChatCompletionMessageParam],
        context: tuple[Any, ...],
        continuation_handle: str | None,
    ) -> None:
        """Remember a successful response, or drop the session on error."""
        if not continuation_handle:
            self._sessions.pop(session_id, None)
            return

        self._prune()
        self._sessions[session_id] = _HotSessionState(
            messages=copy.deepcopy(original_messages),
            continuation_handle=continuation_handle,
            context=context,
            last_access=time.monotonic(),
        )
        self._sessions.move_to_end(session_id)
        while len(self._sessions) > self.max_sessions:
            self._sessions.popitem(last=False)

    def drop(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)

    def clear(self) -> None:
        self._sessions.clear()
