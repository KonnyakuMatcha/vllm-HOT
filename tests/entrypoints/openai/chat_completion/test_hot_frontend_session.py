# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the opt-in HOT frontend session manager."""

from types import SimpleNamespace
from typing import Any

from vllm.entrypoints.openai.chat_completion.hot_session import (
    HotFrontendSessionManager,
)


def _request(messages: list[dict[str, Any]], **overrides: Any) -> Any:
    values: dict[str, Any] = {
        "messages": messages,
        "model": "test-model",
        "cache_salt": "salt",
        "chat_template": None,
        "chat_template_kwargs": None,
        "tools": None,
        "tool_choice": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


SYSTEM = {"role": "system", "content": "system"}
USER1 = {"role": "user", "content": "hello"}
ASSISTANT1 = {"role": "assistant", "content": "hi"}
USER2 = {"role": "user", "content": "again"}
TOOL_RESULT = {"role": "tool", "content": "result", "tool_call_id": "call-1"}


def test_first_request_passthrough_and_next_turn_trims_to_new_messages():
    manager = HotFrontendSessionManager()
    first = manager.prepare("session", _request([SYSTEM, USER1]))

    assert first.used_handle is False
    assert first.continuation_handle is None
    assert first.messages == [SYSTEM, USER1]

    manager.record_response("session", first.original_messages, first.context, "h1")

    second = manager.prepare("session", _request([SYSTEM, USER1, ASSISTANT1, USER2]))

    assert second.used_handle is True
    assert second.continuation_handle == "h1"
    assert second.messages == [USER2]


def test_full_history_without_echoed_assistant_keeps_new_user_message():
    manager = HotFrontendSessionManager()
    first = manager.prepare("session", _request([SYSTEM, USER1]))
    manager.record_response("session", first.original_messages, first.context, "h1")

    second = manager.prepare("session", _request([SYSTEM, USER1, USER2]))

    assert second.used_handle is True
    assert second.messages == [USER2]


def test_tool_result_suffix_drops_previous_assistant_message():
    manager = HotFrontendSessionManager()
    first = manager.prepare("session", _request([SYSTEM, USER1]))
    manager.record_response("session", first.original_messages, first.context, "h1")

    assistant_tool = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call-1",
                "type": "function",
                "function": {"name": "lookup", "arguments": "{}"},
            }
        ],
    }
    second = manager.prepare(
        "session",
        _request([SYSTEM, USER1, assistant_tool, TOOL_RESULT]),
    )

    assert second.used_handle is True
    assert second.messages == [TOOL_RESULT]


def test_diverged_history_drops_handle_and_sends_full_request():
    manager = HotFrontendSessionManager()
    first = manager.prepare("session", _request([SYSTEM, USER1]))
    manager.record_response("session", first.original_messages, first.context, "h1")

    diverged = manager.prepare(
        "session",
        _request([{"role": "system", "content": "changed"}, USER1, USER2]),
    )

    assert diverged.used_handle is False
    assert diverged.continuation_handle is None
    assert diverged.messages == [
        {"role": "system", "content": "changed"},
        USER1,
        USER2,
    ]


def test_context_change_drops_handle():
    manager = HotFrontendSessionManager()
    first = manager.prepare("session", _request([SYSTEM, USER1]))
    manager.record_response("session", first.original_messages, first.context, "h1")

    changed = manager.prepare(
        "session",
        _request([SYSTEM, USER1, ASSISTANT1, USER2], cache_salt="other"),
    )

    assert changed.used_handle is False
    assert changed.continuation_handle is None
    assert changed.messages == [SYSTEM, USER1, ASSISTANT1, USER2]


def test_empty_suffix_drops_handle():
    manager = HotFrontendSessionManager()
    first = manager.prepare("session", _request([SYSTEM, USER1]))
    manager.record_response("session", first.original_messages, first.context, "h1")

    same = manager.prepare("session", _request([SYSTEM, USER1]))

    assert same.used_handle is False
    assert same.continuation_handle is None

    # The session was dropped, so the next request is a full pass-through too.
    after = manager.prepare("session", _request([SYSTEM, USER1, USER2]))
    assert after.used_handle is False


def test_ttl_expiry_drops_state():
    manager = HotFrontendSessionManager(ttl_seconds=-1)
    first = manager.prepare("session", _request([SYSTEM, USER1]))
    manager.record_response("session", first.original_messages, first.context, "h1")

    second = manager.prepare("session", _request([SYSTEM, USER1, ASSISTANT1, USER2]))

    assert second.used_handle is False
    assert second.continuation_handle is None


def test_record_failure_drops_state():
    manager = HotFrontendSessionManager()
    first = manager.prepare("session", _request([SYSTEM, USER1]))
    manager.record_response("session", first.original_messages, first.context, "h1")
    manager.record_response("session", first.original_messages, first.context, None)

    second = manager.prepare("session", _request([SYSTEM, USER1, ASSISTANT1, USER2]))

    assert second.used_handle is False
