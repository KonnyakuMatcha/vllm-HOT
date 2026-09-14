# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""E2E test for the opt-in HOT JSON ``session_id`` frontend manager.

The client sends the full conversation history on every turn and only adds a
stable ``session_id`` to the JSON body.  The API server is responsible for
trimming the request to the post-checkpoint suffix and attaching the HOT
handle.
"""

import json
import os

import pytest
import requests

from tests.utils import RemoteOpenAIServer

MODEL = os.environ.get("HOT_E2E_MODEL", "Qwen/Qwen3.5-0.8B")
TP_SIZE = int(os.environ.get("HOT_E2E_TP", "1"))
MAX_TOKENS = 8

SYSTEM_PROMPT = "Remember this fact exactly. Project codename: ORBIT."
USER_TURNS = [
    "What is the project codename? Answer with one word.",
    "Repeat it. Answer with one word.",
    "What is 2+2? Answer with one number.",
]

pytestmark = [pytest.mark.hybrid_model, pytest.mark.slow_test]


def _server_args() -> list[str]:
    args = [
        "--enforce-eager",
        "--no-async-scheduling",
        "--max-num-seqs",
        "1",
        "--max-model-len",
        "1024",
        "--gpu-memory-utilization",
        "0.45",
        "--mamba-cache-mode",
        "align",
        "--enable-prefix-caching",
        "--no-enable-flashinfer-autotune",
    ]
    if TP_SIZE > 1:
        args.extend(["--tensor-parallel-size", str(TP_SIZE)])
    return args


def _chat(
    server: RemoteOpenAIServer,
    *,
    messages: list[dict],
    session_id: str | None = None,
) -> dict:
    payload = {
        "model": MODEL,
        "messages": messages,
        "max_tokens": MAX_TOKENS,
        "temperature": 0,
        "seed": 0,
        "stream": False,
        "return_token_ids": True,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    if session_id is not None:
        payload["session_id"] = session_id

    response = requests.post(
        server.url_for("v1/chat/completions"),
        json=payload,
        timeout=300,
    )
    response.raise_for_status()
    body = response.json()
    choice = body["choices"][0]
    return {
        "tokens": choice.get("token_ids"),
        "text": choice["message"].get("content") or "",
        "finish_reason": choice.get("finish_reason"),
        "handle": body.get("continuation_handle"),
    }


def _chat_stream(
    server: RemoteOpenAIServer,
    *,
    messages: list[dict],
    session_id: str | None = None,
) -> dict:
    payload = {
        "model": MODEL,
        "messages": messages,
        "max_tokens": MAX_TOKENS,
        "temperature": 0,
        "seed": 0,
        "stream": True,
        "return_token_ids": True,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    if session_id is not None:
        payload["session_id"] = session_id

    text_parts: list[str] = []
    token_ids: list[int] = []
    handle: str | None = None
    finish_reason: str | None = None
    with requests.post(
        server.url_for("v1/chat/completions"),
        json=payload,
        timeout=300,
        stream=True,
    ) as response:
        response.raise_for_status()
        for line in response.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data: "):
                continue
            data = line[len("data: ") :]
            if data == "[DONE]":
                break
            chunk = json.loads(data)
            if chunk.get("continuation_handle"):
                handle = chunk["continuation_handle"]
            for choice in chunk.get("choices") or []:
                delta = choice.get("delta") or {}
                content = delta.get("content")
                if content:
                    text_parts.append(content)
                if choice.get("token_ids"):
                    token_ids.extend(choice["token_ids"])
                if choice.get("finish_reason"):
                    finish_reason = choice["finish_reason"]

    return {
        "tokens": token_ids,
        "text": "".join(text_parts),
        "finish_reason": finish_reason,
        "handle": handle,
    }


def _run(server: RemoteOpenAIServer, *, use_session: bool) -> list[dict]:
    history = [{"role": "system", "content": SYSTEM_PROMPT}]
    results: list[dict] = []
    for turn_idx, user_message in enumerate(USER_TURNS, start=1):
        result = _chat(
            server,
            messages=[*history, {"role": "user", "content": user_message}],
            session_id="frontend-e2e-session" if use_session else None,
        )
        if result["finish_reason"] != "stop":
            raise AssertionError(
                f"turn {turn_idx} did not finish naturally: "
                f"{result['finish_reason']}"
            )
        if use_session and result["handle"] is None:
            raise AssertionError(f"turn {turn_idx} did not return a handle")
        results.append(result)
        history.extend(
            [
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": result["text"]},
            ]
        )
    return results


def _run_streaming_session(server: RemoteOpenAIServer) -> list[dict]:
    history = [{"role": "system", "content": SYSTEM_PROMPT}]
    results: list[dict] = []
    for turn_idx, user_message in enumerate(USER_TURNS, start=1):
        result = _chat_stream(
            server,
            messages=[*history, {"role": "user", "content": user_message}],
            session_id="frontend-e2e-stream-session",
        )
        if result["finish_reason"] != "stop":
            raise AssertionError(
                f"stream turn {turn_idx} did not finish naturally: "
                f"{result['finish_reason']}"
            )
        if result["handle"] is None:
            raise AssertionError(
                f"stream turn {turn_idx} did not return a handle"
            )
        results.append(result)
        history.extend(
            [
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": result["text"]},
            ]
        )
    return results


def test_full_history_json_session_id_matches_control():
    with RemoteOpenAIServer(
        MODEL,
        _server_args(),
        env_dict={"VLLM_ENABLE_HOT_CONTINUATION": "1"},
    ) as server:
        hot_results = _run(server, use_session=True)
        control_results = _run(server, use_session=False)
        stream_results = _run_streaming_session(server)

    for turn_idx, (hot, control) in enumerate(
        zip(hot_results, control_results, strict=True), start=1
    ):
        assert hot["tokens"] == control["tokens"], (
            f"turn {turn_idx}: token mismatch\n"
            f"  HOT    : {hot['tokens']}\n"
            f"  control: {control['tokens']}\n"
            f"  HOT text: {hot['text']!r}\n"
            f"  control text: {control['text']!r}"
        )

    for turn_idx, (stream, control) in enumerate(
        zip(stream_results, control_results, strict=True), start=1
    ):
        assert stream["tokens"] == control["tokens"], (
            f"stream turn {turn_idx}: token mismatch\n"
            f"  stream : {stream['tokens']}\n"
            f"  control: {control['tokens']}\n"
            f"  stream text: {stream['text']!r}\n"
            f"  control text: {control['text']!r}"
        )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q", "-s"]))
