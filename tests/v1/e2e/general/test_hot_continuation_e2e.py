# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""End-to-end token-equivalence test for HOT continuation.

HOT and full-history control run on separate servers because the HOT
implementation keeps exactly one resident checkpoint per engine.  A control
request completing on the same engine would overwrite the checkpoint that the
next HOT request needs.

For every turn the generated token IDs must match exactly.  Both branches also
must finish naturally with ``finish_reason == "stop"`` so the EOS save path is
exercised.
"""

import os

import pytest
import requests

from tests.utils import RemoteOpenAIServer

MODEL = os.environ.get("HOT_E2E_MODEL", "Qwen/Qwen3.5-0.8B")
TP_SIZE = int(os.environ.get("HOT_E2E_TP", "1"))
MAX_TOKENS = 24

SYSTEM_PROMPT = (
    "Remember exactly this fact. Project codename: ORBIT. Access code: 7319."
)
USER_TURNS = [
    "What is the project codename? Answer with one word.",
    "What is the access code? Answer with one number.",
    "Return the codename and the access code separated by a comma.",
    "What is the project codename? Answer with one word.",
    "What is 2+2? Answer with one number.",
    "Repeat the access code. Answer with one number.",
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


def _chat(server: RemoteOpenAIServer, *, messages, cache_salt, handle=None):
    payload = {
        "model": MODEL,
        "messages": messages,
        "max_tokens": MAX_TOKENS,
        "temperature": 0,
        "seed": 0,
        "stream": False,
        "return_token_ids": True,
        "cache_salt": cache_salt,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    if handle is not None:
        payload["continuation_handle"] = handle

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


def _run_hot(server: RemoteOpenAIServer) -> list[dict]:
    results: list[dict] = []
    hot_handle = None
    for turn_idx, user_message in enumerate(USER_TURNS, start=1):
        messages = (
            [{"role": "system", "content": SYSTEM_PROMPT}]
            + [{"role": "user", "content": user_message}]
            if turn_idx == 1
            else [{"role": "user", "content": user_message}]
        )
        result = _chat(
            server,
            messages=messages,
            cache_salt="hot-e2e",
            handle=hot_handle,
        )
        if result["finish_reason"] != "stop":
            raise AssertionError(
                f"HOT turn {turn_idx} did not finish naturally: "
                f"{result['finish_reason']}"
            )
        if result["handle"] is None:
            raise AssertionError(f"HOT turn {turn_idx} did not return a handle")
        results.append(result)
        hot_handle = result["handle"]
    return results


def _run_control(server: RemoteOpenAIServer) -> list[dict]:
    history = [{"role": "system", "content": SYSTEM_PROMPT}]
    results: list[dict] = []
    for turn_idx, user_message in enumerate(USER_TURNS, start=1):
        result = _chat(
            server,
            messages=[*history, {"role": "user", "content": user_message}],
            cache_salt=f"control-e2e-{turn_idx}",
        )
        if result["finish_reason"] != "stop":
            raise AssertionError(
                f"control turn {turn_idx} did not finish naturally: "
                f"{result['finish_reason']}"
            )
        results.append(result)
        history.extend(
            [
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": result["text"]},
            ]
        )
    return results


def test_hot_matches_full_history_token_ids():
    with RemoteOpenAIServer(
        MODEL,
        _server_args(),
        env_dict={"VLLM_ENABLE_HOT_CONTINUATION": "1"},
    ) as hot_server:
        hot_results = _run_hot(hot_server)

    with RemoteOpenAIServer(
        MODEL,
        _server_args(),
        env_dict={"VLLM_ENABLE_HOT_CONTINUATION": "0"},
    ) as control_server:
        control_results = _run_control(control_server)

    for turn_idx, (hot, control) in enumerate(
        zip(hot_results, control_results, strict=True), start=1
    ):
        assert hot["tokens"] == control["tokens"], (
            f"turn {turn_idx}: token mismatch\n"
            f"  HOT    : {hot['tokens']}\n"
            f"  control: {control['tokens']}\n"
            f"  HOT    text: {hot['text']!r}\n"
            f"  control text: {control['text']!r}"
        )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q", "-s"]))
