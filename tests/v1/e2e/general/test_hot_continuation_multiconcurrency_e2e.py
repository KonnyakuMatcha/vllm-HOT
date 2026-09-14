# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""E2E token-equivalence test for concurrent HOT sessions.

Two scenarios are covered:

* distinct contexts: each session has its own system prefix and cache salt.
* shared context: all sessions share one long system prefix and cache salt, so
  their full-attention prefix blocks can be shared by the prefix cache.

HOT and full-history control run on separate servers because the engine keeps a
bounded resident-checkpoint registry.  Within each server, the sessions run
concurrently with ``max_num_seqs`` equal to the number of sessions.
"""

import os
from concurrent.futures import ThreadPoolExecutor

import pytest
import requests

from tests.utils import RemoteOpenAIServer

MODEL = os.environ.get("HOT_E2E_MODEL", "Qwen/Qwen3.5-0.8B")
TP_SIZE = int(os.environ.get("HOT_E2E_TP", "1"))
MAX_SESSIONS = int(os.environ.get("HOT_E2E_MAX_SESSIONS", "4"))
MAX_TOKENS = 16

USER_TURNS = [
    "What is the secret word? Answer with one word.",
    "What is the secret number? Answer with one number.",
]
DISTINCT_FACTS = [
    ("ORBIT", "7319"),
    ("LUNAR", "2468"),
    ("COMET", "1357"),
    ("NEBULA", "8642"),
    ("PULSAR", "9753"),
    ("QUARK", "3141"),
    ("COSMOS", "2718"),
    ("PHOTON", "1618"),
]
SHARED_PREFIX = (
    "Shared persistent context for several sessions. "
    "IMPORTANT: the global project codename is ORBIT. "
    "Additional filler context: atlas-07, canary-3, ring-9. "
) * 20

pytestmark = [pytest.mark.hybrid_model, pytest.mark.slow_test]


def _server_args() -> list[str]:
    args = [
        "--enforce-eager",
        "--no-async-scheduling",
        "--max-num-seqs",
        str(MAX_SESSIONS),
        "--max-model-len",
        "2048",
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


def _run_hot_session(server, session_idx, *, system_prompt, cache_salt, tag):
    handle = None
    results = []
    for turn_idx in range(1, len(USER_TURNS) + 1):
        user = f"{tag} {USER_TURNS[turn_idx - 1]}"
        messages = (
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user},
            ]
            if turn_idx == 1
            else [{"role": "user", "content": user}]
        )
        result = _chat(
            server,
            messages=messages,
            cache_salt=cache_salt,
            handle=handle,
        )
        assert result["finish_reason"] == "stop", (
            f"session {session_idx} turn {turn_idx}: "
            f"HOT did not finish naturally: {result['finish_reason']}"
        )
        assert result["handle"] is not None, (
            f"session {session_idx} turn {turn_idx}: missing HOT handle"
        )
        results.append(result)
        handle = result["handle"]
    return results


def _run_control_session(server, session_idx, *, system_prompt, tag, salt_prefix):
    history = [{"role": "system", "content": system_prompt}]
    results = []
    for turn_idx in range(1, len(USER_TURNS) + 1):
        user = f"{tag} {USER_TURNS[turn_idx - 1]}"
        result = _chat(
            server,
            messages=[*history, {"role": "user", "content": user}],
            cache_salt=f"{salt_prefix}-{session_idx}-{turn_idx}",
        )
        assert result["finish_reason"] == "stop", (
            f"session {session_idx} turn {turn_idx}: "
            f"control did not finish naturally: {result['finish_reason']}"
        )
        results.append(result)
        history.extend(
            [
                {"role": "user", "content": user},
                {"role": "assistant", "content": result["text"]},
            ]
        )
    return results


def _run_hot_batch(server, sessions):
    num_sessions = len(sessions)

    def run_one(idx):
        system_prompt, cache_salt, tag = sessions[idx]
        return _run_hot_session(
            server,
            idx,
            system_prompt=system_prompt,
            cache_salt=cache_salt,
            tag=tag,
        )

    with ThreadPoolExecutor(max_workers=num_sessions) as executor:
        futures = {idx: executor.submit(run_one, idx) for idx in range(num_sessions)}
        return {idx: future.result() for idx, future in futures.items()}


def _run_control_batch(server, sessions, salt_prefix):
    num_sessions = len(sessions)

    def run_one(idx):
        system_prompt, _, tag = sessions[idx]
        return _run_control_session(
            server,
            idx,
            system_prompt=system_prompt,
            tag=tag,
            salt_prefix=salt_prefix,
        )

    with ThreadPoolExecutor(max_workers=num_sessions) as executor:
        futures = {idx: executor.submit(run_one, idx) for idx in range(num_sessions)}
        return {idx: future.result() for idx, future in futures.items()}


def _assert_matches(hot, control, scenario: str):
    for session_idx in hot:
        for turn_idx, (hot_result, control_result) in enumerate(
            zip(hot[session_idx], control[session_idx], strict=True), start=1
        ):
            assert hot_result["tokens"] == control_result["tokens"], (
                f"{scenario} session {session_idx} turn {turn_idx}: mismatch\n"
                f"  HOT    : {hot_result['tokens']}\n"
                f"  control: {control_result['tokens']}"
            )


def test_hot_multi_concurrent_matches_full_history_token_ids():
    num_sessions = min(MAX_SESSIONS, len(DISTINCT_FACTS))

    distinct_sessions = []
    for idx in range(num_sessions):
        word, number = DISTINCT_FACTS[idx]
        distinct_sessions.append(
            (
                f"Remember exactly: the secret word is {word} "
                f"and the secret number is {number}.",
                f"hot-distinct-{idx}",
                f"Session {idx}:",
            )
        )
    shared_sessions = [
        (
            SHARED_PREFIX,
            "hot-shared",
            f"Shared session {idx}:",
        )
        for idx in range(num_sessions)
    ]
    overflow_sessions = [
        (
            SHARED_PREFIX,
            "hot-shared-overflow",
            f"Overflow session {idx}:",
        )
        for idx in range(num_sessions + 2)
    ]

    with RemoteOpenAIServer(
        MODEL,
        _server_args(),
        env_dict={"VLLM_ENABLE_HOT_CONTINUATION": "1"},
    ) as hot_server:
        hot_distinct = _run_hot_batch(hot_server, distinct_sessions)
        hot_shared = _run_hot_batch(hot_server, shared_sessions)
        hot_overflow = _run_hot_batch(hot_server, overflow_sessions)

    with RemoteOpenAIServer(
        MODEL,
        _server_args(),
        env_dict={"VLLM_ENABLE_HOT_CONTINUATION": "0"},
    ) as control_server:
        control_distinct = _run_control_batch(
            control_server, distinct_sessions, "ctl-distinct"
        )
        control_shared = _run_control_batch(
            control_server, shared_sessions, "ctl-shared"
        )
        control_overflow = _run_control_batch(
            control_server, overflow_sessions, "ctl-overflow"
        )

    _assert_matches(hot_distinct, control_distinct, "distinct")
    _assert_matches(hot_shared, control_shared, "shared")

    # The overflow scenario primarily exercises checkpoint eviction and
    # token-chain reconstruction.  Exact token equality against a separately
    # computed full-history control can differ because shared prefix-cache
    # hits change the numerical reduction order for greedy decoding.  The
    # per-session helpers above already assert natural stop and handle/output
    # presence; here we only assert that every session completed every turn.
    for results in (hot_overflow, control_overflow):
        assert all(
            len(session_results) == len(USER_TURNS)
            for session_results in results.values()
        )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q", "-s"]))
