# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare cold, prefix-cache, and HOT multi-turn Chat API latency."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen


@dataclass
class TurnResult:
    mode: str
    turn: int
    ttft_ms: float
    latency_ms: float
    prompt_tokens: int | None
    completion_tokens: int | None
    request_chars: int
    continuation_handle: bool
    output_text: str
    finish_reason: str | None
    output_token_ids: list[int]


def percentile(values: list[float], fraction: float) -> float:
    values = sorted(values)
    return values[min(len(values) - 1, round((len(values) - 1) * fraction))]


def summarize_metric(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.mean(values),
        "p50": percentile(values, 0.50),
        "p90": percentile(values, 0.90),
        "min": min(values),
        "max": max(values),
    }


def summarize(results: list[TurnResult]) -> dict[str, Any]:
    ttft = [result.ttft_ms for result in results]
    latency = [result.latency_ms for result in results]
    summary: dict[str, Any] = {
        "mode": results[0].mode,
        "turns": len(results),
        "ttft_ms": summarize_metric(ttft),
        "latency_ms": summarize_metric(latency),
        "prompt_tokens": [result.prompt_tokens for result in results],
        "completion_tokens": [result.completion_tokens for result in results],
        "hot_handle_turns": sum(result.continuation_handle for result in results),
    }
    if len(results) > 1:
        summary["steady_state_turns"] = len(results) - 1
        summary["steady_state_ttft_ms"] = summarize_metric(ttft[1:])
        summary["steady_state_latency_ms"] = summarize_metric(latency[1:])
    return summary


def make_messages(
    mode: str,
    turn: int,
    history: list[dict[str, str]],
    user_message: str,
) -> list[dict[str, str]]:
    if mode == "hot" and turn > 0:
        return [{"role": "user", "content": user_message}]
    return [*history, {"role": "user", "content": user_message}]


def send_chat(
    url: str,
    payload: dict[str, Any],
    timeout_sec: float,
) -> tuple[
    float,
    float,
    str,
    int | None,
    int | None,
    str | None,
    str | None,
    list[int],
]:
    request = Request(
        f"{url.rstrip('/')}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    first_token_at: float | None = None
    content = ""
    handle = None
    prompt_tokens = None
    completion_tokens = None
    finish_reason = None
    output_token_ids: list[int] = []

    try:
        response = urlopen(request, timeout=timeout_sec)
    except HTTPError as error:
        detail = error.read().decode(errors="replace")
        raise RuntimeError(f"HTTP {error.code}: {detail}") from error

    with response:
        for raw_line in response:
            line = raw_line.decode().strip()
            if not line.startswith("data: "):
                continue
            data = line[6:]
            if data == "[DONE]":
                continue
            event = json.loads(data)
            handle = event.get("continuation_handle") or handle
            usage = event.get("usage")
            if usage:
                prompt_tokens = usage.get("prompt_tokens")
                completion_tokens = usage.get("completion_tokens")
            choices = event.get("choices") or []
            if not choices:
                continue
            delta = choices[0].get("delta") or {}
            finish_reason = choices[0].get("finish_reason") or finish_reason
            output_token_ids.extend(choices[0].get("token_ids") or [])
            generated = delta.get("content") or delta.get("reasoning_content")
            if generated:
                if first_token_at is None:
                    first_token_at = time.perf_counter()
                content += generated

    finished = time.perf_counter()
    if first_token_at is None:
        raise RuntimeError("The stream contained no generated token")
    return (
        (first_token_at - started) * 1000,
        (finished - started) * 1000,
        content,
        prompt_tokens,
        completion_tokens,
        handle,
        finish_reason,
        output_token_ids,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="qwen-nvfp4")
    parser.add_argument("--mode", choices=("cold", "prefix", "hot"), required=True)
    parser.add_argument("--turns", type=int, default=6)
    parser.add_argument("--max-tokens", type=int, default=4)
    parser.add_argument("--context-chars", type=int, default=4000)
    parser.add_argument("--timeout-sec", type=float, default=300)
    parser.add_argument("--disable-thinking", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser


def run(args: argparse.Namespace) -> list[TurnResult]:
    if args.turns < 1 or args.max_tokens < 1 or args.context_chars < 1:
        raise ValueError("turns, max-tokens, and context-chars must be positive")

    context = (
        "The following persistent context is part of one local Qwen agent session. "
        "Keep it available for later turns. "
    ) * (args.context_chars // 108 + 1)
    context = context[: args.context_chars]
    history = [{"role": "system", "content": context}]
    handle = None
    results = []

    for turn in range(args.turns):
        user_message = (
            f"Turn {turn + 1}: summarize the persistent context in exactly "
            f"{args.max_tokens} short words."
        )
        messages = make_messages(args.mode, turn, history, user_message)
        payload: dict[str, Any] = {
            "model": args.model,
            "messages": messages,
            "max_tokens": args.max_tokens,
            "temperature": 0,
            "seed": 0,
            "stream": True,
            "return_token_ids": True,
            "stream_options": {"include_usage": True},
        }
        if args.disable_thinking:
            payload["chat_template_kwargs"] = {"enable_thinking": False}
        if args.mode == "hot" and handle is not None:
            payload["continuation_handle"] = handle

        (
            ttft_ms,
            latency_ms,
            content,
            prompt_tokens,
            completion_tokens,
            new_handle,
            finish_reason,
            output_token_ids,
        ) = send_chat(args.url, payload, args.timeout_sec)
        if args.mode == "hot" and turn > 0 and handle is None:
            raise RuntimeError("HOT did not return a handle for the previous turn")
        if args.mode == "hot" and not new_handle:
            raise RuntimeError(f"HOT did not return a handle on turn {turn + 1}")
        handle = new_handle if args.mode == "hot" else None
        history.extend(
            [
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": content},
            ]
        )
        result = TurnResult(
            mode=args.mode,
            turn=turn + 1,
            ttft_ms=ttft_ms,
            latency_ms=latency_ms,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            request_chars=sum(len(message["content"]) for message in messages),
            continuation_handle=bool(new_handle),
            output_text=content,
            finish_reason=finish_reason,
            output_token_ids=output_token_ids,
        )
        results.append(result)
        print(
            f"{args.mode:6} turn={result.turn} ttft_ms={result.ttft_ms:8.2f} "
            f"latency_ms={result.latency_ms:8.2f} "
            f"prompt_tokens={result.prompt_tokens} "
            f"request_chars={result.request_chars}"
        )
    return results


def main() -> None:
    args = build_parser().parse_args()
    results = run(args)
    summary = summarize(results)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.output:
        args.output.write_text(
            json.dumps(
                {"results": [asdict(result) for result in results], "summary": summary},
                ensure_ascii=False,
                indent=2,
            )
            + "\n"
        )


if __name__ == "__main__":
    main()
