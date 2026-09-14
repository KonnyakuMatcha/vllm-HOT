# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark concurrent multi-turn HOT vs prefix streaming TTFT."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen


@dataclass
class TurnResult:
    mode: str
    session: int
    turn: int
    ttft_ms: float
    latency_ms: float
    prompt_tokens: int | None
    continuation_handle: bool
    output_text: str
    finish_reason: str | None


def percentile(values: list[float], fraction: float) -> float:
    values = sorted(values)
    return values[min(len(values) - 1, round((len(values) - 1) * fraction))]


def summarize(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.mean(values),
        "p50": percentile(values, 0.50),
        "p90": percentile(values, 0.90),
        "min": min(values),
        "max": max(values),
    }


def stream_chat(
    url: str,
    model: str,
    messages: list[dict[str, str]],
    *,
    cache_salt: str,
    handle: str | None,
    max_tokens: int,
    timeout_sec: float,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0,
        "seed": 0,
        "stream": True,
        "return_token_ids": True,
        "stream_options": {"include_usage": True},
        "cache_salt": cache_salt,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    if handle is not None:
        payload["continuation_handle"] = handle

    request = Request(
        f"{url.rstrip('/')}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    try:
        response = urlopen(request, timeout=timeout_sec)
    except HTTPError as error:
        detail = error.read().decode(errors="replace")
        raise RuntimeError(f"HTTP {error.code}: {detail}") from error

    ttft_ms: float | None = None
    text = ""
    prompt_tokens: int | None = None
    new_handle: str | None = None
    finish_reason: str | None = None

    with response:
        for raw_line in response:
            line = raw_line.decode(errors="replace").strip()
            if not line.startswith("data: "):
                continue
            data = line[6:]
            if data == "[DONE]":
                continue
            event = json.loads(data)
            usage = event.get("usage")
            if usage:
                prompt_tokens = usage.get("prompt_tokens")
            new_handle = event.get("continuation_handle") or new_handle
            choices = event.get("choices") or []
            if not choices:
                continue
            choice = choices[0]
            finish_reason = choice.get("finish_reason") or finish_reason
            delta = choice.get("delta") or {}
            generated = delta.get("content") or delta.get("reasoning_content")
            if generated:
                if ttft_ms is None:
                    ttft_ms = (time.perf_counter() - started) * 1000
                text += generated

    if ttft_ms is None:
        raise RuntimeError("The stream contained no generated token")
    return {
        "ttft_ms": ttft_ms,
        "latency_ms": (time.perf_counter() - started) * 1000,
        "prompt_tokens": prompt_tokens,
        "handle": new_handle,
        "output_text": text,
        "finish_reason": finish_reason,
    }


def make_system_prompt(session: int, context_chars: int) -> str:
    filler = (
        f"Session {session}. This is persistent context for one concurrent "
        "multi-turn benchmark. Keep it available. Filler detail: "
        "atlas-07 canary-3 ring-9. "
    )
    return (filler * (context_chars // len(filler) + 1))[:context_chars]


def run_session(
    url: str,
    model: str,
    mode: str,
    session: int,
    *,
    turns: int,
    max_tokens: int,
    context_chars: int,
    timeout_sec: float,
) -> list[TurnResult]:
    system_prompt = make_system_prompt(session, context_chars)
    history: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]
    handle: str | None = None
    results: list[TurnResult] = []

    for turn in range(1, turns + 1):
        user = f"Turn {turn}: answer with one short word."
        if mode == "hot" and turn > 1:
            messages = [{"role": "user", "content": user}]
        else:
            messages = [*history, {"role": "user", "content": user}]

        response = stream_chat(
            url,
            model,
            messages,
            cache_salt=f"{mode}-{session}",
            handle=handle,
            max_tokens=max_tokens,
            timeout_sec=timeout_sec,
        )
        if mode == "hot":
            if not response["handle"]:
                raise RuntimeError(
                    f"HOT mode did not return a handle for session {session} "
                    f"turn {turn}"
                )
            handle = response["handle"]

        results.append(
            TurnResult(
                mode=mode,
                session=session,
                turn=turn,
                ttft_ms=response["ttft_ms"],
                latency_ms=response["latency_ms"],
                prompt_tokens=response["prompt_tokens"],
                continuation_handle=bool(response["handle"]),
                output_text=response["output_text"],
                finish_reason=response["finish_reason"],
            )
        )
        history.extend(
            [
                {"role": "user", "content": user},
                {"role": "assistant", "content": response["output_text"]},
            ]
        )
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", required=True)
    parser.add_argument("--mode", choices=("hot", "prefix"), required=True)
    parser.add_argument("--sessions", type=int, default=4)
    parser.add_argument("--turns", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--context-chars", type=int, default=20000)
    parser.add_argument("--timeout-sec", type=float, default=600)
    parser.add_argument("--output", type=Path)
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.sessions < 1 or args.turns < 1 or args.max_tokens < 1:
        raise ValueError("sessions, turns, and max-tokens must be positive")

    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.sessions) as executor:
        futures = {
            session: executor.submit(
                run_session,
                args.url,
                args.model,
                args.mode,
                session,
                turns=args.turns,
                max_tokens=args.max_tokens,
                context_chars=args.context_chars,
                timeout_sec=args.timeout_sec,
            )
            for session in range(args.sessions)
        }
        session_results = {
            session: future.result() for session, future in futures.items()
        }
    elapsed_sec = time.perf_counter() - started

    results = [
        result
        for session in range(args.sessions)
        for result in session_results[session]
    ]
    per_turn = []
    for turn in range(1, args.turns + 1):
        values = [result.ttft_ms for result in results if result.turn == turn]
        per_turn.append(summarize(values))
    steady = [result.ttft_ms for result in results if result.turn > 1]
    summary: dict[str, Any] = {
        "mode": args.mode,
        "sessions": args.sessions,
        "turns": args.turns,
        "elapsed_sec": elapsed_sec,
        "per_turn_ttft_ms": per_turn,
        "steady_state_ttft_ms": summarize(steady) if steady else None,
        "prompt_tokens": [result.prompt_tokens for result in results],
    }
    return {"results": [asdict(result) for result in results], "summary": summary}


def main() -> None:
    args = build_parser().parse_args()
    output = run(args)
    print(json.dumps(output["summary"], indent=2))
    if args.output:
        args.output.write_text(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    main()
