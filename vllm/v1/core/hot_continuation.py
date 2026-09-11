# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks


def reconstruct_continuation_tokens(
    forwarded_tokens: tuple[int, ...],
    last_token: int,
    request_tokens: tuple[int, ...],
) -> tuple[tuple[int, ...], bool] | None:
    """Validate a full history or build one from a tail-only request."""
    boundary_tokens = forwarded_tokens + (last_token,)
    boundary = len(forwarded_tokens)
    if len(request_tokens) > boundary:
        if request_tokens[: boundary + 1] != boundary_tokens:
            return None
        return request_tokens, False
    return boundary_tokens + request_tokens, True


@dataclass
class HotContinuationCheckpoint:
    """The single resident continuation state for a HOT-enabled worker."""

    handle: str
    forwarded_tokens: tuple[int, ...]
    last_token: int
    fingerprint: str
    boundary: int
    tail_valid_tokens: tuple[int, ...]
    blocks: KVCacheBlocks
    mamba_state_block_indices: tuple[int | None, ...]
