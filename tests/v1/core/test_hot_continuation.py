# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for HOT continuation ownership transfer and scheduler gates."""

import pytest
import torch

import vllm.envs as envs
from vllm.config import CacheConfig, ModelConfig, SchedulerConfig, VllmConfig
from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.hot_continuation import (
    HotContinuationCheckpoint,
    reconstruct_continuation_tokens,
)
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.core.single_type_kv_cache_manager import (
    MambaManager,
    register_all_kvcache_specs,
)
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    MambaSpec,
)
from vllm.v1.request import Request
from vllm.v1.structured_output import StructuredOutputManager

pytestmark = pytest.mark.cpu_test


@pytest.fixture(autouse=True)
def _auto_init_hash_fn():
    init_none_hash(sha256)


def test_reconstructs_tail_and_rejects_wrong_full_history():
    prefix = (10, 11, 12)

    assert reconstruct_continuation_tokens(prefix, 13, (20, 21)) == (
        (10, 11, 12, 13, 20, 21),
        True,
    )
    assert reconstruct_continuation_tokens(prefix, 13, (10, 11, 12, 13, 20)) == (
        (10, 11, 12, 13, 20),
        False,
    )
    assert reconstruct_continuation_tokens(prefix, 13, (9, 11, 12, 13, 20)) is None


def test_reconstruct_continuation_tokens_boundary_rules():
    """The successor prompt carries only the post-boundary tail."""
    forwarded = (1, 2, 3, 4)
    last_token = 5

    assert reconstruct_continuation_tokens(forwarded, last_token, ()) == (
        (1, 2, 3, 4, 5),
        True,
    )
    assert reconstruct_continuation_tokens(forwarded, last_token, (1, 2, 3, 4, 5)) == (
        (1, 2, 3, 4, 5),
        False,
    )
    assert (
        reconstruct_continuation_tokens(forwarded, last_token, (1, 2, 3, 4, 6, 7))
        is None
    )
    assert reconstruct_continuation_tokens((), last_token, (5, 6, 7)) == (
        (5, 6, 7),
        False,
    )
    assert reconstruct_continuation_tokens((), last_token, ()) == (
        (5,),
        True,
    )


def _make_align_hybrid_config(block_size: int, num_blocks: int) -> KVCacheConfig:
    return KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["full_attention"],
                FullAttentionSpec(
                    block_size=block_size,
                    num_kv_heads=1,
                    head_size=1,
                    dtype=torch.float32,
                ),
            ),
            KVCacheGroupSpec(
                ["mamba"],
                MambaSpec(
                    block_size=block_size,
                    shapes=((1, 1),),
                    dtypes=(torch.float32,),
                    mamba_cache_mode="align",
                ),
            ),
        ],
    )


def _make_hot_scheduler(block_size: int = 16, num_blocks: int = 100) -> Scheduler:
    model_config = ModelConfig(
        model="facebook/opt-125m",
        trust_remote_code=True,
        dtype="float16",
        seed=42,
        skip_tokenizer_init=True,
    )
    vllm_config = VllmConfig(
        scheduler_config=SchedulerConfig(
            max_num_seqs=1,
            max_num_batched_tokens=8192,
            max_model_len=8192,
            enable_chunked_prefill=True,
            is_encoder_decoder=False,
            watermark=0.0,
        ),
        model_config=model_config,
        cache_config=CacheConfig(
            block_size=block_size,
            enable_prefix_caching=True,
            mamba_cache_mode="align",
        ),
    )
    vllm_config.cache_config.num_gpu_blocks = num_blocks
    kv_cache_config = _make_align_hybrid_config(block_size, num_blocks)
    register_all_kvcache_specs(vllm_config)
    return Scheduler(
        vllm_config=vllm_config,
        kv_cache_config=kv_cache_config,
        structured_output_manager=StructuredOutputManager(vllm_config),
        block_size=block_size,
        hash_block_size=block_size,
        log_stats=True,
    )


def _make_hot_request(
    request_id: str,
    prompt_token_ids: list[int],
    *,
    continuation_handle: str | None = None,
    cache_salt: str | None = None,
    output_token_ids: list[int] | None = None,
    block_size: int = 16,
) -> Request:
    sampling_params = SamplingParams(
        max_tokens=64,
        extra_args=(
            {"continuation_handle": continuation_handle}
            if continuation_handle is not None
            else None
        ),
    )
    request = Request(
        request_id=request_id,
        prompt_token_ids=list(prompt_token_ids),
        sampling_params=sampling_params,
        pooling_params=None,
        cache_salt=cache_salt,
        block_hasher=get_request_block_hasher(block_size, sha256),
    )
    if output_token_ids:
        request.append_output_token_ids(list(output_token_ids))
    return request


def _finish_and_save(
    scheduler: Scheduler,
    request: Request,
    mamba_state_index: int | None = None,
) -> str | None:
    scheduler.kv_cache_manager.allocate_slots(
        request,
        request.num_tokens,
        0,
        scheduler.kv_cache_manager.empty_kv_cache_blocks,
    )
    request.num_computed_tokens = request.num_tokens - 1
    if mamba_state_index is not None:
        _mamba_manager(scheduler).last_state_block_idx[request.request_id] = (
            mamba_state_index
        )
    return scheduler._save_hot(request)


def _save_owner_request(
    scheduler: Scheduler,
    prompt_token_ids: list[int],
    output_token_ids: list[int],
    mamba_state_index: int | None = None,
) -> str:
    request = _make_hot_request(
        "owner",
        prompt_token_ids,
        output_token_ids=output_token_ids,
    )
    handle = _finish_and_save(scheduler, request, mamba_state_index)
    assert handle is not None
    return handle


def _mamba_manager(scheduler: Scheduler) -> MambaManager:
    for manager in scheduler.kv_cache_manager.coordinator.single_type_managers:
        if isinstance(manager, MambaManager):
            return manager
    raise AssertionError("hybrid scheduler is missing a Mamba manager")


def test_save_hot_detaches_blocks_and_records_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(envs, "VLLM_ENABLE_HOT_CONTINUATION", True)
    scheduler = _make_hot_scheduler()
    prompt_token_ids = list(range(40))
    output_token_ids = [100, 101, 102, 103, 104]
    request = _make_hot_request(
        "owner", prompt_token_ids, output_token_ids=output_token_ids
    )

    handle = _finish_and_save(scheduler, request)

    assert handle is not None
    checkpoint = scheduler.hot_checkpoint
    assert isinstance(checkpoint, HotContinuationCheckpoint)
    assert checkpoint.handle == handle
    assert checkpoint.forwarded_tokens == tuple(
        prompt_token_ids + output_token_ids[:-1]
    )
    assert checkpoint.last_token == output_token_ids[-1]
    assert checkpoint.boundary == len(prompt_token_ids) + len(output_token_ids) - 1
    assert checkpoint.fingerprint == scheduler._hot_fingerprint(request)
    assert scheduler.kv_cache_manager.get_blocks("owner").blocks == ((), ())


def test_claim_hot_tail_only_restores_state(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(envs, "VLLM_ENABLE_HOT_CONTINUATION", True)
    scheduler = _make_hot_scheduler()
    prompt_token_ids = list(range(40))
    output_token_ids = [100, 101, 102, 103, 104]
    mamba_state_index = 2
    handle = _save_owner_request(
        scheduler,
        prompt_token_ids,
        output_token_ids,
        mamba_state_index,
    )
    assert handle is not None
    checkpoint_block_ids = tuple(
        [block.block_id for block in group_blocks]
        for group_blocks in scheduler.hot_checkpoint.blocks.blocks
    )
    successor = _make_hot_request(
        "successor", [200, 201, 202], continuation_handle=handle
    )

    assert scheduler._claim_hot(successor) is True

    boundary = len(prompt_token_ids) + len(output_token_ids) - 1
    expected_prompt = prompt_token_ids + output_token_ids + [200, 201, 202]
    assert successor.prompt_token_ids == expected_prompt
    assert successor.num_computed_tokens == boundary
    assert successor.hot_claimed is True
    assert scheduler.hot_checkpoint is None
    assert scheduler.kv_cache_manager.get_block_ids("successor") == (
        checkpoint_block_ids
    )
    assert (
        _mamba_manager(scheduler).last_state_block_idx["successor"] == mamba_state_index
    )


def test_claim_hot_full_history_rejects_token_mismatch(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(envs, "VLLM_ENABLE_HOT_CONTINUATION", True)
    scheduler = _make_hot_scheduler()
    handle = _save_owner_request(scheduler, list(range(40)), [100, 101, 102, 103, 104])
    wrong_history = [0] * 40 + [100, 101, 102, 103, 999, 200]
    successor = _make_hot_request(
        "successor", wrong_history, continuation_handle=handle
    )

    assert scheduler._claim_hot(successor) is False

    assert scheduler.hot_checkpoint is None
    assert successor.num_computed_tokens == 0
    assert successor.hot_claimed is False
    assert successor.prompt_token_ids == wrong_history


def test_claim_hot_rejects_wrong_handle(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(envs, "VLLM_ENABLE_HOT_CONTINUATION", True)
    scheduler = _make_hot_scheduler()
    handle = _save_owner_request(scheduler, list(range(40)), [100, 101, 102, 103, 104])
    successor = _make_hot_request(
        "successor", [200], continuation_handle="not-" + handle
    )

    assert scheduler._claim_hot(successor) is False

    assert scheduler.hot_checkpoint is None
    assert successor.hot_claimed is False


def test_claim_hot_rejects_fingerprint_mismatch(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(envs, "VLLM_ENABLE_HOT_CONTINUATION", True)
    scheduler = _make_hot_scheduler()
    handle = _save_owner_request(scheduler, list(range(40)), [100, 101, 102, 103, 104])
    successor = _make_hot_request(
        "successor", [200], continuation_handle=handle, cache_salt="other"
    )

    assert scheduler._claim_hot(successor) is False

    assert scheduler.hot_checkpoint is None


def test_save_hot_requires_off_by_one_boundary(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(envs, "VLLM_ENABLE_HOT_CONTINUATION", True)
    scheduler = _make_hot_scheduler()
    request = _make_hot_request(
        "owner", list(range(40)), output_token_ids=[100, 101, 102, 103, 104]
    )
    scheduler.kv_cache_manager.allocate_slots(
        request,
        request.num_tokens,
        0,
        scheduler.kv_cache_manager.empty_kv_cache_blocks,
    )
    request.num_computed_tokens = request.num_tokens

    assert scheduler._save_hot(request) is None
    assert scheduler.hot_checkpoint is None


def test_save_hot_replaces_previous_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(envs, "VLLM_ENABLE_HOT_CONTINUATION", True)
    scheduler = _make_hot_scheduler()
    first_handle = _save_owner_request(
        scheduler, list(range(40)), [100, 101, 102, 103, 104]
    )
    first_checkpoint = scheduler.hot_checkpoint
    assert first_handle is not None

    second_handle = _save_owner_request(scheduler, list(range(50, 80)), [200, 201, 202])

    assert second_handle is not None
    assert second_handle != first_handle
    assert scheduler.hot_checkpoint is not first_checkpoint
    assert scheduler.hot_checkpoint.forwarded_tokens == tuple(
        list(range(50, 80)) + [200, 201]
    )


def test_release_hot_returns_blocks_to_pool(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(envs, "VLLM_ENABLE_HOT_CONTINUATION", True)
    scheduler = _make_hot_scheduler(num_blocks=32)
    free_before = scheduler.kv_cache_manager.block_pool.get_num_free_blocks()
    _save_owner_request(scheduler, list(range(40)), [100, 101, 102, 103, 104])
    free_after_save = scheduler.kv_cache_manager.block_pool.get_num_free_blocks()
    assert free_after_save < free_before

    scheduler._release_hot()

    assert scheduler.hot_checkpoint is None
    assert (
        scheduler.kv_cache_manager.block_pool.get_num_free_blocks() >= free_after_save
    )


def test_save_hot_disabled_when_multiple_requests_can_run(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(envs, "VLLM_ENABLE_HOT_CONTINUATION", True)
    scheduler = _make_hot_scheduler()
    scheduler.max_num_running_reqs = 2
    request = _make_hot_request("owner", list(range(40)), output_token_ids=[100, 101])

    assert _finish_and_save(scheduler, request) is None
    assert scheduler.hot_checkpoint is None


def test_hot_disables_async_scheduling(monkeypatch: pytest.MonkeyPatch):
    """HOT save must not race a pipelined decode step."""
    from types import SimpleNamespace

    from vllm.model_executor.models.config import MambaModelConfig

    monkeypatch.setattr(envs, "VLLM_ENABLE_HOT_CONTINUATION", True)
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(
            architecture="TestMamba",
            supports_mamba_prefix_caching=True,
            max_model_len=1024,
        ),
        cache_config=SimpleNamespace(
            enable_prefix_caching=True,
            mamba_cache_mode="align",
            mamba_block_size=16,
        ),
        scheduler_config=SimpleNamespace(
            async_scheduling=True,
            enable_chunked_prefill=True,
        ),
    )

    MambaModelConfig.verify_and_update_config(vllm_config)

    assert vllm_config.scheduler_config.async_scheduling is False
