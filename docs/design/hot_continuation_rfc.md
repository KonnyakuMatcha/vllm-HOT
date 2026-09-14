# [RFC] Exact Single-Session HOT Continuation for Hybrid Mamba Models

Status: Draft / WIP, not yet proposed upstream.

## Motivation

Hybrid Mamba/attention models such as Qwen3.5/Qwen3.6 35B-A3B have two
different kinds of state:

- full-attention layers keep per-token KV entries in paged blocks;
- linear-attention / GDN layers keep one recurrent state per sequence.

For a single sequential agent session, the next turn usually appends only a
small user/tool tail.  Existing prefix caching can reuse a token prefix, but
for hybrid recurrent state it must reconstruct or replay the recurrent state
at a shared boundary.  HOT ("Hot Ongoing Transfer") targets the exact
successor case instead: a completed turn hands its live state directly to the
next request.

This RFC proposes a narrow, opt-in continuation path for one active session
per checkpoint.  It is intended to complement, not replace, general prefix
caching or application-directed shared-prefix checkpoints.

## Proposed Behavior

1. A completed request returns a server-issued `continuation_handle`.
2. The client sends only the new tail plus that handle on the next turn.
3. The scheduler validates the handle, model/session fingerprint, and exact
   token boundary.
4. On a hit, full-attention block ownership and the latest Mamba state slot
   are transferred to the successor request.
5. On a miss or eviction, the successor is recovered safely without
   prefilling only the new tail.

Checkpoint contents:

```text
HotContinuationCheckpoint
  handle
  forwarded_tokens / last_token
  fingerprint
  boundary
  full-attention block ownership
  Mamba state block indices
```

## Scheduler Lifecycle

- `_save_hot()` runs after a naturally stopped request is committed.
- `_claim_hot()` runs at admission before ordinary prefix-cache lookup.
- A checkpoint registry is keyed by `continuation_handle`.
- Capacity is bounded by the number of runnable sessions.
- Evicted block-backed checkpoints are demoted to a lightweight token chain.
- A tail-only successor whose checkpoint was evicted reconstructs the full
  prompt from the token chain and falls back to ordinary full prefill.
- TTLs bound dormant checkpoint and token-chain memory.
- Invalid or expired handles return HTTP 400 instead of silently producing a
  tail-only completion.

## Frontend `session_id` Adapter

Most OpenAI-compatible clients are stateful at the application layer but send
the full message history on every turn. Requiring them to remember a
`continuation_handle` and manually strip the assistant turn is therefore a
poor fit for the default Chat Completions API.

When `VLLM_ENABLE_HOT_CONTINUATION=1`, the API server can instead maintain an
opt-in mapping from a stable JSON `session_id` to:

- the full message list that produced the previous checkpoint;
- the server-issued `continuation_handle`; and
- a fingerprint of the template/tools/cache salt used to produce it.

On the next request the server computes the longest common prefix of the
client's full history. If the stored history is still a prefix, it sends only
the new suffix to the engine. A leading assistant message is dropped because
its tokens are already owned by the checkpoint. If the history diverges, the
session is discarded and the request falls back to a full prefill. The
explicit `continuation_handle` API remains available for clients that want to
implement the same optimization themselves.

The adapter is intentionally an in-memory API-server cache in this draft.
Multi-worker deployments need sticky routing or a shared session store; the
normal stateless Chat Completions behavior is unchanged when HOT is disabled.

## Relationship to Existing Work

### #55697 / #55873 / #55875 / #55876: Application-Directed Prefix Checkpoints

The #55697 series lets an application declare a shared prefix boundary and
coordinates producer/consumer scheduling so multiple consumers reuse a
producer's Mamba checkpoint.

HOT differs as follows:

- HOT is sequential continuation, not 1-to-N shared-prefix fan-out.
- HOT transfers live ownership of the exact latest boundary.
- HOT uses a server-issued handle, not an in-prompt marker.
- HOT does not require a shared prefix across sessions.
- HOT currently requires `async_scheduling=False`.

They could be complementary: #55697 targets shared-prefix multi-request
workloads; HOT targets one exact successor with a small tail.

### #52959: Internal State Checkpoints for Mamba Align Mode

#52959 derives intermediate Mamba checkpoints inside one forward pass.  HOT
does not change the kernel or chunking; it transfers the latest materialized
state slot across requests.  HOT could later benefit from the same internal
checkpoint primitive.

### #45702: Partial Cache Hits for Hybrid Models

#45702 introduces fine-grained hash-based partial prefix reuse.  HOT is
handle-based and exact, with no hash lookup or block matching.  A HOT miss
currently falls back to the ordinary path supported by that infrastructure.

## Evaluation

### Setup (this branch)

- Model: `Qwen/Qwen3.6-35B-A3B-NVFP4`
- TP=2, two RTX 5090 pairs
- 4 concurrent sessions, each with an independent ~20k-character context
  (~5.1k prompt tokens)
- 3 turns, `max_tokens=8`, greedy
- HOT server: async scheduling disabled
- Prefix server: async scheduling enabled
- Both: prefix caching enabled, `mamba_cache_mode=align`

### TTFT

| Mode | Turn 1 | Turn 2 | Turn 3 | Steady mean | Steady p50 |
| --- | ---: | ---: | ---: | ---: | ---: |
| HOT | 1453.3 ms | 417.2 ms | 181.4 ms | 299.3 ms | 237.7 ms |
| Prefix | 1501.0 ms | 561.3 ms | 291.2 ms | 426.3 ms | 383.6 ms |

HOT steady-state TTFT was about **29.8% lower** than the ordinary prefix
path in this workload.

A separate full-latency run with the same model/shape measured:

| Mode | Steady mean latency |
| --- | ---: |
| HOT | 570.6 ms |
| Prefix | 930.8 ms |

### Existing #55697 Results

The #55697 RFC reports, for Qwen3.5 35B on L40S and a 1-to-9 shared-prefix
workload with a ~624-token shared prefix:

- TTFT: 482 ms -> 248 ms (**-48.6%**)
- Throughput: **+110.2% QPS**
- FLOPs: **-53.4%**

### Interpretation

The numbers are not directly comparable because the workloads differ:

| Dimension | #55697 series | HOT |
| --- | --- | --- |
| Reuse shape | 1-to-N shared prefix | exact sequential successor |
| Client request | full suffix each turn | new tail only |
| Reuse source | shared prefix checkpoint + scheduler coordination | live state ownership transfer |
| Concurrent consumers | yes | per-session checkpoints |
| Async scheduling | part of existing scheduler design | currently unsupported |

HOT's measured benefit is smaller, but it covers a workload that shared-prefix
reuse does not optimize well: independent long contexts with one exact
sequential successor and a tiny tail.

## Correctness Status

- Single-session natural EOS token equivalence: verified.
- Multi-session token equivalence up to 8 concurrent sessions: verified.
- Shared-prefix + checkpoint eviction + token-chain fallback: verified.
- Invalid/expired handle: returns HTTP 400.
- JSON `session_id` full-history adapter: non-streaming and streaming token
  equivalence verified against a stateless control on a 3-turn session.
- Not yet verified: speculative decoding, KV connectors, multimodal inputs,
  PP > 1, LoRA, abort/preempt races under production load.

## Known Limitations

1. HOT currently requires `async_scheduling=False`.
   - An experimental relaxed path failed a 2-session, 6-turn token
     equivalence test.  Async support needs a request-level drain or state
     rollback transaction, not just a boundary adjustment.
2. The feature is still env-gated in this branch.
   - Upstream should use `CacheConfig` / `EngineArgs` options.
3. Checkpoint capacity is bounded by runnable sessions; overflow falls back
   to full prefill via token chains.
4. No producer-consumer 1-to-N sharing yet.
5. TTL and memory accounting are currently basic; no user-visible metrics.
6. The frontend `session_id` adapter is an in-memory, per-API-worker cache;
   multi-worker deployments need sticky routing or a shared store.
7. Tool-call and other non-text agent suffixes are structurally supported,
   but still need end-to-end coverage with a tool-capable model.

## Open Questions

1. Should this be a standalone mode, or a specialization inside the #55697
   prefix-checkpoint design?
2. Is a server-issued `continuation_handle` an acceptable API, or should the
   trigger be an in-prompt marker / `session_id` instead?
3. Should the first upstream contribution be only the ownership-transfer
   primitive, leaving API and async work to follow-ups?
4. What is the preferred fallback contract for an invalid/expired handle:
   400, or client-resend-full-history?

## Future Work

- Request-level drain / commit-fence state machine for async scheduling.
- Transactional or copy-on-write Mamba state / attention tail for safe async
  rollback.
- Shared checkpoint registry and producer-consumer coordination if merged
  with #55697.
- Speculative-decoding accepted-token repair.
- Metrics for checkpoint residency, hit/miss/eviction, and pinned bytes.
