# IE12X HOT Continuation Migration Guide for vLLM

This document is the implementation brief for porting the IE12X continuation
optimizations into vLLM. It is written for an engineer who will implement the
work in the vLLM tree, with vLLM responsible for prefill and the IE12X-style
runtime responsible for the latency-sensitive continuation/decode path.

The target is not a general-purpose session database. It is a deliberately
small optimization for one active Qwen3.6-35B-A3B agent conversation at a
time.

## 1. Target Workload

IE12X is specialized for:

- one active agent session or one active conversation per worker;
- many sequential turns, usually one user/tool tail after the previous
  assistant response;
- a large accumulated context and a small new tail per turn;
- latency-sensitive time-to-first-token and prefill wall time;
- greedy or otherwise stable single-completion generation;
- Qwen3.6-35B-A3B on a Blackwell-class GPU, using the model's NVFP4 weights;
- explicit control of the request order, so a completed turn is normally
  followed by its successor.

This workload is different from a shared multi-tenant prefix cache. A prefix
cache optimizes repeated token prefixes that can be rediscovered by hashing
and block lookup. HOT (Hot Ongoing Transfer) keeps the exact live continuation
state from the last completed turn and transfers its ownership to the next
turn. It avoids both re-prefill and the recurrent-state replay that a hybrid
prefix cache may require.

The optimization is intentionally narrow. Do not add a session tree, an LRU
of arbitrary conversations, cross-process persistence, or a message-history
store as part of the first port.

## 2. Model State: Two Different Caches

Qwen3.6 has two attention families with different state semantics. Treating
them as one token-indexed KV array is the central migration mistake.

### 2.1 Full attention

Full-attention layers retain ordinary K/V entries for every forwarded token.
In vLLM these entries are represented by paged physical blocks and a logical
block table. The final block can be partially filled. HOT must transfer that
tail block with its valid-token count; rounding the boundary to a full block
would either expose uncomputed positions or force an unnecessary replay.

The full-attention part is naturally token-addressable. It can coexist with
the normal vLLM block allocator, provided the blocks are pinned or ownership is
transferred atomically while the checkpoint is live.

### 2.2 GDN / linear attention

GDN is a recurrent state derived from the Mamba family. Its live state is an
iterative state, not a collection of independently meaningful per-token KV
entries. The state is large and should not be discretized into ordinary prefix
blocks for this workload.

For the single-session path, retain exactly one latest GDN state at the
completed request boundary. The state is indivisible from the point of view of
HOT: either the whole state and its slot ownership transfer, or the request
falls back to normal prefill/replay. Do not align GDN to every full-attention
block merely to match a generic prefix-cache policy.

### 2.3 The boundary is shared, the granularity is not

The two caches must describe one logical token boundary, but they do not need
the same physical granularity:

```text
logical boundary = tokens that completed forward execution
full attention   = paged blocks, including a valid partial tail block
GDN             = one live recurrent slab at that boundary
```

The first implementation should use the request boundary as the synchronization
point. This is the agent-specific choice: keep the newest GDN state so the next
turn never replays the previous assistant output, while retaining the exact
full-attention tail needed to continue correctly.

## 3. IE12X HOT Semantics

### 3.1 Saved token boundary

The saved token chain is:

```text
forwarded_tokens = input_tokens + output_tokens[:-1]
```

The last sampled output token is excluded because sampling happens after the
forward that produced its logits. It has not yet contributed to attention KV or
the recurrent state. The next request must submit that token as its first
forwarded token, followed by the new user/tool tail.

This off-by-one rule is a correctness invariant, not an implementation detail.

### 3.2 Checkpoint contents

Represent the checkpoint as one owned object, conceptually:

```text
HotContinuationCheckpoint {
    owner_request_id
    forwarded_tokens (or a compact immutable token digest plus exact token span)
    forwarded_token_count
    model/runtime compatibility fingerprint
    full_attention block-table ownership
    full_attention valid length of the tail block
    GDN state tensors and recurrent slot ownership
    any required SWA/auxiliary state
    creation/completion telemetry
}
```

The exact token sequence must remain available for an exact-prefix check. A
hash or client-provided session identifier is only a quick rejection filter;
it cannot be the final correctness check.

The fingerprint must reject changes to model revision, tokenizer/template
behavior, LoRA adapter, KV dtype/layout, recurrent layout, and any execution
option that changes the interpretation of cached state.

### 3.3 Save ordering

The save path must run only after all asynchronous decode work that writes the
state has completed:

1. drain or synchronize the decode pipeline;
2. determine `input_tokens + output_tokens[:-1]`;
3. commit the accepted GDN boundary and any hybrid state fixups;
4. retain/pin the full-attention blocks, including the partial tail;
5. transfer recurrent-slot ownership to the checkpoint;
6. publish the checkpoint atomically;
7. only then release unrelated request resources.

Never call the generic `finish_request` path first if it marks the request
finished and causes recurrent commit to be skipped. That ordering can save a
checkpoint whose full-attention token boundary is correct while its GDN state
is still advanced through a rejected speculative chunk.

On speculative decoding, stop/max-token/EOS handling must first select the
accepted row and repair the recurrent state, then create HOT. If that ordering
cannot be guaranteed, disable HOT for the speculative path until it can.

### 3.4 Claim ordering

When a new request arrives:

1. check the compatibility fingerprint;
2. tokenize/render the new tail;
3. verify that the request's reconstructed token chain starts with the saved
   `forwarded_tokens`;
4. atomically transfer full-attention block ownership and GDN slot ownership;
5. set the new request's computed/forwarded count to the checkpoint boundary;
6. submit only the missing tokens to the prefill/decode scheduler;
7. clear the checkpoint so two live requests cannot own the same state.

A failed check must invalidate and release the checkpoint exactly once, then
fall back to vLLM's ordinary prefix-cache/prefill path. Never partially claim
one cache family and fall back for the other.

## 4. Token-Stable Continuation API

The API should carry a server-issued continuation handle. In IE12X the handle
is returned as the response `id` and is supplied on the next request as
`continuation_handle`.

The handle is a candidate selector, not proof of identity. The server still
checks the exact token boundary and compatibility fingerprint. A single-entry
handle is sufficient for the first port:

- first turn: normal prompt submission;
- response: return a handle for the saved checkpoint;
- next turn: send only the new user/tool tail plus the handle;
- server: render the tail, prepend the saved token chain internally, and claim
  the live GPU state;
- any mismatch: HTTP/API error or explicit fallback according to the chosen
  contract, with no stale state retained.

Do not require the client to reserialize prior assistant thinking/content. The
model's generated reasoning tokens are already part of the saved token chain.
This is especially important for Qwen reasoning mode, tool-call parser output,
and templates whose whitespace or role markers can change between turns.

The first port should support one completion and one successor. Branching,
parallel continuations, and multiple handles can be added only after ownership
and output-equivalence tests are green.

## 5. Mapping to vLLM

Use existing vLLM cache abstractions wherever possible. The relevant concepts
in this tree are:

- `KVCacheCoordinator` for coordinating cache groups;
- `SingleTypeKVCacheManager` and `BlockPool` for physical block ownership;
- `FullAttentionSpec` for full-attention pages;
- `MambaSpec` for recurrent-state pages;
- `vllm/v1/request.py` request state and computed-token accounting;
- `vllm/v1/worker/mamba_utils.py` for Mamba state copy/restore semantics;
- the scheduler's request lifecycle and finish/abort paths.

Do not replace the coordinator with a second allocator. Add the smallest
ownership operation needed to pin/transfer one request's already allocated
group state. A useful shape is a checkpoint object holding the per-group block
references plus the Mamba state slot; the coordinator remains responsible for
allocation and release.

The vLLM hybrid manager currently has a deliberate distinction between
`mamba_cache_mode="all"` and `mamba_cache_mode="align"`:

- `all` makes Mamba checkpoints align to chunk/block boundaries for generic
  prefix caching;
- `align` uses the attention block size as the Mamba block size.

That policy is appropriate for a shared prefix cache, but it is not the target
HOT policy. For the single-session Qwen path, retain the newest live GDN state
at the completed request boundary. Do not force GDN into `all` or `align` just
to satisfy the common scheduler block size. Full attention may still use the
normal attention block size and its partial tail.

If an existing vLLM code path assumes every hybrid cache group has the same
hit granularity, introduce a HOT-only fast path at admission/finish rather than
changing global prefix-cache semantics. The HOT path should be disabled when
the model/backend cannot provide an atomic state transfer.

## 6. Prefill/Decode Split

The intended deployment has vLLM perform prefill, while the optimized runtime
handles continuation-aware scheduling and decode. The split must preserve one
ownership protocol:

```text
client -> continuation-aware gateway
       -> vLLM prefill (cold or fallback prefix path)
       -> shared/full-attention + GDN state handoff
       -> IE12X-style decode/continuation worker
       -> handle + generated tokens
```

For a HOT hit, do not send the entire history back through vLLM prefill. Pass
the claimed state and only the missing token tail. For a miss, use the normal
vLLM prefill path and create a new checkpoint only after the first request
completes safely.

If the split uses a KV connector or IPC transport, the transfer protocol must
include both cache families. A full-attention-only transfer is not a valid
Qwen3.6 continuation. The GDN state is too large to copy on every token, so
prefer same-device ownership transfer; cross-process transfer is a later
optimization with an explicit bandwidth measurement.

## 7. Lifecycle and Failure Rules

The checkpoint is single-owner. Its state must be released on:

- token-prefix mismatch;
- model, LoRA, tokenizer/template, or cache-layout mismatch;
- cancellation, timeout, or generation error;
- CUDA error, graph invalidation, engine reset, or shutdown;
- checkpoint replacement by a newer completed request;
- failed block-table or recurrent-slot transfer;
- unsupported speculative, SWA, or backend configuration.

Release must be idempotent and must cover every group. Do not release the
recurrent slot first and then try to save HOT. Do not publish a checkpoint
while an in-flight GPU operation can still mutate its buffers.

For the first implementation, set `max_num_seqs=1` or an equivalent HOT
eligibility gate. A single-entry checkpoint cannot safely represent completion
order from multiple active requests. Leave ordinary vLLM batching and prefix
caching unchanged for ineligible traffic.

## 8. Implementation Plan

### Phase A: trace existing ownership

Before editing, trace these paths end to end in the vLLM tree:

1. request admission and prefix-cache hit;
2. prefill completion and computed-token updates;
3. decode step and Mamba state update;
4. normal finish, abort, timeout, and engine reset;
5. block eviction and Mamba state copy/restore.

Write down the exact owner of every full-attention block and every Mamba state
slot at each point. Do not implement a new owner field until this map is clear.

### Phase B: minimal checkpoint object

Add one internal checkpoint type and one release function. It should own or pin:

- all non-cross-attention group blocks needed at the boundary;
- the partial full-attention tail length;
- the GDN/Mamba state slot and tensors;
- the immutable token boundary and compatibility fingerprint.

Keep metrics minimal: saves, claims, misses by reason, transfer failures, and
resident token/state bytes.

### Phase C: finish and claim hooks

Hook save after successful completion and recurrent commit. Hook claim before
ordinary prefix admission allocates replacement state. Make claim atomic from
the scheduler's point of view. On any exception, release the candidate and
enter the existing fallback path.

### Phase D: API adapter

Add a continuation handle to the API layer without changing the base request
model for ordinary users. The adapter should preserve exact generated token
ids, including hidden reasoning tokens and tool-call delimiters. Keep message
history semantics in the API layer; keep GPU state ownership in the engine.

### Phase E: vLLM prefill handoff

Implement a narrow handoff for the cold/miss path first. Then add the HOT hit
path with a feature flag. Do not combine this work with a global rewrite of
hybrid block alignment or prefix-cache retention.

## 9. Correctness Tests

The cheapest useful oracle is output equivalence with HOT disabled. For the
same seed, model, sampling parameters, and generated token limit:

1. run a cold first turn and a cold full-history second turn;
2. run the same first turn with HOT enabled and a tail-only second turn;
3. compare generated token ids, not only decoded text;
4. repeat with thinking enabled and with a tool-call response;
5. repeat with a partial full-attention tail block;
6. force EOS/max-token in the middle of a speculative chunk;
7. force a handle mismatch, cancellation, reset, and transfer failure;
8. verify the next request falls back cleanly and produces the cold result.

The speculative-stop test is mandatory for hybrid models. The accepted
recurrent boundary must be committed before HOT is saved; otherwise the
checkpoint can contain future/rejected GDN state even when the full-attention
KV length looks correct.

Add tests at the lowest level that catches the bug: pure ownership tests for
the coordinator, scheduler lifecycle tests for save/claim/release, and a small
real-model integration test for token equivalence. Do not use a fixed text
golden for nondeterministic sampling.

## 10. Performance Evaluation

Measure all three arms in one process/session whenever possible:

1. cold: no prefix cache and no HOT;
2. warm prefix: ordinary vLLM prefix caching and hybrid replay;
3. HOT: latest-state transfer with a tail-only successor request.

Use the same model, GPU clocks, tokenizer, sampling settings, and turn script.
Record per turn:

- prompt and completion token counts;
- full prompt length and newly submitted tail length;
- cached full-attention tokens;
- GDN state bytes and transfer time;
- HOT save/claim/miss reason;
- TTFT, prefill wall time, decode time, and end-to-end latency;
- peak GPU memory.

Run at least these histories and tails:

| History | New tail | Purpose |
| ---: | ---: | --- |
| 8K | 32/128 tokens | small agent turn |
| 32K | 128/512 tokens | normal long context |
| 64K | 128/512 tokens | recurrent replay pressure |
| 128K | 128/512 tokens | maximum retained context |

Also vary the previous assistant output length. The expected HOT advantage
grows when the previous output is long, because ordinary hybrid prefix reuse
may need to replay generated tokens to reconstruct GDN state. Decode tokens/s
should not be expected to improve materially; TTFT and prefill are the primary
signals.

For a realistic multi-turn quality check, replay one stratified dialogue from
each MT-Bench-101 task first, then expand. Never insert the dataset's reference
assistant answer into the model context. Judge the model's own transcript in
both arms with identical settings; cache metrics and quality scores are
separate measurements.

## 11. What Not to Port

Do not copy these IE12X details into vLLM blindly:

- IE12X's single-entry server map and response-id handle format;
- IE12X-specific request structs or CUDA slot IDs;
- global disabling of vLLM prefix caching;
- a universal Mamba `all`/`align` rewrite;
- token hashing without exact token verification;
- a checkpoint saved before speculative recurrent commit;
- a cross-process GDN copy before measuring bandwidth and synchronization;
- a multi-session LRU before single-session output equivalence is proven.

The portable idea is the ownership transfer and boundary invariant, not the
surrounding server plumbing.

## 12. Definition of Done

The migration is ready for broader use only when all of the following hold:

- HOT is default-off and has an explicit eligibility gate;
- cold, warm-prefix, and HOT outputs are token-equivalent on the target model;
- thinking tokens and tool-call tokens survive continuation;
- partial full-attention tails continue without replay or duplication;
- GDN state is transferred as one coherent latest boundary;
- speculative stop cannot save a future/rejected recurrent state;
- every mismatch, reset, cancellation, and transfer error releases cleanly;
- a HOT miss produces the same output as the cold/prefix fallback;
- measured TTFT/prefill improvement is reported from same-session A/B runs;
- no global vLLM hybrid-cache behavior changed without an independent benchmark.

The first useful milestone is therefore small: one Qwen3.6 process, one active
request, one latest checkpoint, one exact continuation handle, and a passing
token-equivalence test. Everything else is a follow-up.
