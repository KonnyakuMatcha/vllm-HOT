<!-- markdownlint-disable MD001 MD041 -->
<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/vllm-project/vllm/main/docs/assets/logos/vllm-logo-text-dark.png">
    <img alt="vLLM" src="https://raw.githubusercontent.com/vllm-project/vllm/main/docs/assets/logos/vllm-logo-text-light.png" width=55%>
  </picture>
</p>

<h3 align="center">
Easy, fast, and cheap LLM serving for everyone
</h3>

<p align="center">
| <a href="https://docs.vllm.ai"><b>Documentation</b></a> | <a href="https://blog.vllm.ai/"><b>Blog</b></a> | <a href="https://arxiv.org/abs/2309.06180"><b>Paper</b></a> | <a href="https://x.com/vllm_project"><b>Twitter/X</b></a> | <a href="https://discuss.vllm.ai"><b>User Forum</b></a> | <a href="https://slack.vllm.ai"><b>Developer Slack</b></a> |
</p>

🔥 We have built a vLLM website to help you get started with vLLM. Please visit [vllm.ai](https://vllm.ai) to learn more.
For events, please visit [vllm.ai/events](https://vllm.ai/events) to join us.

---

## About

vLLM is a fast and easy-to-use library for LLM inference and serving.

Originally developed in the [Sky Computing Lab](https://sky.cs.berkeley.edu) at UC Berkeley, vLLM has grown into one of the most active open-source AI projects built and maintained by a diverse community of many dozens of academic institutions and companies from over 2000 contributors.

vLLM is fast with:

- State-of-the-art serving throughput
- Efficient management of attention key and value memory with [**PagedAttention**](https://blog.vllm.ai/2023/06/20/vllm.html)
- Continuous batching of incoming requests, chunked prefill, prefix caching
- Fast and flexible model execution with piecewise and full CUDA/HIP graphs
- Quantization: FP8, MXFP8/MXFP4, NVFP4, INT8, INT4, GPTQ/AWQ, GGUF, compressed-tensors, ModelOpt, TorchAO, and [more](https://docs.vllm.ai/en/latest/features/quantization/index.html)
- Optimized attention kernels including FlashAttention, FlashInfer, TRTLLM-GEN, FlashMLA, and Triton
- Optimized GEMM/MoE kernels for various precisions using CUTLASS, TRTLLM-GEN, CuTeDSL
- Speculative decoding including n-gram, suffix, EAGLE, DFlash
- Automatic kernel generation and graph-level transformations using torch.compile
- Disaggregated prefill, decode, and encode

vLLM is flexible and easy to use with:

- Seamless integration with popular Hugging Face models
- High-throughput serving with various decoding algorithms, including *parallel sampling*, *beam search*, and more
- Tensor, pipeline, data, expert, and context parallelism for distributed inference
- Streaming outputs
- Generation of structured outputs using xgrammar or guidance
- Tool calling and reasoning parsers
- OpenAI-compatible API server, plus Anthropic Messages API and gRPC support
- Efficient multi-LoRA support for dense and MoE layers
- Support for NVIDIA GPUs, AMD GPUs, Intel GPUs, and x86/ARM/PowerPC CPUs. Additionally, diverse hardware plugins such as Google TPUs, Intel Gaudi, IBM Spyre, Huawei Ascend, Rebellions NPU, Apple Silicon, MetaX GPU, and more.

vLLM seamlessly supports 200+ model architectures on Hugging Face, including:

- Decoder-only LLMs (e.g., Llama, Qwen, Gemma)
- Mixture-of-Expert LLMs (e.g., Mixtral, DeepSeek-V3, Qwen-MoE, GPT-OSS)
- Hybrid attention and state-space models (e.g., Mamba, Qwen3.5)
- Multi-modal models (e.g., LLaVA, Qwen-VL, Pixtral)
- Embedding and retrieval models (e.g., E5-Mistral, GTE, ColBERT)
- Reward and classification models (e.g., Qwen-Math)

Find the full list of supported models [here](https://docs.vllm.ai/en/latest/models/supported_models.html).

## Getting Started

Install vLLM with [`uv`](https://docs.astral.sh/uv/) (recommended) or `pip`:

```bash
uv pip install vllm
```

Or [build from source](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/index.html#build-wheel-from-source) for development.

Visit our [documentation](https://docs.vllm.ai/en/latest/) to learn more.

- [Installation](https://docs.vllm.ai/en/latest/getting_started/installation.html)
- [Quickstart](https://docs.vllm.ai/en/latest/getting_started/quickstart.html)
- [List of Supported Models](https://docs.vllm.ai/en/latest/models/supported_models.html)

## Qwen HOT Multi-Turn Benchmark

This fork includes a specialized HOT continuation path for local Qwen small
models, and a benchmark that compares it with vLLM prefix caching. The complete
benchmark is in [`benchmarks/multi_turn/benchmark_hot_vs_prefix.py`](benchmarks/multi_turn/benchmark_hot_vs_prefix.py).

The measured model was `/ssd/nfs/models/Qwen/Qwen3.6-35B-A3B-NVFP4`. Both runs
used two GPUs, `--tensor-parallel-size 2`, `--max-num-seqs 1`,
`--max-model-len 96000`, and CUDA graphs (`--enforce-eager` was not used):

```bash
MODEL=/ssd/nfs/models/Qwen/Qwen3.6-35B-A3B-NVFP4

# HOT: continuation enabled, prefix caching disabled
CUDA_VISIBLE_DEVICES=0,1 VLLM_ENABLE_HOT_CONTINUATION=1 \
  .venv/bin/python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" --served-model-name qwen-nvfp4 \
  --tensor-parallel-size 2 --max-num-seqs 1 --max-model-len 96000 \
  --port 8000

.venv/bin/python benchmarks/multi_turn/benchmark_hot_vs_prefix.py \
  --mode hot --context-chars 96000 --output /tmp/qwen-hot.json

# Prefix: HOT disabled, automatic prefix caching enabled
CUDA_VISIBLE_DEVICES=0,1 VLLM_ENABLE_HOT_CONTINUATION=0 \
  .venv/bin/python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" --served-model-name qwen-nvfp4 \
  --tensor-parallel-size 2 --max-num-seqs 1 --max-model-len 96000 \
  --enable-prefix-caching --port 8000

.venv/bin/python benchmarks/multi_turn/benchmark_hot_vs_prefix.py \
  --mode prefix --context-chars 96000 --output /tmp/qwen-prefix.json
```

Use `--mode cold` with both optimizations disabled for the no-cache baseline.
Run every mode in a fresh server process with identical model, GPU, context,
turn count, and generation settings.

### Test Method and TTFT Definition

The workload is one four-turn conversation. The first turn establishes the
context; subsequent turns append a short user tail. The persistent context is
96,000 characters, about 18,526 prompt tokens on turn 1, and each turn
generates four tokens. Requests use the streaming Chat API with fixed
`seed=0`, `temperature=0`, and `return_token_ids=true`.

The benchmark starts its timer immediately before sending the HTTP request.
TTFT is recorded when the first non-empty `delta.content` or
`delta.reasoning_content` arrives. Empty chunks and role-only chunks are
ignored. Latency ends when the SSE stream receives `[DONE]`. The steady-state
comparison uses turns 2-4 because turn 1 includes initial request and runtime
warmup effects.

### Recorded Results

Results below are milliseconds; `mean` and `p50` cover turns 2-4:

| Mode | TTFT mean | TTFT p50 | Latency mean | Latency p50 |
| --- | ---: | ---: | ---: | ---: |
| Cold | 1,226.5 | 1,227.2 | 1,232.5 | 1,233.1 |
| Prefix cache | 168.1 | 166.8 | 169.3 | 168.2 |
| HOT | 125.8 | 124.3 | 140.1 | 138.5 |

First-turn TTFT was 1,917.0 ms for cold, 2,638.2 ms for prefix cache, and
1,905.5 ms for HOT. These values are informational only and are not used to
measure cache reuse. The prefix-cache server reported a 72.5% hit rate.

For this single-session workload, HOT reduced steady-state TTFT by about 25%
relative to prefix caching and about 90% relative to cold execution. HOT does
not change model weights or the sampling policy.

### HOT Compared with Prefix Cache

HOT is a sequential continuation fast path. At the end of a turn it saves one
resident checkpoint containing the exact executed-token boundary, full-
attention KV ownership, and the latest GDN/Mamba recurrent state. The server
returns a continuation handle. On the next turn, the client sends only the new
tail; the scheduler validates the handle and exact boundary, transfers state
ownership, and starts from the saved computed-token count. A mismatch releases
the checkpoint and falls back to the normal vLLM path.

Prefix caching is a shared hash/token-block lookup. The client sends the full
conversation, matching prefix blocks are reused, and the unmatched suffix is
prefilled. It supports shared prefixes, branching requests, and multi-tenant
workloads without requiring requests to arrive in conversation order. For
hybrid models, its recurrent state follows the generic cache granularity.

| Property | Prefix cache | HOT |
| --- | --- | --- |
| Admission | Hash and look up reusable prefix blocks | Claim one exact continuation handle |
| Client request | Full conversation history | New tail after the first turn |
| State reuse | Shared cached blocks | Direct ownership transfer of live state |
| GDN/Mamba state | Generic aligned cache policy | Latest recurrent state at the boundary |
| Workload | Shared, branching, multi-tenant | One sequential active session |
| Failure behavior | Cache miss and normal prefill | Invalidate checkpoint and fall back |

HOT is not a replacement for prefix caching. It is specialized for a single
active sequential session; prefix caching remains the general-purpose sharing
mechanism and fallback path. Exact output-token equality is not used as an
intelligence criterion because repeated Qwen NVFP4 runs showed CUDA/streaming
sampling nondeterminism even with fixed seed and temperature. A semantic probe
did preserve the remembered project value `ORBIT` and number `7319` through
HOT continuation.

## Contributing

We welcome and value any contributions and collaborations.
Please check out [Contributing to vLLM](https://docs.vllm.ai/en/latest/contributing/index.html) for how to get involved.

## Citation

If you use vLLM for your research, please cite our [paper](https://arxiv.org/abs/2309.06180):

```bibtex
@inproceedings{kwon2023efficient,
  title={Efficient Memory Management for Large Language Model Serving with PagedAttention},
  author={Woosuk Kwon and Zhuohan Li and Siyuan Zhuang and Ying Sheng and Lianmin Zheng and Cody Hao Yu and Joseph E. Gonzalez and Hao Zhang and Ion Stoica},
  booktitle={Proceedings of the ACM SIGOPS 29th Symposium on Operating Systems Principles},
  year={2023}
}
```

## Contact Us

<!-- --8<-- [start:contact-us] -->
- For technical questions and feature requests, please use GitHub [Issues](https://github.com/vllm-project/vllm/issues)
- For discussing with fellow users, please use the [vLLM Forum](https://discuss.vllm.ai)
- For coordinating contributions and development, please use [Slack](https://slack.vllm.ai)
- For security disclosures, please use GitHub's [Security Advisories](https://github.com/vllm-project/vllm/security/advisories) feature
- For collaborations and partnerships, please contact us at [collaboration@vllm.ai](mailto:collaboration@vllm.ai)
<!-- --8<-- [end:contact-us] -->

## Media Kit

- If you wish to use vLLM's logo, please refer to [our media kit repo](https://github.com/vllm-project/media-kit)
