# Architecture

## Model loading and input

`dsm` memory-maps a GGUF v3 file. Tensor weights stay in that mapping. The
loader accepts `qwen3`, `qwen3moe`, `qwen35`, and `qwen35moe` architectures.
`qwen3` and `qwen35` are dense. The `moe` variants route work to experts.
`qwen35` and `qwen35moe` combine Gated DeltaNet recurrent layers with gated
attention layers.

The tokenizer reads the token and merge data in the GGUF file and applies the
Qwen byte-BPE tokenizer. `--chat` renders a user turn in ChatML form. The
server renders chat messages and tool calls in the Qwen template. Qwen3.5
generation starts with its thinking marker and removes reasoning from earlier
assistant turns when it rebuilds a prompt.

## CPU reference

The CPU path is the reference forward implementation. It evaluates one token
at a time in FP32, dequantizes weights as needed, and keeps an F32 KV cache.
It is useful for comparisons and for running without Metal.

## Metal backend

The Metal backend wraps the mapped weights in Metal buffer views. A file that
is larger than the device maximum buffer length is covered by overlapping
views. The overlap lets every tensor fall inside one view without copying its
weights. The backend compiles the shaders in `metal/dsm.metal` at runtime.
Host-side Metal code is in `src/metal*.m`.

Decode evaluates one token at a time. It runs normalization, projections,
RoPE, KV writes, causal attention, feed-forward work, and output logits on the
GPU. The standard Qwen path uses grouped-query attention. MoE layers select
the routed experts before their feed-forward projections.

Prefill evaluates prompt tokens in batches. `mul_mm` handles dense matrix
products and `mul_mm_id` handles routed expert products. Long prompts are
split into chunks. The prefill attention path uses tiled and flash-attention
kernels where their supported head dimensions apply.

For Qwen3.5 models, recurrent layers carry Gated DeltaNet state and convolution
history alongside the attention KV cache. The server stores checkpoints of that
state. On a shared chat prefix, it rewinds to the newest usable checkpoint and
prefills only the remaining tokens.

## HTTP server

`dsm serve` loads one model and exposes `GET /v1/models` plus
`POST /v1/chat/completions`. It accepts OpenAI-style messages, text content,
sampling fields, stop strings, tools, and optional SSE streaming. The server
handles one request at a time and reuses prompt prefixes between requests.

## Verification tools

Build these targets with `make TARGET` when their model or llama.cpp inputs are
available.

| Target | Purpose |
|---|---|
| `check-prefill` | Compares batched prefill logits and continuations with the token path. |
| `check-ref` | Compares DSM logits and greedy output with llama.cpp. |
| `check-rewind` | Checks rewind and checkpoint reuse against a fresh prefill. |
| `check-metal-gemv` | Compares Metal GEMV and matrix kernels with CPU results. |
| `check-attn` | Checks the flash-attention kernel against tiled attention. |
| `bench-prefill` | Measures prompt-prefill throughput. |
| `bench-gemv` | Measures GEMV kernel throughput. |
