# Configuration

## Command line

```
dsm -m MODEL -p PROMPT [-n N] [--backend metal|cpu] [--threads N] [--temp T]
    [--chat] [--dump-logits FILE] [--tokens-only]
dsm -m MODEL --render-chat REQUEST.json
dsm serve -m MODEL [--backend metal|cpu] [--host 127.0.0.1] [--port 8080]
    [--ctx N] [--threads N] [--alias NAME]
```

| Option | Default | Meaning |
|---|---|---|
| `-n N` | 16 | tokens to generate |
| `--backend` | `metal` when available | `cpu` runs the scalar reference path |
| `--threads N` | 8 | CPU threads (CPU backend only) |
| `--temp T` | 0 | sampling temperature; 0 is greedy |
| `--chat` | off | wrap the prompt in the model's chat template |
| `--dump-logits FILE` | off | write the logits of every position |
| `--tokens-only` | off | print the token ids of the prompt and exit |
| `--render-chat FILE` | | print the chat template applied to an OpenAI-style request |
| `--ctx N` (serve) | min(model context, 32768) | context size of the server session |
| `--alias NAME` (serve) | GGUF file name without `.gguf` | model id reported by `/v1/models` |

The server handles one request at a time and exposes `GET /v1/models` and
`POST /v1/chat/completions`. Request fields read: `messages`, `max_tokens` (or
`max_completion_tokens`), `temperature`, `top_p`, `stop`, `stream` and
`tools`.

## Environment variables

Unset means the default. The kill switches exist to compare a fast path with
the slower path it replaced; leave them unset for normal use.

| Variable | Default | Effect | Kind |
|---|---|---|---|
| `DSM_METAL_DIR` | `<dsm dir>/metal` | directory that contains `dsm.metal` (compiled at startup) | setting |
| `DSM_PREFILL_CHUNK` | 1024 (512 for the dense 27B) | prompt rows per prefill chunk, capped to the context | setting |
| `DSM_CKPT` | 3 | recurrent-state checkpoints kept for prompt reuse on Qwen3.6/Qwen3.8 (2-4) | setting |
| `DSM_MM_F32` | 1 on Qwen3.6/Qwen3.8, 0 otherwise | stage `mul_mm` tiles in float (1) or half (0) | precision |
| `DSM_PREFILL=0` | on | prefill one token at a time | kill switch |
| `DSM_HYBRID_PREFILL=0` | on | prefill Qwen3.6/Qwen3.8 one token at a time | kill switch |
| `DSM_MUL_MM_ID=0` | on | per-token expert GEMVs instead of `mul_mm_id` in prefill | kill switch |
| `DSM_MM_F32W=0` | on | per-row GEMV instead of `mul_mm` for F32 router/alpha/beta weights | kill switch |
| `DSM_ATTN_PREFILL=0` | on | older prefill attention kernels (no key tiling) | kill switch |
| `DSM_ATTN_FA=0` | on | tiled dim-128 attention instead of flash attention in Qwen3 prefill | kill switch |
| `DSM_ATTN_DECODE=0` | on | single-pass decode attention instead of split-K | kill switch |
| `DSM_DECODE_FUSE=0` | on | separate norm, RoPE and residual dispatches in decode | kill switch |
| `DSM_PROFILE=1` | off | per-stage timings of the token path, printed at exit (forces the token loop) | diagnostics |
| `DSM_STAGE_PROF` (any value) | off | per-stage GPU timings of Qwen3.6/Qwen3.8 batched prefill in `bench-prefill` | diagnostics |
| `DSM_CKPT_PROFILE` (any value) | off | print size and copy time of each checkpoint | diagnostics |
| `DSM_DUMP_DIR=DIR` | off | write per-layer hidden states of Qwen3.6/Qwen3.8 decode as `l_out-<layer>-<pos>.f32` (forces the token loop) | diagnostics |
| `DSM_MAX_BUFFER_BYTES=N` | device limit | map the weights in Metal views of at most N bytes, so tensors cross view boundaries (`make check-views`) | test |

`llama-ref` (a verification tool, not part of `dsm`) reads `REF_BATCH`
(default 512; 1 runs llama.cpp token by token).
