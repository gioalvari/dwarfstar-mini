# dwarfstar-mini

dwarfstar-mini is a small LLM inference engine for Apple Silicon, written in C
and Objective-C with Metal shaders. It loads GGUF files and runs Qwen3,
Qwen3-MoE and the hybrid Qwen3.6 / Qwen3.8 models (Gated DeltaNet plus
attention) on the GPU. It also includes an OpenAI-compatible chat server.

The code is meant to be read: plain C and Objective-C, one Metal shader file,
no dependencies beyond the system frameworks. Every optimization is checked
against a reference path and against llama.cpp. On the tested models, speed is close to
llama.cpp, and faster on some workloads.

## Highlights

Measured on an M4 Pro with 48 GB, against llama.cpp build 9860 on the same
GGUF files:

- Qwen3.6-35B-A3B UD-Q4_K_S: 55.4 tokens/s decode (llama.cpp 55.6), and
  522 tokens/s prefill on a 3,990-token prompt (llama.cpp 351 at 4,096).
- Qwen3.8-27B Q4_K_M, the strongest model tested: 11.3 tokens/s decode
  (llama.cpp 11.5).
- Qwen3-1.7B Q4_K_M: 1,670 tokens/s prefill on a 570-token prompt, 136 tokens/s
  decode.
- The server reuses the shared prefix of a chat across requests, including the
  recurrent state of the hybrid models, so each turn only processes new tokens.

## Requirements

- An Apple Silicon Mac. It was developed on an M4 Pro (48 GB, macOS 26). CI
  builds it on macOS 15.
- Xcode command line tools: `xcode-select --install`.
- Enough memory for the model: about 21 GB for Qwen3.6-35B-A3B UD-Q4_K_S and
  19 GB for Qwen3.8-27B Q4_K_M. The weights are memory-mapped, not copied.

## Quickstart

```sh
git clone https://github.com/gioalvari/dwarfstar-mini.git
cd dwarfstar-mini
make
mkdir -p models
curl -L -o models/Qwen3-1.7B-Q8_0.gguf \
  https://huggingface.co/Qwen/Qwen3-1.7B-GGUF/resolve/main/Qwen3-1.7B-Q8_0.gguf
```

Run a prompt:

```sh
./dsm -m models/Qwen3-1.7B-Q8_0.gguf -p 'The capital of France is' -n 32 --temp 0
```

Start the server and send a chat request:

```sh
./dsm serve -m models/Qwen3-1.7B-Q8_0.gguf --port 8080
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Say hello."}],"max_tokens":16}'
```

Larger models that were tested (sizes are the exact file sizes):

| Model | Size | URL |
|---|---:|---|
| Qwen3.6-35B-A3B UD-Q4_K_S | 20.9 GB | `https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF/resolve/main/Qwen3.6-35B-A3B-UD-Q4_K_S.gguf` |
| Qwen3.8-27B Q4_K_M | 19.4 GB | `https://huggingface.co/ggml-org/Qwen3.8-27B-GGUF/resolve/main/Qwen3.8-27B-Q4_K_M.gguf` |

[examples/opencode.json](examples/opencode.json) shows how to point an
OpenAI-compatible client at the server.

## Supported models and weight types

| Architecture | Models tested | Notes |
|---|---|---|
| `qwen3` | Qwen3-1.7B Q2_K / Q4_K_M / Q5_K_M / Q8_0 | Dense |
| `qwen3moe` | Qwen3-30B-A3B, Qwen3-Coder-30B-A3B | Mixed Q3_K/Q4_K experts and Q8_0 |
| `qwen35moe` | Qwen3.6-35B-A3B UD-Q4_K_S / UD-Q4_K_M | Hybrid Gated DeltaNet + attention, MoE |
| `qwen35` | Qwen3.8-27B Q4_K_M | Hybrid Gated DeltaNet + attention, dense |

Weight types on Metal: Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q8_0 and F32.

## Speed

Tokens/s on an M4 Pro 48 GB, warm runs. Prefill uses a 570-token prompt
(llama.cpp: 512 tokens).

| Model | prefill | llama.cpp prefill | decode | llama.cpp decode |
|---|---:|---:|---:|---:|
| Qwen3.6-35B-A3B UD-Q4_K_S | 674-693 | 719 | 55.4 | 55.6 |
| Qwen3.8-27B Q4_K_M | 93-100 | 125.5 | 11.3 | 11.5 |
| Qwen3-30B-A3B | 680 | 828 | 64.7 | 63.2 |
| Qwen3-1.7B Q4_K_M | 1670 | 2048 | 136 | 149.5 |

Long prompts, the method and all models are in
[docs/benchmarks.md](docs/benchmarks.md).

## Model quality

Lower-bit quantizations lose some quality. On Qwen3-1.7B, Q8_0 is close to
BF16 (perplexity +0.6%), Q4_K_M costs +8.6% and Q2_K is not usable.
[docs/quality.md](docs/quality.md) has the local measurements and the
published benchmark scores of each model.

## Limitations

- Apple Silicon and Metal only. The CPU path is a slow reference.
- Only the architectures and weight types listed above.
- The KV cache is F32, so long contexts use more memory than in llama.cpp.
- No vision input and no multi-token prediction.
- The server runs one request at a time and only implements
  `GET /v1/models` and `POST /v1/chat/completions` (with streaming and tool
  calls). Sampling supports temperature, top-p/top-k/min-p, deterministic
  seeds, and OpenAI presence/frequency penalties.

## Docs

- [Architecture](docs/architecture.md): how the engine works.
- [Benchmarks](docs/benchmarks.md): method and full results.
- [Model quality](docs/quality.md): quantization loss and model scores.
- [Configuration](docs/configuration.md): CLI flags and `DSM_*` variables.
- [Development history](docs/history/README.md): notes from each phase.

## Development

`make test` runs the unit tests. With models in `models/`, these targets check
the GPU paths: `check-prefill` (batched prefill against the token-by-token
path), `check-ref` (logits against llama.cpp), `check-rewind`,
`check-metal-gemv`, `check-attn`, `check-views` (weights split across several
Metal buffers), and the benchmarks `bench-prefill` and `bench-gemv`.

## License and acknowledgements

MIT, see [LICENSE](LICENSE). Several Metal kernels are derived from
llama.cpp/ggml (MIT), see [NOTICE](NOTICE). The project started from ideas in
[antirez/ds4](https://github.com/antirez/ds4).
