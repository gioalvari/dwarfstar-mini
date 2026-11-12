# dwarfstar-mini

## What it is

dwarfstar-mini is a small C and Objective-C/Metal GGUF inference engine for
Apple Silicon. It is for people who want a readable local engine and a direct
comparison point for llama.cpp. It exists to show a small engine can match
llama.cpp speed on measured Qwen3 and Qwen3.6 decode workloads on a Mac.

## Highlights

- Qwen3.6-35B-A3B UD-Q4_K_S measured 55.4 tokens/s decode.
- Qwen3.8-27B Q4_K_M measured 11.3 tokens/s decode.
- Qwen3-1.7B Q4_K_M measured 1670 tokens/s at 570-token prefill.
- The server reuses chat prefixes with Metal state checkpoints.

## Requirements

- An Apple Silicon Mac with Metal.
- macOS with the Metal APIs used by your Xcode command line tools. macOS 15 adds
  an optional `MTLResidencySet` optimization; it is not required.
- Xcode command line tools: `xcode-select --install`.

## Quickstart

```sh
git clone https://github.com/gioalvari/dwarfstar-mini.git
cd dwarfstar-mini
make
mkdir -p models
```

Download the small Qwen3 model. Check the URL and its final content length
first, then download it.

```sh
URL='https://huggingface.co/Qwen/Qwen3-1.7B-GGUF/resolve/main/Qwen3-1.7B-Q8_0.gguf'
curl -sIL "$URL" | grep -i '^content-length\|^HTTP'
curl -L "$URL" -o models/Qwen3-1.7B-Q8_0.gguf
```

Other tested downloads:

```sh
URL='https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF/resolve/main/Qwen3.6-35B-A3B-UD-Q4_K_S.gguf'
curl -sIL "$URL" | grep -i '^content-length\|^HTTP'
URL='https://huggingface.co/ggml-org/Qwen3.8-27B-GGUF/resolve/main/Qwen3.8-27B-Q4_K_M.gguf'
curl -sIL "$URL" | grep -i '^content-length\|^HTTP'
```

Run a prompt:

```sh
./dsm -m models/Qwen3-1.7B-Q8_0.gguf -p 'The capital of France is' -n 32 --temp 0
```

Start the local server and send a chat request:

```sh
./dsm serve -m models/Qwen3-1.7B-Q8_0.gguf --port 8080
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Say hello."}],"max_tokens":16}'
```

## Supported models and weight types

| Architecture | Models tested | Notes |
|---|---|---|
| `qwen3` | Qwen3-1.7B Q2_K / Q4_K_M / Q5_K_M / Q8_0 | Dense |
| `qwen3moe` | Qwen3-30B-A3B, Qwen3-Coder-30B-A3B | Asymmetric Q3_K/Q4_K experts and Q8_0 tested |
| `qwen35moe` | Qwen3.6-35B-A3B UD-Q4_K_S / UD-Q4_K_M | Hybrid Gated DeltaNet and attention |
| `qwen35` | Qwen3.8-27B Q4_K_M | Hybrid dense |

Metal supports Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q8_0, and F32 weights.

## Speed

M4 Pro 48 GB, warm tokens/s:

| Model | DSM prefill | DSM decode | llama.cpp decode |
|---|---:|---:|---:|
| Qwen3.6-35B-A3B UD-Q4_K_S | 674-693 | 55.4 | 55.6 |
| Qwen3.8-27B Q4_K_M | 93-100 | 11.3 | 11.5 |
| Qwen3-1.7B Q4_K_M | 1670 | 136 | 149.5 |

See [benchmark methodology and the full table](docs/benchmarks.md).

## Model quality

Quantization changes output quality. The local perplexity, KLD, and published
model-quality tables are in [docs/quality.md](docs/quality.md).

## Limitations

- Apple Silicon only.
- Only the listed architectures and weight types are supported.
- The KV cache is F32.
- Vision and MTP tensors are not evaluated.
- The server handles one request at a time.
- The server accepts chat completions, not text completions or embeddings.
- Chat sampling supports `temperature` and `top_p`; it does not expose other
  sampling parameters.

## Docs

- [Architecture](docs/architecture.md)
- [Benchmarks](docs/benchmarks.md)
- [Model quality](docs/quality.md)
- [Configuration](docs/configuration.md)
- [Development history](docs/history/README.md)

## Development

Run unit tests with `make test`. Model-dependent verification targets are
`check-prefill`, `check-ref`, `check-rewind`, `check-metal-gemv`, `check-attn`,
`bench-prefill`, and `bench-gemv`.

## License and acknowledgements

MIT. Metal kernels are derived from llama.cpp/ggml under MIT; see [NOTICE](NOTICE).
The project is inspired by [antirez/ds4](https://github.com/antirez/ds4).
