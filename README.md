# dwarfstar-mini

Minimal C + Metal LLM inference engine for Apple Silicon, inspired by
[antirez/ds4](https://github.com/antirez/ds4). Target: M4 Pro 48 GB.

## Supported models

| Architecture | Models tested | Notes |
|---|---|---|
| `qwen3` (dense) | Qwen3-1.7B Q2_K / Q4_K_M / Q5_K_M / Q8_0 | |
| `qwen3moe` | Qwen3-30B-A3B, Qwen3-Coder-30B-A3B (asymmetric Q3_K/Q4_K experts, Q8_0) | OpenAI tool calls for the Coder chat template |
| `qwen35moe` | Qwen3.6-35B-A3B UD-Q4_K_S / UD-Q4_K_M | Gated DeltaNet + gated attention, 256 experts |
| `qwen35` (dense) | Qwen3.8-27B Q4_K_M | |

Supported weight types on Metal: Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q8_0, F32.
Decode, batched prefill (`mul_mm` / `mul_mm_id`), and the OpenAI-compatible
server (`./dsm serve`) run on Metal. For the recurrent qwen35 models, the
server reuses the prompt prefix through state checkpoints.

Speed on M4 Pro (tok/s, 570-token prompt, warm), compared with llama.cpp
`llama-bench -fa 1` (pp512 / tg128):

| Model | DSM prefill | llama prefill | DSM decode | llama decode |
|---|---|---|---|---|
| Qwen3.6-35B-A3B UD-Q4_K_S | 742 | 754 | 54.7 | 52.1 |
| Qwen3.6-35B-A3B UD-Q4_K_M | 715 | 628 | 47.1 | 44.3 |
| Qwen3.8-27B Q4_K_M | 93-100 | 125.5 | 11.3 | 11.5 |
| Qwen3-30B-A3B asym | 613 | 849 | ~63 | 66.8 |
| Qwen3-1.7B Q4_K_M | 1064 | 1882 | ~125 | 143 |

Details, correctness gates and history: `docs/F6-plan.md` (batched prefill)
and `docs/F7-plan.md` (qwen35 / qwen35moe).

## Build and run

```sh
make dsm
./dsm -m models/Qwen3-1.7B-Q4_K_M.gguf -p 'The capital of France is' -n 32 --temp 0
./dsm serve -m models/Qwen3.6-35B-A3B-UD-Q4_K_S.gguf --port 8080 --ctx 32768
make test
```

Models are not committed (`models/` is ignored). Validation tools:
`check-metal-gemv`, `check-prefill`, `check-rewind`, `bench-prefill`, and
`llama-ref` / `check-ref` for logit comparison against llama.cpp (built
against the Homebrew `llama.cpp`).

Metal GEMV, `mul_mm` and attention kernel ports include code from llama.cpp,
licensed under MIT (see `NOTICE`).
