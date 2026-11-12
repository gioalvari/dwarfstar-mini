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

Speed on M4 Pro 48 GB (tok/s, warm, same session, 2026-11-12). DSM prefill
uses 570- and 3990-token prompts; llama.cpp build 9860 uses
`llama-bench -fa 1` pp512 / pp4096 / tg128:

| Model | DSM pp570 | llama pp512 | DSM pp3990 | llama pp4096 | DSM decode | llama tg128 |
|---|---|---|---|---|---|---|
| Qwen3.6-35B-A3B UD-Q4_K_S | 674-693 | 719 | 522 | 351 | 55.4 | 55.6 |
| Qwen3.6-35B-A3B UD-Q4_K_M | 614 | 778 | 541 | 367 | 49.9 | 54.3 |
| Qwen3.8-27B Q4_K_M | 93-100 | 125.5 | - | - | 11.3 | 11.5 |
| Qwen3-30B-A3B asym | 680 | 828 | 355 | ~473 | 64.7 | 63.2 |
| Qwen3-1.7B Q4_K_M | 1670 | 2048 | 1222 | 1735 | 136 | 149.5 |
| Qwen3-1.7B Q5_K_M | 1568 | 1909 | 1176 | 1641 | 113.8 | 125.2 |
| Qwen3-1.7B Q8_0 | 1784 | 2126 | 1275 | 1803 | 95.7 | 103.0 |

The 27B row is from 2026-11-09 (`docs/F7-plan.md`, Result 5).

Details, correctness gates and history: `docs/F6-plan.md` (batched prefill),
`docs/F7-plan.md` (qwen35 / qwen35moe) and `docs/F8.md` (qwen3 prefill and
decode against llama.cpp).

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
