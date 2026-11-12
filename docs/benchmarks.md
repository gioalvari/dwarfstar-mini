# Benchmarks

All results below are tokens per second on an M4 Pro with 48 GB. DSM prefill
uses the best warm `bench-prefill` repetition. Decode uses `./dsm -n 128`.
llama.cpp is build 9860 and uses `llama-bench -fa 1`. Prefill prompts contain
570 and 3990 tokens; llama.cpp uses 512 and 4096 token cases.

## Latest results

| Model | DSM pp570 | llama pp512 | DSM pp3990 | llama pp4096 | DSM decode | llama tg128 |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3.6-35B-A3B UD-Q4_K_S | 674-693 | 719 | 522 | 351 | 55.4 | 55.6 |
| Qwen3.6-35B-A3B UD-Q4_K_M | 614 | 778 | 541 | 367 | 49.9 | 54.3 |
| Qwen3.8-27B Q4_K_M | 93-100 | 125.5 | - | - | 11.3 | 11.5 |
| Qwen3-30B-A3B asym | 680 | 828 | 355 | ~473 | 64.7 | 63.2 |
| Qwen3-1.7B Q4_K_M | 1670 | 2048 | 1222 | 1735 | 136 | 149.5 |
| Qwen3-1.7B Q5_K_M | 1568 | 1909 | 1176 | 1641 | 113.8 | 125.2 |
| Qwen3-1.7B Q8_0 | 1784 | 2126 | 1275 | 1803 | 95.7 | 103.0 |

Run-to-run variation was about ±5-10% for the same configuration. The
Qwen3.6 short-prefill result ranged from 624 to 693 tokens/s.

## Progress history

- [F2](history/F2.md) added the scalar CPU reference.
- [F3](history/F3.md) added Metal decode and mapped weight buffers.
- [F4](history/F4.md) added K-quantized kernels.
- [F4b](history/F4b.md) added Qwen3 MoE support.
- [F5](history/F5.md) added the OpenAI-compatible server.
- [F6](history/F6-plan.md) added batched prefill and `mul_mm` paths.
- [F7](history/F7-plan.md) added the hybrid Qwen3.6 and Qwen3.8 models and checkpoint reuse.
- [F8](history/F8.md) added flash attention and the current Qwen3 results.

The full development-note index is in [history](history/README.md).
