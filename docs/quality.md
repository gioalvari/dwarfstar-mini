# Model quality and quantization loss

Speed is in the README. This page covers what quantization costs and how the
supported models compare. Numbers measured on this machine are marked
**local**. Everything else is published and cited at the end.

## Local: quantization loss (2026-11-12)

Setup: llama.cpp build 9860 `llama-perplexity` on WikiText-2 raw (test split),
context 512, 32 chunks, M4 Pro 48 GB.
- **KLD** compares each quantized model's token distribution with the BF16
  logits of the same model.
- **Same top-1** is the share of positions where the quantized model picks the
  same most likely token as BF16.
- **RMS Δp** is the root mean square of the change in the probability of the
  correct token.

### Qwen3-1.7B against BF16

| Weights | PPL | ΔPPL | KLD mean / median / p99 | Same top-1 | RMS Δp |
|---|---|---|---|---|---|
| BF16 | 14.533 | - | - | - | - |
| Q8_0 | 14.623 | +0.6% | 0.0025 / 0.0008 / 0.027 | 98.0% | 1.6% |
| Q5_K_M | 14.680 | +1.0% | 0.032 / 0.012 / 0.33 | 92.7% | 5.5% |
| Q4_K_M | 15.783 | +8.6% | 0.099 / 0.034 / 1.11 | 87.7% | 9.2% |
| Q2_K | 144.6 | ×10 | 2.55 / 1.53 / 14.5 | 46.4% | 36.3% |

Q8_0 is close to lossless. Q4_K_M costs much more on a 1.7B model than on the
27B (see below), and Q2_K is not usable. The local Q5_K_M was requantized from
Q8_0 (`llama-quantize --allow-requantize`), so it carries two roundings and
its numbers are slightly pessimistic.

### Qwen3.6-35B-A3B: UD-Q4_K_S against UD-Q4_K_M

BF16 (65 GB) does not fit in 48 GB, so this compares the two quantizations
with each other:

| Weights | PPL | KLD mean / median / p99 vs UD-Q4_K_M | Same top-1 | RMS Δp |
|---|---|---|---|---|
| UD-Q4_K_M | 5.464 | reference | - | - |
| UD-Q4_K_S | 5.459 | 0.0098 / 0.0029 / 0.11 | 96.3% | 3.0% |

The two are equivalent. In dwarfstar UD-Q4_K_S is smaller (20.9 GB vs
22.1 GB) and faster in decode.

### Perplexity of the large models

| Tokenizer | Model | PPL |
|---|---|---|
| Qwen3 (151,936) | Qwen3-30B-A3B asym | 7.677 |
| Qwen3.6 / 3.8 (248,320) | Qwen3.6-35B-A3B UD-Q4_K_S | 5.459 |
| Qwen3.6 / 3.8 (248,320) | Qwen3.8-27B Q4_K_M | 5.613 |

Perplexity is only comparable within one tokenizer. It measures how well the
model predicts text, not reasoning: Qwen3.6 predicts WikiText slightly better
than the 27B but is clearly weaker on hard reasoning and code (next section).

## Published: capabilities (BF16, vendor model cards)

Each generation is evaluated with its own harness and settings, so treat
cross-row comparisons as indicative.

| Model | MMLU-Pro | GPQA | LiveCodeBench v6 | Other |
|---|---|---|---|---|
| Qwen3-1.7B (thinking) | 56.5 | 40.1 | - | IFEval 72.5 |
| Qwen3-30B-A3B, original April release (thinking) | 78.5 | 65.8 | 57.4 | AIME25 70.9, IFEval 86.5 |
| Qwen3.6-35B-A3B | 85.2 | 86.0 | 80.4 | SWE-bench Verified 73.4 |
| Qwen3.8-27B | - | 89.2 (Diamond) | 90.3 | SWE-bench Pro 61.7, Terminal Bench 2.1 73.0, HLE 30.8 |

- The local `Qwen3-30B-A3B-asym` file is the original April release (40,960
  context), not the 2507 refresh. Qwen3-30B-A3B-Instruct-2507 reports GPQA
  70.4 and AIME25 61.3.
- Artificial Analysis gives Qwen3.8-27B an Intelligence Index of 20 at maximum
  reasoning effort (10 without reasoning). On their HLE subset it scores
  15.3%, against 6.9% for Qwen3.6-35B-A3B. Nanbeige4.3-33B also scores 20:
  it is stronger on terminal tasks and weaker on HLE. Within 48 GB, Qwen3.8-27B
  is a strong default but not best at everything.
- The 1.7B Q8 card reports thinking-mode scores in line with BF16: GPQA 39.1
  vs 40.1, MMLU-Redux 74.9 vs 73.9.

## Published: quantization loss of the large models

Qwen3.8-27B, bartowski GGUFs against BF16 (WikiText, 100 chunks). The local
file is the ggml-org Q4_K_M, a different quantization of the same type:

| Weights | PPL | KLD | Same top-1 |
|---|---|---|---|
| BF16 | 6.744 | - | - |
| Q8_0 | 6.751 | 0.0009 | 98.7% |
| Q5_K_M | 6.754 | 0.0053 | 96.9% |
| Q4_K_M | 6.764 | 0.0139 | 95.0% |

For Qwen3.6-35B-A3B against BF16, community runs report UD-Q4_K_M at PPL 6.736
vs 6.722 (KLD 0.011-0.014, same top-1 95.4%) and UD-Q4_K_S at KLD 0.0128
(same top-1 95.0%). Q8_0 reaches KLD 0.0059.

## What to run on 48 GB

| Use | Model | Why |
|---|---|---|
| Best quality | Qwen3.8-27B Q4_K_M | Strongest on reasoning and code. ~11 tok/s decode, thinking on by default. Q8_0 (28.6 GB) fits but is tight; BF16 (53.8 GB) does not. |
| Interactive | Qwen3.6-35B-A3B UD-Q4_K_S | ~55 tok/s decode, ~5x the 27B, and far stronger than the original 30B-A3B. |
| Small / fast | Qwen3-1.7B Q8_0 | Near BF16 at 1.8 GB. Q4_K_M already costs 8.6% PPL at this size. |

## Sources

- Qwen model cards: [Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B),
  [Qwen3-1.7B-GPTQ-Int8](https://huggingface.co/Qwen/Qwen3-1.7B-GPTQ-Int8),
  [Qwen3-30B-A3B](https://huggingface.co/Qwen/Qwen3-30B-A3B),
  [Qwen3-30B-A3B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507),
  [Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B),
  [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B).
- Artificial Analysis:
  [Qwen3.8-27B vs Qwen3.6-35B-A3B](https://artificialanalysis.ai/models/comparisons/qwen3-8-27b-vs-qwen3-6-35b-a3b),
  [Nanbeige4.3-33B](https://artificialanalysis.ai/articles/nanbeige-has-released-nanbeige4-3-33b-scoring-20-on-the-intelligence-index).
- Quantization loss:
  [bartowski Qwen3.8-27B GGUF](https://huggingface.co/bartowski/Qwen3.8-27B-GGUF),
  [Qwen3.6 APEX comparison](https://huggingface.co/mudler/Qwen3.6-35B-A3B-APEX-GGUF),
  [Qwen3.6 PARO comparison](https://huggingface.co/shisa-ai/Qwen3.6-35B-A3B-PARO-full4096-e5-packed).
