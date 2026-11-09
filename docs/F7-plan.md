# F7 plan: Qwen hybrid (`qwen35moe` first, then `qwen35`) on Metal (from @oracle review)

Goal: correct greedy text on **Qwen3.6-35B-A3B-UD-Q4_K_S** using the simplest
correct Metal path. Perf work comes after that. Then Qwen3.8-27B dense, then
Q5_K for UD-Q4_K_M.

Inputs: /tmp/dsm-f7/RESEARCH.md (llama.cpp 954de248 refs, "RS §n" below), and
the DSM tree at 5e0494f, read-only. I read the target GGUF header myself with a
DSM-based dump in /tmp/dsm-f7/hdr (meta.txt, tmpl.jinja). The file was still
downloading: 1.38 / 20.89 GB, but the header and tensor directory were complete.

## Facts verified from the target file header (UD-Q4_K_S)

- arch `qwen35moe`, 40 blocks, dim 2048, heads 16 / kv 2, key=value dim **256**,
  rope dim **64**, sections [11,11,10,0], freq_base 1e7, eps 1e-6, experts
  256 / used 8, expert FFN 512, shared FFN 512, ssm conv 4, state 128,
  groups 16, time_step_rank (V heads) 32, inner 4096, full_attention_interval 4.
- No `attention.recurrent_layers`. Schedule: layer i is full attention iff
  (i+1)%4==0 (3,7,...,39), so 30 DeltaNet + 10 attention layers. No `nextn.*`
  (MTP) tensors; 733 tensors total.
- Metadata absent, so use the defaults: `expert_weights_norm`,
  `expert_weights_scale` (1.0), gating func (softmax), `rope.type`. In
  llama.cpp the arch is LLAMA_ROPE_TYPE_IMROPE (llama-model.cpp:3823).
- Tensor types:

  | Type | Tensors |
  |---|---|
  | Q8_0 | token_embd, attn_q (fused Q+gate, [2048,8192]), attn_k/v, attn_output, attn_qkv [2048,8192], attn_gate [2048,4096], ssm_out [4096,2048], all `*_shexp` |
  | Q4_K | ffn_gate/up_exps ×40, ffn_down_exps ×37 |
  | **Q6_K** | ffn_down_exps ×3, output [2048,248320] |
  | F32 | norms, ffn_gate_inp [2048,256], ffn_gate_inp_shexp [2048], ssm_alpha/beta [2048,32], ssm_a, ssm_dt.bias, ssm_conv1d [4,8192], ssm_norm [128] |

  **The research note "Q4_K_S: all routed matrices Q4_K" is wrong.** Three
  layers have Q6_K `ffn_down_exps`.
- The FFN norm is named `post_attention_norm.weight`, not `ffn_norm.weight`.
- Tokenizer: gpt2 / pre `qwen35`, 248320 tokens, 247587 merges, BOS 248044
  (add_bos=0), EOS 248046, pad 248055.
- Chat template (8 KB Jinja): `<|im_start|>role\n…<|im_end|>\n`. The generation
  prompt is `<|im_start|>assistant\n<think>\n` (or `<think>\n\n</think>\n\n` when
  enable_thinking=false). **Earlier assistant turns are re-rendered with their
  reasoning stripped** (tmpl.jinja:93-106). The tool format is identical to
  DSM's current Qwen3-Coder XML.
- Size 20.9 GB < maxBufferLength 30.15 GB, so one weight view. KV for the 10
  attention layers in F32 is 40 KB/token, so a 32k context needs 1.3 GB.
  DeltaNet state is 30 × 2 MiB + conv 30 × 96 KiB ≈ 63 MiB per sequence.

Graph details confirmed in llama.cpp source:
- attn_q output is **per-head interleaved** `[q_h(256) | gate_h(256)]`, stride
  512 (qwen35moe.cpp:293-310).
- V head h uses QK head **h % 16**. The unfused path uses `ggml_repeat_4d`,
  which tiles (:446-450); the fused Metal kernel uses modulo.
- GDN L2 norm is `x / sqrt(sum(x²) + eps)` (models.h:14).
- The DeltaNet output gate is RMSNorm(o)·**SiLU**(z). The attention output gate
  is **sigmoid**.
- AR step order: `S *= exp(g)`, `sk = S·k`, `d = β(v − sk)`, `S += k dᵀ`,
  `o = S·q` with q pre-scaled by 1/√128 (delta-net-base.cpp:335-367).
- Conv: `out[c] = Σ_{j<4} w[c][j] · x[t−3+j]` (tap 3 = current token), then SiLU.
- IMROPE with text-only positions (t=h=w=pos) is mathematically NEOX rope over
  the first 64 of 256 dims (pairs i, i+32), theta = pos · 1e7^(−2i/64)
  (ops.cpp:6240 sector mapping). Dims 64..255 are untouched. **Verify via the
  oracle (risk R3).**

## Decisions

1. **New code paths; existing ones stay byte-identical.** Add
   `encode_layer_hybrid` / `encode_layer_hybrid_batch`. qwen3/qwen3moe keep
   `encode_layer*` unchanged. Gate for every step: `--dump-logits` on
   Qwen3-1.7B-Q8_0 and 30B-asym `cmp`-identical before and after.
2. **Metal only.** `--backend cpu` and the CPU server path reject `qwen35*`
   with a clear error. Don't extend forward.c, because a second in-house
   implementation would repeat the same spec misreadings.
3. **Reference: llama.cpp via Homebrew libllama (b9860, which has qwen35moe).**
   - A tiny tool links `-lllama -lggml -lggml-base` and dumps logits and named
     intermediate tensors via `cb_eval`.
   - It runs as a separate process from DSM: libggml exports `gguf_*` symbols
     that may collide with DSM's `gguf_open/gguf_get_*`.
   - DSM token ids are fed to both sides, so tokenizer and model gates stay
     independent.
   - Noise floor: llama Metal vs llama CPU (`n_gpu_layers=0`) on the same ids.
4. **Port llama kernels verbatim, legacy branches, MIT header**, as in F6:
   - `kernel_gated_delta_net_f32` (FP32 state, `simd_sum`, NSG = 128/32 = 4,
     specialized S_v=128, K=1);
   - `kernel_ssm_conv_f32_f32` and `_batched`;
   - function constants become program-scope constants or template params.
5. **Prefill uses the same serial fused GDN kernel** with n_tokens = rows. This
   is what llama.cpp itself selects on Metal (RS §2 "fused"). The 64-token
   chunked GDA (cumsum/tri/solve_tri) is an optional step only if profiling
   shows GDN > 25% of prefill time.
6. **All SSM math in FP32**: conv, alpha/beta (F32 weights go through the
   per-row `f32_gemv` fallback), softplus/exp gates, state, and gated norm.
   Half precision appears only in existing Q8_0/Q4_K `mul_mm(_id)`
   activations, exactly as on qwen3moe.
7. **Fail loudly.** Init fails if any mapped tensor has no `tensor_view` or no
   pipeline for its type (expert Q6_K!). This avoids the silent `return` pattern
   that caused the F6 Q8_0 view bug.
8. **Session state.** Recurrent state is not position-indexed. Replace
   "truncate to any pos" with "rewind to the newest checkpoint ≤ pos" (Step 2d).

## Step 0: loader, layer schedule, tokenizer, chat template (no GPU)

**0a. Loader** (`src/model.h`, `src/model.c`, `tools/gguf_inspect.c`). Size: M, ~200 LOC.
- `dsm_model` additions:
  - `enum dsm_arch {QWEN3, QWEN3MOE, QWEN35, QWEN35MOE} arch`;
  - `uint8_t *is_recurrent` (per layer), `uint32_t *attn_index`,
    `n_attn_layers`;
  - `rope_dims`, `rope_sections[4]`;
  - `ssm_conv`, `ssm_state`, `ssm_groups`, `ssm_vheads`, `ssm_inner`;
  - `shared_hidden`, `attn_gate_fused` (q projection width = 2 × q_width).
- `dsm_layer` additions:
  - DeltaNet: `qkv, z, conv1d, dt_bias, ssm_a, alpha, beta, ssm_norm, ssm_out`;
  - shared expert: `gate_shexp, up_shexp, down_shexp, gate_inp_shexp`;
  - map `post_attention_norm` to the existing `ffn_norm` field.
- Layer schedule: read `%s.attention.recurrent_layers` (bool array) if present,
  else interval. Validate shapes per layer kind.
- Validate metadata:
  - sections t+h+w == rope_dims/2 and e == 0;
  - V heads % groups == 0;
  - key_dim == value_dim;
  - inner == ssm_vheads × 128.
- Ignore `nextn.*` / `mtp.*` tensors. Fail on any other unknown `blk.*` tensor.
- Add a `gguf_get_array` + int32/bool accessor if needed (gguf.c already has
  `gguf_array_i32_at`).
- Gates:
  - `gguf_inspect --layers` prints `L 0-2 rec, 3 attn, …` (30/10), mapped tensor
    count = 733, unknown = 0;
  - existing models: `--dump-logits` cmp-identical; `make test` passes.

**0b. Tokenizer** (`src/tokenizer.c`, `tools/compare_tokens.sh`). Size: M.
- Accept pre `qwen35`, which is the Qwen2 regex plus `\p{M}` in the letter
  class.
- DSM's `is_letter_at` treats every non-ASCII lead byte as a letter. That's
  already approximate for non-ASCII punctuation, CJK punctuation and combining
  marks. Add a small UTF-8 decode plus category tables for the ranges that
  matter (General Punctuation U+2000-206F, CJK Symbols U+3000-303F, fullwidth
  forms, combining marks U+0300-036F).
- Build a special-token list (type 3/4) at init instead of scanning all 248k
  tokens per byte position (tokenizer.c:129 is O(n·vocab)).
- Fix the `<|endoftext|>` length bug (compares 15 bytes for a 13-byte string,
  tokenizer.c:128).
- Gate: `compare_tokens.sh` with `ORACLE=/opt/homebrew/bin/llama-tokenize`
  (parametrize the path; it currently points at ../llama.cpp) on a ≥20-prompt
  corpus: accents, CJK, emoji, combining marks, long digit runs, code, `\r\n`,
  `<think>`, `<tool_call>`, im_start/im_end. Must be 20/20 exact on Qwen3.6 and
  still 6/6 on Qwen3-1.7B.

**0c. Chat template** (`src/main.c --chat`, `src/server.c make_prompt`). Size: S.
- Keep the hard-coded renderer. For `arch == QWEN35*`:
  - append `<think>\n` to the generation prompt (or `<think>\n\n</think>\n\n`
    when the request sets `enable_thinking:false` / `chat_template_kwargs`);
  - re-render earlier assistant turns without reasoning, matching
    tmpl.jinja:93-106.
- Server: strip `<think>…</think>` from output into `reasoning_content`.
- Gate: rendered prompts byte-identical to `llama-template-analysis` / llama-server
  `/apply-template` for 4 fixtures (plain, system, multi-turn with reasoning,
  tools).

## Step 1: reference tool + harness (Makefile targets `llama-ref`, `check-ref`)

- `tools/llama_ref.c`. Size: S-M, ~250 LOC.
  - Build: `$(CC) -I/opt/homebrew/include -o llama-ref tools/llama_ref.c -L/opt/homebrew/lib -lllama -lggml -lggml-base -Wl,-rpath,/opt/homebrew/lib`
  - Input: `MODEL IDS.txt [--ngl 0|99] [--logits-at all|last|k,k,…] [--greedy N] [--dump-tensors 'l_out-*,linear_attn_out-*,attn_output-*,ffn_moe_out-*,q_conv_predelta-0,…'] OUT_DIR`
  - Context: `type_k = type_v = GGML_TYPE_F32`, flash attention off,
    n_batch = n_ubatch = 512, `cb_eval` copies the named tensors to
    `OUT_DIR/<name>.f32`.
- `tools/ref_check.c` (DSM side, links $(SRC)). Size: M.
  - Run DSM token by token on the same ids, take per-position logits and
    optional `DSM_DUMP_DIR` per-layer residuals, and compare.
  - Report per position: KL(llama‖dsm), max|dprob|, top-1/top-5 agreement, and
    the first greedy divergence with its logit margin.
  - Per layer: relative L2 of `l_out-L`.
- Debug hook in metal.m behind `DSM_DUMP_DIR` (decode only): one command buffer
  per layer, write the residual after each layer. Same pattern as the F6 bisect
  tools in /tmp/dsm-debug.
- Gate for the tool itself: llama Metal vs llama CPU gives a KL floor `F` on 3
  prompts × 64 positions (record it). Greedy text from llama-ref is coherent.
  b9860 is older than the research commit (R11).

**Result (2026-11-07, b9860):** On the 570-token repeated-prefill prompt with
Qwen3.6-35B-A3B-UD-Q4_K_S, Metal versus CPU gave `F` (median KL) =
`1.193815e-4`, max KL = `17.008373`, and top-1 agreement = `98.9474%`.
The very large tail differences mean this one build/backend pair is not yet a
passing oracle floor; retain the raw dumps when diagnosing R11.

## Step 2: decode path (first milestone: correct greedy text)

Only `src/metal.m` and `metal/dsm.metal`. This is one lane: wait for the
flash-decoding fixer to land first.

**2a. Buffers and state** (`dsm_metal_init`):
- `qg[8192]`, `gate[4096]`, `qkv_mixed[8192]`, `conv_out[8192]`, `z[4096]`,
  `ab[64]` (alpha, beta), `gdn_out[4096]`;
- `conv_state[n_rec][3][8192]`;
- `ssm_state[n_rec][32][128][128]` F32 (zeroed at init and at rewind to 0);
- KV sized `n_attn_layers × context × kv × 256`. Cache kernels get
  `attn_index[layer]` as their `layer` param, so the kernels themselves don't
  change.

**2b. Kernels** (new, in dsm.metal; existing param structs untouched):
- `ssm_conv_step(qkv_mixed, conv_state_l, conv_w, out)`: one thread per channel
  (8192). Computes the 4-tap dot with taps [s0, s1, s2, x], then SiLU, then
  shifts the state.
- `l2norm_heads(x, width=128, groups, eps)`: 16 q heads + 16 k heads; the
  caller scales q by 1/√128 (or fold it into the kernel).
- `gdn_gates(ab_raw, dt_bias, ssm_a, beta_out, g_out)`: β = σ(b),
  g = ssm_a · softplus(a + dt). Use a stable softplus: x > 20 ? x : log1p(exp x).
- `gated_delta_net` (port, K=1): grid (1, 32 V heads), threads (32, 4). QK head
  = h % 16. State layout as in llama (transposed).
- `gated_rmsnorm_silu(o, z, ssm_norm, eps)`: groups 32 × width 128.
- `split_q_gate(qg, q, gate)`: de-interleave per head, stride 512.
- `rope_neox_partial(x, RopePartial{heads, head_dim=256, rot=64, pos, base})`.
  Add a new kernel; don't change `rope_neox`.
- `sigmoid_mul(att, gate, n)`.
- `shexp_add(x, y_sh, gate_logit_buf)`: x += σ(g) · y. g comes from `f32_gemv`
  with out = 1 on `ffn_gate_inp_shexp`.
- **`expert_q6k_gemv`**: port the `q6k_gemv` body with expert ids and per-slot
  input stride. **This blocks decode**: 3 layers have Q6_K `down_exps`, and
  today `expert_gemv_pipeline` returns nil, so they would be silently skipped.

**2c. Encoding** (`encode_layer_hybrid`):
- Recurrent layer:
  1. norm;
  2. GEMV qkv (Q8_0), GEMV z (Q8_0), f32 GEMV alpha and beta;
  3. conv;
  4. l2norm q/k;
  5. gates;
  6. GDN;
  7. gated norm;
  8. GEMV ssm_out;
  9. residual add.
- Attention layer:
  1. norm;
  2. GEMV qg (8192), k, v;
  3. split;
  4. rmsnorm q (16 × 256) and k (2 × 256);
  5. partial rope;
  6. kv store at `attn_index`;
  7. `attention_decode`: already generic in dim, scale `rsqrt(p.dim)` = 1/16
     is correct. Check that the flash-decoding `attention_decode_split` handles
     dim 256 before enabling it;
  8. `sigmoid_mul`;
  9. GEMV o;
  10. residual add.
- FFN for every layer:
  1. post_attention_norm;
  2. existing router f32 GEMV, `route_topk` (256 experts = 8 simdgroups, exactly
     its `maxes[8]` limit; add a static check), expert gate/up/swiglu/down,
     `moe_add`;
  3. shared gate/up GEMV (Q8_0), `swiglu`, down GEMV;
  4. `shexp_add`.
- Gates (all via `check-ref`, 3 prompts ≥ 64 tokens so that rope pos > 0 and
  the conv history is exercised):
  - per-layer `l_out` rel L2 ≤ 1e-2 at every layer for the first 16 positions
    (bisect with `linear_attn_out-L`, `attn_output-L`, `ffn_moe_out-L`);
  - top-1 equal at ≥ 99% of positions;
  - KL median ≤ max(1e-3, 3F), max ≤ 1e-2;
  - top-5 sets equal at the last position;
  - 64-token greedy identical on 3 prompts, or the first divergence has a
    llama logit margin < 0.05 (reported);
  - decode deterministic: 2 runs `cmp`-identical (F6 race lesson);
  - `./dsm --chat` answers coherently.
- Size: L, ~600 LOC across metal+m.

**Result 2a-2c (2026-11-08, `ee006bc`).** The first attempt failed because the
1-D `ffn_gate_inp_shexp` weight had `dims[1] = 0`. The F32 GEMV then dispatched
zero rows, so the shared-expert gate stayed σ(0) = 0.5 (fixed in `ee006bc`).
After the fix:
- 5-token prompts (`The capital of France is`, `def fibonacci(n):`): `l_out`
  rel L2 ≤ 2.7e-5 on all 40 layers × 5 positions; KL median 4.4e-11 and
  7.6e-11; top-1 5/5; 32-token greedy identical to `llama-ref --greedy 32`;
  two runs identical.
- 570-token P, all positions:

  | comparison | KL median | KL max | top-1 | top-5 |
  |---|---:|---:|---:|---:|
  | DSM vs llama Metal | 6.3e-5 | 8.2 | 98.4% | 88.4% |
  | DSM vs llama CPU | 1.08e-4 | 17.5 | 98.1% | 85.2% |
  | llama Metal vs llama CPU (floor) | 1.19e-4 | 17.0 | 98.9% | 85.5% |

  DSM is as far from the CPU reference as llama's own Metal backend is (R11),
  so the max-KL gate does not apply to qwen35moe. Use median KL plus greedy
  output instead. The 64-token greedy continuation of P is identical to
  `llama-ref`.
- Speed (token loop, no batched prefill yet): prefill 41 tok/s, decode
  39-43 tok/s.

**2d. Session / rewind** (`src/metal.h`, `src/metal.m`, `src/server.c`, `tools/prefill_check.c`). Size: M.
- API change: `uint32_t dsm_metal_rewind(dsm_metal*, uint32_t pos)` returns the
  position actually restored, ≤ pos. For non-hybrid models it equals pos (same
  as today's truncate). For hybrid models:
  - checkpoints are a blit copy of `ssm_state` + `conv_state` (~63 MB) taken at
    the end of every prefill call;
  - keep a ring of N = 2-4 checkpoints (`DSM_CKPT`);
  - restore the newest checkpoint ≤ pos, else 0 (zero state);
  - KV rows beyond pos are just ignored, as today.
- server.c: `reused = dsm_metal_rewind(...)`, then prefill from `reused`. The
  `--reused` adjustment at line 115 must also be checkpoint-aware: re-feeding
  the last prompt token requires a checkpoint before it.
- Thinking-stripping makes partial reuse the common case. Expect the prefix
  to rewind to the end of the previous prompt.
- Gates:
  - multi-turn script (3 turns, with reasoning): turn-N logits with reuse vs a
    fresh full prefill of the same prompt give KL < 1e-6 and identical greedy
    output;
  - `rewind(k)` for k in {0, ckpt, ckpt+5} then prefill gives the same result
    as fresh.

**Result 2d (2026-11-09, bc49544).** `dsm_metal_rewind` restores a lazy
`DSM_CKPT` (2--4, default 3) ring of 83.75 MiB recurrent-state snapshots and
invalidates future snapshots after every branch. The server checkpoints before
the final `<|im_start|>` (or last token) and after the prompt, so reasoning
stripping can restore the assistant-header prefix. `check-rewind` passed 19 and
581 token prompts at rewind 0/checkpoint/checkpoint+5 with KL 0, same argmax,
top-5, and 32-token greedy continuations; checkpoint copies measured 3.2--11.0
ms while another GPU lane was active.

## Step 3: prefill (batched) via existing mul_mm / mul_mm_id

`encode_layer_hybrid_batch`. Size: L.
- Projections: `encode_matmul` (Q8_0 mul_mm) for qkv, z, qg, k, v, o, ssm_out,
  shexp. F32 alpha/beta/router/shexp-gate use the per-row f32 GEMV fallback
  (correct, slow; fine for now).
- Experts use `mul_mm_id` (q4_K, q6_K exist). Check `mul_mm_id_map0` with 256
  threads = experts and its threadgroup memory 2 × 256 × 4.
- `ssm_conv_batch`: one thread per channel, sequential over rows with the
  3-entry history. Writes the last 3 inputs back, which must handle rows < 3.
- `gated_delta_net` with n_tokens = rows (same port; the kernel loops tokens
  internally).
- Batched l2norm, gates, gated norm (groups × rows), `split_q_gate_batch`,
  `rope_neox_partial_batch`, `sigmoid_mul`, `shexp_add_batch`.
- Attention: `attention_prefill` hard-codes DIM=128, and at 256 the threadgroup
  memory would be 64 KB. Add a templated `attention_prefill_d256` with ROWS=2,
  KEYS=8 → 2·8·256·4 + 2·8·256·4 = 32 KB (q_per_kv = 8). Perf comes later.
- Gates:
  - `check-prefill` (MoE gate: KL < 1e-4, dprob < 1e-3, argmax, top-5,
    64-token greedy) at DSM_PREFILL_CHUNK ∈ {7, 512};
  - n ∈ {1, 2, 3, 4, 33, 511, 512, 513, 1500};
  - split-prompt test: prefill(a) + prefill(b) == prefill(a‖b) (state carry);
  - `check-ref` on the last-token logits with the same thresholds as Step 2.

### Result 3 (2026-11-09)

Batched hybrid prefill is on by default for qwen35 (`DSM_HYBRID_PREFILL=0`
falls back to the token loop). It uses dedicated `hb_*` buffers, one command
buffer per chunk, one serial `gated_delta_net_batch` dispatch per DeltaNet
layer, and row-by-row decode attention kernels (dim 256 `attention_prefill`
is M2).

- Drift root cause: precision, not indexing. With chunk rows ≤ 8 (GEMV path)
  the batch path is bit-exact with the token loop. Above that, half-precision
  staging in `kernel_mul_mm`/`kernel_mul_mm_id` (rel ~3e-4 per op) flips
  near-tied expert choices (first flip: layer 1, gap 1e-4), and the recurrent
  state carries the flip forward. llama.cpp shows the same order of drift
  between its batched and token-by-token runs (last-position KL 1.3e-4 at
  n=33, 2.4e-3 at n=513, 2.8e-2 at n=570).
- Fix: FP32-staged `mul_mm_*_f32` / `mul_mm_id_*_f32` (12 KiB threadgroup
  memory), used by default for qwen35 only (`DSM_MM_F32=0|1` overrides).
  qwen3/qwen3moe keep the half kernels and are unchanged.
- `check-prefill` (Qwen3.6, 570-token repeated prompt P), chunk 512:
  n=33 KL 5.4e-13, n=100 1.4e-13, n=300 3.5e-6, n=511 1.2e-7,
  n=512 7.0e-6, n=513 1.3e-5 (max_dprob 4.2e-4), n=570 4.7e-6 with 64-token
  greedy identical; split 1/100/300 pass (split 100 and 300 are identical to
  the unsplit run). Chunk 7, n ∈ {1, 2, 3, 4, 33}: exact (max_abs 0).
- n=1500: KL 3.5e-8, top-5 same, but greedy differs from DSM's token loop at
  token 46. The hybrid continuation is byte-identical to llama.cpp
  token-by-token (`REF_BATCH=1 ./llama-ref ... --greedy 64`); llama's batched
  run diverges at byte 230 and DSM's token loop at byte 133. So the token loop
  is the outlier there, and the greedy gate at n ≥ 1000 compares against
  llama token-by-token instead.
- `check-metal-gemv` with FP32 staging: per-op rel ≤ 8.2e-7. `check-rewind`
  on P: KL 0. Server: 3 chat turns with checkpoint reuse, fresh-restart turn 3
  identical to the warm run.
- Speed (`./dsm -p P -n 16`, 570 tokens): token loop 52-53 tok/s, batched
  455-479 tok/s (≥ 302 target; llama pp512 754). FP32 staging costs ≤ 3%.

## Step 4: performance (only after Steps 0-3 are green)

- Baseline (2026-11-08, llama.cpp build 9860 `fdb1db877`, GPU idle): pp512
  754.38 ± 2.12 tok/s, tg128 52.06 ± 0.41 tok/s. Targets: decode ≥ 31 tok/s
  (already met by the token path at 39-43), prefill ≥ 302 tok/s.
  - Command: `llama-bench -m models/Qwen3.6-35B-A3B-UD-Q4_K_S.gguf -p 512 -n 128 -fa 1 -r 3`, warm.
  - DSM: `bench-prefill` warm (F6 lesson: measure the 2nd run) and
    `./dsm -n 128 --temp 0`.
- Targets (relative, provisional): decode ≥ 60% of llama tg128, prefill ≥ 40%
  of pp512. For reference, the F6 30B-A3B results were 36 vs 66.8 tg and
  390 warm vs 849 pp.
- Likely levers, in order:
  1. F6 perf items (residency, fused gate+up `mul_mm_id`, attention tiling);
  2. f32 `mul_mm` for alpha/beta/router (or concatenate alpha‖beta into one
     F32 GEMV);
  3. fuse conv + l2norm + gates into one kernel;
  4. GDN NSG tuning;
  5. chunked GDA only if GDN > 25% of prefill.
- Gate: Step 2/3 correctness gates still pass.

## Step 5: `qwen35` dense (Qwen3.8-27B Q4_K_M)

Size: M, once Steps 0-3 exist.
- Dense FFN via the existing `encode_matmul` gate/up/down; no shared expert or
  router.
- 64 layers (48 recurrent / 16 attention), dim 5120, 24 heads / 4 kv
  (q_per_kv = 6, which `attention_prefill_d256` must support), **48 V heads**
  (3 per QK head, still modulo), inner 6144, state 3 MiB per layer
  (144 MiB total).
- token_embd Q4_K (CPU dequant already supported).
- GDN kernel grid y = 48. Specialize nothing on 32.
- Types Q4_K/Q6_K/Q8_0/F32 are all present already.
- Gate: same `check-ref` / `check-prefill` set on the 27B. Memory: 19 GB +
  state, which fits in 40 GB.

### Result 5 (2026-11-09)

Model: ggml-org `Qwen3.8-27B-Q4_K_M.gguf` (19,403,684,480 B; 851 tensors,
48 DeltaNet + 16 attention layers; Q4_K FFN and token_embd, Q8_0
attention/DeltaNet projections and output, F32 norms and ssm vectors).
Commits: `4931846` dense FFN decode, `819c003` dense FFN batched prefill,
`d2c69d5` keeps F32 alpha/beta on the row GEMV path (the 27B stores them as
Q8_0 and goes through `encode_matmul`).

| Gate | Result |
|---|---|
| `gguf-inspect --layers` | 851 mapped, 0 unknown, 48 DeltaNet, 16 attention |
| `check-ref` vs llama Metal, "The capital of France is" | KL median 6.3e-5, max 1.2e-4, top-1 5/5, top-5 100% PASS |
| `check-ref`, "def fibonacci(n):" | KL median 4.0e-5, max 4.7e-5, top-1 4/4 PASS |
| greedy 32 tokens, both prompts | identical to `llama-ref --greedy 32`; two runs identical |
| `check-prefill` P570, chunk 512 | max_abs 1.2e-3, KL 8.1e-11, top-5 same, greedy 64 same |
| `check-prefill --n 33`, chunk 7 | exact (max_abs 0) |
| `check-prefill --n 300 --split 100` | max_abs 1.2e-3, KL 6.0e-11 |
| `check-rewind` (19-token prompt) | KL 3.2e-12, greedy 32 same, stale invalidated |
| qwen3 unchanged | 1.7B Q4_K_M 0.010843754/1.18e-9; 30B-asym 0.53684092/5.48e-7 |

Speed (570-token P, best of 2) vs `llama-bench -p 512 -n 128 -fa 1 -r 2`:

| | DSM | llama.cpp |
|---|---|---|
| prefill, token loop (`DSM_HYBRID_PREFILL=0`) | 11.7 tok/s | |
| prefill, batched | 100.9 tok/s | 125.5 (pp512) |
| decode | 11.3 tok/s (short prompt 11.5) | 11.5 (tg128) |

The dense model is weight-bandwidth bound in decode (≈15 GB read per token),
so DSM and llama.cpp are at parity; prefill is 80% of llama.cpp.

## Step 6 (optional): Q5_K for UD-Q4_K_M (`ffn_down_exps` Q5_K)

Size: M.
- Port llama `mul_mv_q5_K` (the current `q5k_gemv` is the naive `K_GEMV`
  macro), plus `dequantize_q5_K`, `mul_mm_q5_K`, `mul_mm_id_q5_K` and
  `expert_q5k_gemv`.
- Gates: `check-metal-gemv` rel < 2e-3 on a real Q5_K expert tensor; then the
  Step 2/3 gates on UD-Q4_K_M.

## Top correctness risks, and which gate catches each

| # | Risk | Caught by |
|---|---|---|
| R1 | V→QK head map `h % 16` vs `h / 2` | Layer 0 `linear_attn_out-0` rel err (Step 2 per-layer gate) |
| R2 | attn_q per-head interleave (stride 512) vs concatenated Q‖gate | Layer 3 `attn_output-3` |
| R3 | IMROPE ≠ partial NEOX(64) for text, or wrong base/pairing; pos 0 hides it | Layer 3 at positions ≥ 1 (prompts ≥ 64 tokens) |
| R4 | Conv tap order / history off-by-one; rows < 3 in batch | Decode positions ≥ 3; prefill n ∈ {1,2,3,4} + split-prompt test |
| R5 | State reset or not carried (chunk, prefill→decode, server reuse) | `check-prefill` chunk 7, split test, multi-turn rewind test |
| R6 | `ssm_a` sign / double exp, softplus overflow, L2 eps (sum vs mean) | Layer 0 diff |
| R7 | SiLU vs sigmoid swapped between DeltaNet gate and attention gate | Layer 0 / layer 3 diffs |
| R8 | Q6_K expert down silently skipped (nil pipeline) | Decision 7 init check + per-layer diff at those 3 layers |
| R9 | KV indexed by global layer instead of `attn_index`, buffer overrun | Layer 7+ diffs; init sizing check |
| R10 | Shared-expert gate (scalar σ), softmax→top-8→renorm with 256 experts | `ffn_moe_out-L` vs `ffn_out-L` dumps |
| R11 | Oracle itself (b9860) differs from current llama.cpp | Metal vs CPU llama floor; coherent text; spot-check one layer against RESEARCH.md formulas |
| R12 | Tokenizer non-ASCII classes / special tokens | Step 0b 20-prompt corpus vs llama-tokenize |
| R13 | Template drift (thinking stripping, `<think>` prompt) | Step 0c fixtures vs llama's template renderer |
| R14 | Nondeterminism (missing barriers in new reductions) | 2-run `cmp` gate in Steps 2 and 3 |

## Files and parallel lanes

| Lane | Files | Steps | Notes |
|---|---|---|---|
| A loader | src/model.h, src/model.c, src/gguf.[ch] (array accessor), tools/gguf_inspect.c | 0a | Starts now |
| B tokenizer/template | src/tokenizer.[ch], tools/compare_tokens.sh, src/main.c (`--chat` only) | 0b, 0c-main | Starts now |
| C reference | tools/llama_ref.c, tools/ref_check.c, Makefile (`llama-ref`, `check-ref`) | 1 | Starts now; ref_check needs A's model fields |
| D Metal | metal/dsm.metal, src/metal.m | 2a-c, 3, 4 | Single lane; after the flash-decoding fixer lands |
| E session | src/metal.h (rewind API), src/server.c, tools/prefill_check.c | 0c-server, 2d | After D's 2c |

## Blockers found

1. The model file is incomplete (1.38 / 20.89 GB), so there's no llama-bench
   baseline and no oracle run yet.
2. Expert Q6_K GEMV is missing (3 down_exps layers). Without it, decode
   silently skips them.
3. `attention_prefill` is hard-coded to DIM=128; 256 doesn't fit in 32 KB
   threadgroup memory with the current tiling.
4. `rope_neox` has no partial-rotation support. Add a new kernel.
5. Tokenizer:
   - the non-ASCII letter approximation;
   - the O(vocab) special-token scan (248k);
   - the `<|endoftext|>` length bug;
   - the oracle path is hard-coded to ../llama.cpp.
6. The server's prefix-truncate model is incompatible with recurrent state
   (needs checkpoints). The Qwen3.6 template's thinking-stripping makes this
   hit on every multi-turn chat.
7. DSM has no F32 `mul_mm`. alpha/beta/router run per-row in prefill
   (correct, slow).
