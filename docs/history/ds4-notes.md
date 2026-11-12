# ds4 (antirez/DwarfStar 4) — Metal path study notes

Reference clone: `../ds4-ref` (shallow). Only the Apple/Metal path is covered.

## Build
- `make`, host flags `-O3 -ffast-math -mcpu=native -fobjc-arc`, links Foundation + Metal.
- Shaders compiled at runtime from source (`newLibraryWithSource`), concatenated
  with a generated preamble (`ds4_metal.m:4676-4828`). We will start the same way,
  ship a precompiled `.metallib` later.

## GGUF + weights
- In-house GGUF v3 reader: header, metadata, tensor directory; no payload copy
  (`ds4.c:2666-2899`). Type table with block elems/bytes (`ds4.c:2323-2490`).
- Metal: `mmap(MAP_SHARED, PROT_READ)` + `newBufferWithBytesNoCopy`
  (`ds4_metal.m:2285`). Few page-aligned overlapping views bounded by
  `maxBufferLength`, so each tensor lies inside one view (`ds4_metal.m:2203-2314`).
  For small models one view is enough; keep the "view + inner offset" abstraction.

## Forward graph
- Prefill = graph with T>1 tokens, decode = same path with T=1 (`ds4.c:59871`).
- One ordered command sequence per graph, wait at the end; optional early flush
  to overlap host encoding (`ds4.c:59729-59816`). No command buffer per kernel.
- Key kernels: `kernel_mul_mv_q8_0_f32` (`metal/dense.metal:188`), Q4 matvec
  (`dense.metal:2172`), tiled `kernel_mul_mm` (`dense.metal:2515`),
  `kernel_swiglu_f32` (`glu.metal:16`), `kernel_soft_max` (`softmax.metal:31`),
  `kernel_flash_attn_ext*` (`flash_attn.metal`), argsort, Qwen router top-k
  (`qwen4.metal:1122`), MoE expert kernels (`qwen4.metal:2752-3637`,
  `moe.metal` Q2_K / IQ2_XXS / MXFP4 variants).

## MoE
- Router GEMV -> GPU top-k -> expert ids/weights (`ds4.c:59485`).
- Prefill: per-expert token lists + expert GEMM; decode: per-(token, slot) GEMV;
  threshold tuned by benchmark (`ds4.c:59505-59561`).
- SSD streaming: LRU-like cache of (layer, expert) gate/up/down buffers
  (`ds4_metal.m:940-1043`). Copy the separation (router / expert addressing /
  resident-or-streamed cache), not the implementation.

## KV
- GPU KV is model-specific state; disk KV store (`ds4_kvstore.[ch]`) keyed by
  SHA-1 of rendered prefix, budgeted eviction. Defer for us.

## Tokenizer / chat
- Vocab + merges from GGUF metadata; hand-written byte-level BPE pretokenizers
  per family (`ds4.c:43592-44164`). ChatML rendered directly (`ds4.c:44422`).

## Model specificity
- Static shape records validated at load (`ds4.c:546-925`); tensor names, layer
  schedule, attention form, MoE format, special tokens all hardcoded per family.

## Testing
- Scalar reference vs Metal, poisoned buffers, byte-exact A/B
  (`tests/test_mxfp4_metal.c`), full logits/cache parity (`tests/test_deepseek41_graph.c`).

## Principles to copy
1. mmap GGUF + no-copy Metal buffers.
2. Separate decode GEMV from prefill GEMM.
3. One ordered command sequence per forward.
4. Quantization = explicit kernel/layout contract per type.
5. Exactness harnesses before optimization.

## Avoid (for now)
Runtime env overrides, global singletons, 50k-line backend, hyper-connections /
PLE / GDN / vision, SSD expert slabs, speculative decoding, distributed, disk KV,
dozens of quant variants.
