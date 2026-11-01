# F6 plan: chunked Metal prefill (from @oracle review)

Goal: replace token-by-token prefill with chunked GEMM prefill in 5 small,
separately committable steps, each gated by parity.

## Decisions

1. Keep the decode path untouched. Add a separate batch path (`encode_*_batch`)
   with its own `[chunk=512][width]` buffers. Decode stays bitwise identical;
   diffs only add code. ds4 does the same (docs/ds4-notes.md:56).
2. Copy kernels verbatim from llama.cpp, legacy (non-`GGML_METAL_HAS_TENSOR`)
   branches only. ds4's `kernel_mul_mm` is the same copy
   (ds4-ref/metal/dense.metal:2515, instantiated :2931/:2933, dispatched
   ds4_metal.m:19480-19530); map0/mul_mm_id at ds4-ref/metal/moe.metal:7742,
   :7924, :8836. Skip `*_nax_direct_rhs`/MPP variants (tensor hardware).
3. Skip `mul_mv_ext` (llama.cpp uses it only for 2-8 rows,
   ggml-metal-ops.cpp:2769).
4. MoE via `mul_mm_id`, not a per-token expert GEMV grid (per-token reads
   ~0.9 GB experts/token; mul_mm_id reads each expert once per chunk).
5. Precision: `kernel_mul_mm` rounds activations to half in threadgroup memory.
   Steps 0-1 gate: max|dlogit| < 1e-2. Steps 2-3 gate: max|dlogit| < 5e-2,
   same argmax and top-5, identical 64-token greedy on 3 prompts, per-op rel
   error < 2e-3.

## Step 0: refactor metal.m (no behaviour change) + parity tool

- 0a. Reformat only, one statement per line. Check with
  `clang -fobjc-arc -O2 -c src/metal.m` (no -g) before/after: `otool -tV`
  identical.
- 0b. Extract helpers in metal.m, no new buffers:
  ```c
  static void encode_gemv_at(dsm_metal*, id<MTLComputeCommandEncoder>, const gguf_tensor *w,
                             id<MTLBuffer> x, NSUInteger x_off, id<MTLBuffer> y, NSUInteger y_off);
  static void encode_norm_at(..., id<MTLBuffer> x, NSUInteger x_off, const gguf_tensor *w,
                             id<MTLBuffer> out, NSUInteger out_off, uint32_t width, uint32_t groups);
  static void encode_kv_store(dsm_metal*, e, uint32_t layer, uint32_t pos);
  static void encode_attention_decode(dsm_metal*, e, uint32_t layer, uint32_t pos);
  static void encode_layer(dsm_metal*, e, uint32_t layer);   // body of dsm_metal_token's loop
  ```
  `encode_gemv` becomes a wrapper with offsets 0,0; `profile_layer` uses the
  same helpers.
- 0c. `tools/prefill_check.c` + Makefile target `check-prefill`;
  `--dump out.bin` writes last-token logits for a fixed prompt.
- Gate: dumped logits bitwise identical (`cmp`) before/after.

## Step 1: chunked prefill with existing kernels (correctness scaffold)

API (metal.h):
```c
int dsm_metal_prefill(dsm_metal *s, const int32_t *tokens, uint32_t n, float *logits); // last-token logits
```
main.c prompt loop and server.c prefill loop call it for n > 1;
`DSM_PREFILL=0` = old loop; `DSM_PREFILL_CHUNK` (default 512).

New state: batch buffers `bx, bnorm, bq, bk, bv, batt, btmp, bgate, bup,
broute, bids, bw`, each `[chunk][width]` (~80 MB).

```c
static void encode_matmul(dsm_metal*, e, const gguf_tensor *w, id<MTLBuffer> x, id<MTLBuffer> y, uint32_t rows);
static void encode_expert_matmul(dsm_metal*, e, const gguf_tensor *w, id<MTLBuffer> x,
                                 uint32_t x_rows_per_tok /*1 gate/up, 8 down*/, id<MTLBuffer> y, uint32_t T);
static void encode_layer_batch(dsm_metal*, e, uint32_t layer, uint32_t pos0, uint32_t T);
static int  prefill_chunk(dsm_metal*, const int32_t *tok, uint32_t T, float *logits /*NULL unless last*/);
```
- encode_matmul: Q8_0 → `q8_gemv` with ne11=T, nb11=in*4, grid y=T (kernel
  reads r1=tgpig.y, dsm.metal:56; verify dst offset includes r1*ne0). Other
  types: loop rows with encode_gemv_at.
- encode_expert_matmul: loop t<T with existing expert kernels, offsets on x, y
  and bids (t*8*4).
- One command buffer per chunk, one wait. Embeddings dequantized on CPU into
  bx (shared). Output norm + output GEMV only on last row of last chunk (x_off).
- Kernels: reuse rmsnorm (groups = T, T*heads, T*kv_heads),
  add_inplace/swiglu/expert_swiglu (n*T, slots 8*T). `route_topk`: offset by
  tgpig.x, dispatch T threadgroups (decode = 1, identical). New:
  `rope_neox_batch(x, RopeB{heads,dim,pos0,rows,base})` with
  pos = pos0 + idx/(heads*dim/2); `cache_write_batch(src, cache,
  CacheB{layer,pos0,context,width,rows})`; `moe_add_batch(x,tmp,w,dim,rows)`;
  `attention_prefill`.
- attention_prefill: grid (T, kv_heads); G = heads/kv_heads simdgroups per
  threadgroup (one q-head each). TG mem: q[G][128] f32 (4 KB) + K tile
  [32][128] f32 (16 KB). For row t (qpos = pos0+t, uniform per TG), key tiles of
  32: cooperative K load, barrier; lane j: s_j = q·K_j·rsqrt(dim), -inf if
  base+j > qpos; online softmax m' = max(m, simd_max(s)), p = exp(s-m'),
  c = exp(m-m'), l = l*c + simd_sum(p), o *= c; for j:
  o += simd_shuffle(p,j) * V[base+j][4*lane..+3] (float4 from device);
  barrier. Write batt[t][h*128 + 4*lane] = o/l. KV for the chunk is written
  earlier in the same serial encoder. Assert key_dim == value_dim == 128 else
  fall back to old path.
- Gate (check-prefill): n in {1,2,33,511,512,513,1500} plus
  DSM_PREFILL_CHUNK=7; state A = token-by-token after truncate(0), state B =
  prefill; max|d| < 1e-2 (expect ~1e-5), same argmax, same 32-token greedy.
- Target 60-100 tok/s.

## Step 2: dense kernel_mul_mm (Q8_0, then Q6_K/Q4_K/Q3_K)

Copy into dsm.metal (with MIT header): non-tensor `kernel_mul_mm`
(llama kernels/mul_mm.metal:155-364); dequantizers from kernels/dequantize.h:
dequantize_q8_0 (:326), dequantize_q3_K (:401), get_scale_min_k4_just2 (:429),
dequantize_q4_K (:435), dequantize_q6_K (:477); QK_NL (:13); FOR_UNROLL
(common.h:21); `ggml_metal_kargs_mul_mm` (ggml-metal-impl.h ~505-519);
instantiations kernel_mul_mm_{q8_0,q3_K,q4_K,q6_K}_f32 (mul_mm.metal:906-912).
Function constants → program-scope constants: FC_mul_mm_bc_inp=false (host
asserts K % 32 == 0: true for 2048/4096/768), bc_out=true; ne12=ne13=r2=r3=1.

Host (rows > 8, supported type): ne00=in, ne02=1, nb01=row_bytes,
nb02=nb03=0, ne12=1, nb10=4, nb11=in*4, nb12=nb13=0, ne0=out, ne1=rows;
buffers args@0, w@1, x@2, y@3; threadgroup mem 8192 B; threadgroups
(ceil(rows/32), ceil(out/64), 1), threads (32,4,1). Sources:
ggml-metal-ops.cpp:2853-2898, ggml-metal-device.cpp:965-1023.
Test hook `int dsm_metal_matmul_test(dsm_metal*, const gguf_tensor*, const float *x, uint32_t rows, float *y);`
in tools/metal_gemv_check.m vs per-row GEMV, 37 rows, rel < 2e-3.
Target 100-180 tok/s.

## Step 3: MoE mul_mm_id

Copy `kernel_mul_mm_id_map0` (mul_mm.metal:368-509, `<8>` only), non-tensor
branches of `kernel_mul_mm_id` (:549-~890), instantiations
_id_q3_K_f32/_id_q4_K_f32 (:974-975), kargs mul_mm_id_map0/mul_mm_id
(impl.h:564-597). Skip mul_mm_id_zero (no -1 ids). Set FC_mul_mm_id_map0_ne02
(index 707) via MTLFunctionConstantValues = m->experts.

encode_expert_matmul:
1. map0: args {ne10=in, ne11=x_rows_per_tok, nb11=in*4,
   nb12=x_rows_per_tok*in*4, ne21=T, ne20=8, nb21=8*4}; bids → tpe, hids;
   TG mem pad16(2*E*4 + E*8*2); 1 threadgroup of E threads.
2. mul_mm_id: ne00=in, ne02=E, nb01=row_bytes, nb02=expert_stride,
   ne11=x_rows_per_tok, nb10=4, nb11/nb12 as above, ne20=8, ne0=out, ne1=8;
   TG mem 8192; threadgroups (max_tiles, ceil(out/64), 1), threads (128,1,1);
   max_tiles as ops.cpp:2906-2915; host reference ops.cpp:2975-3080.
3. Scratch: tpe (1 + 3*max_tiles) u32, hids (8T + E) i32.
Output layout [t][slot][out] matches bgate/btmp. Test hook
`dsm_metal_expert_matmul_test`. Target 400-900 tok/s on <=4k prompts; aim
>= 60% of `llama-bench -m <gguf> -p 512 -n 0`.

## Step 4 (only if attention > 30% at >= 8k context)

Tile R=4 query rows per threadgroup in attention_prefill: o[R][4] in
registers, K tile reused R times, V staged as half in TG memory, tail rows
active only while key <= qpos. Don't port flash_attn_ext (too big). Target
>= 250 tok/s averaged over a 16k prompt.

## Files per step

| Step | Files |
|---|---|
| 0 | src/metal.m, tools/prefill_check.c, Makefile |
| 1 | src/metal.[hm], metal/dsm.metal, src/main.c, src/server.c, tools/prefill_check.c |
| 2 | metal/dsm.metal, src/metal.[hm], tools/metal_gemv_check.m |
| 3 | metal/dsm.metal, src/metal.[hm], tools/metal_gemv_check.m |
| 4 | metal/dsm.metal |

Never edit decode kernels' param structs; add new structs. Each step ends with
check-prefill passing and tok/s documented for 2k and 16k prompts.
