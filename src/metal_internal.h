#ifndef DSM_METAL_INTERNAL_H
#define DSM_METAL_INTERNAL_H

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "metal.h"
#include "quants.h"

#include <assert.h>
#include <math.h>
#include <libgen.h>
#include <mach-o/dyld.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

typedef struct { int32_t ne00, ne02; uint64_t nb01, nb02, nb03; int32_t ne12; uint64_t nb10, nb11, nb12, nb13; int32_t ne0, ne1; int16_t r2, r3; } mul_mm_params;
typedef struct { int32_t ne10, ne11; uint64_t nb11, nb12; int32_t ne21, ne20; uint64_t nb21; } mul_mm_id_map0_params;
typedef struct { int32_t ne00, ne02; uint64_t nb01, nb02, nb03; int32_t ne11; uint64_t nb10, nb11, nb12, nb13; int32_t ne20, ne0, ne1; int16_t r2, r3; } mul_mm_id_params;
typedef struct { uint32_t in, out; } gemv_params;
typedef struct {
    int32_t ne00, ne01, ne02;
    uint64_t nb00, nb01, nb02, nb03;
    int32_t ne10, ne11, ne12;
    uint64_t nb10, nb11, nb12, nb13;
    int32_t ne0, ne1, nr0;
    int16_t r2, r3;
} gemv_mul_mv_params;
typedef struct { uint32_t width, groups; float eps; } norm_params;
typedef struct { uint32_t heads, dim, pos; float base; } rope_params;
typedef struct { uint32_t layer, pos, context, width; } cache_params;
typedef struct {
    uint32_t heads, kv_heads, dim, pos, context, layer;
    float base, eps;
} qk_norm_rope_params;
typedef struct { uint32_t heads, kv_heads, dim, pos, context, layer; } attn_params;
typedef struct { uint32_t heads, kv_heads, dim, pos, context, layer, splits; } attn_split_params;
typedef struct { uint32_t heads, dim, pos0, rows; float base; } rope_batch_params;
typedef struct { uint32_t layer, pos0, context, width, rows; } cache_batch_params;
typedef struct { uint32_t heads, kv_heads, dim, pos0, context, layer, rows; } attn_prefill_params;
typedef struct { uint32_t in, out, rows, x_stride, y_stride; } gemv_rows_params;
typedef struct { uint32_t in, out, expert_stride, out_stride, input_stride; } expert_gemv_params;
typedef struct { uint32_t heads, dim, rot, pos; float base; } rope_partial_params;
typedef struct { uint32_t heads, dim, rot, pos0, rows; float base; } rope_partial_batch_params;
typedef struct { uint32_t width, history; } ssm_conv_params;
typedef struct { uint32_t qk_heads, v_heads, state_dim, layer; float eps; } gdn_params;
typedef struct { uint32_t width, history, rows; } ssm_conv_batch_params;
typedef struct { uint32_t qk_heads, v_heads, state_dim, layer, rows, qk_stride, v_stride, out_stride; } gdn_batch_params;

typedef struct { id<MTLBuffer> buffer; uint64_t file_offset, bytes; } weight_view;
typedef enum {
    PROFILE_QKV_GEMV,
    PROFILE_QK_NORM_ROPE,
    PROFILE_KV_STORE,
    PROFILE_ATTENTION,
    PROFILE_O_GEMV,
    PROFILE_MOE_ROUTER,
    PROFILE_MOE_TOPK,
    PROFILE_MOE_EXPERT_GATE_UP,
    PROFILE_MOE_EXPERT_DOWN,
    PROFILE_MOE_ACCUMULATE,
    PROFILE_FFN_GATE_UP_GEMV,
    PROFILE_SWIGLU,
    PROFILE_FFN_DOWN_GEMV,
    PROFILE_RMSNORM,
    PROFILE_RESIDUAL,
    PROFILE_OUTPUT_GEMV,
    PROFILE_OTHER,
    PROFILE_CATEGORY_COUNT,
} profile_category;
typedef struct {
    int enabled;
    uint64_t tokens, dispatches;
    uint64_t weight_bytes[PROFILE_CATEGORY_COUNT];
    double gpu_seconds[PROFILE_CATEGORY_COUNT], cpu_encode_seconds;
} dsm_metal_profile;
typedef struct {
    id<MTLBuffer> conv_state, ssm_state;
    uint32_t pos;
    int valid;
} recurrent_checkpoint;
struct dsm_metal {
    const dsm_model *m; uint32_t pos, context, n_views;
    id<MTLDevice> device; id<MTLCommandQueue> queue; id<MTLResidencySet> residency;
    id<MTLComputePipelineState> q8, q2k, q3k, q4k, q5k, q6k, f32, norm, rope, rope_partial, cache, attention, attention_split, attention_merge, add, add_rmsnorm, qk_norm_rope, swiglu, route_topk, expert_q8, expert_q8_fast, expert_q3k, expert_q4k, expert_q5k, expert_q6k, expert_swiglu, moe_add;
    id<MTLComputePipelineState> split_q_gate, split_gdn_qk, l2norm, ssm_conv, gdn_gates, gdn, gated_rmsnorm, sigmoid_mul, shexp_add, f32_rows, shexp_add_batch;
    id<MTLComputePipelineState> ssm_conv_batch, split_gdn_qk_batch, gdn_gates_batch, gdn_batch;
    id<MTLComputePipelineState> rope_batch, rope_partial_batch, cache_batch, attention_prefill, attention_prefill_d256, attention_prefill_d128_tiled, attention_prefill_fa128, moe_add_batch;
    id<MTLComputePipelineState> mm_q8, mm_q2k, mm_q3k, mm_q4k, mm_q5k, mm_q6k, mm_id_map0, mm_id_q8, mm_id_q2k, mm_id_q3k, mm_id_q4k, mm_id_q5k, mm_id_q6k;
    /* FP32-staged mul_mm/mul_mm_id, indexed q8_0,q2_K,q3_K,q4_K,q6_K; used when mm_f32. */
    id<MTLComputePipelineState> mmf[6], mmf_id[6];
    id<MTLComputePipelineState> mm_f32w;
    int mm_f32, mm_f32w_enabled, decode_fuse;
    weight_view *views;
    id<MTLBuffer> x, norm_x, q, k, v, att, tmp, gate, up, route, expert_ids, expert_weights, scores, keys, values, logits, attention_partial;
    id<MTLBuffer> qg, qkv_mixed, conv_out, z, alpha, beta, conv_state, ssm_state, shared_gate;
    id<MTLBuffer> bx, bnorm, bq, bk, bv, batt, btmp, bgate, bup, broute, bids, bw, mm_id_tpe, mm_id_hids;
    id<MTLBuffer> hb_qkv, hb_conv, hb_dq, hb_dk, hb_z, hb_g, hb_beta, hb_mix, hb_qg, hb_aq, hb_agate, hb_ak, hb_av, hb_out, hb_shg, hb_sgate, hb_sup, hb_sdown;
    uint32_t batch_chunk;
    recurrent_checkpoint *checkpoints;
    uint32_t checkpoint_count, checkpoint_next;
    dsm_metal_profile profile;
};


id<MTLBuffer> dsm_mtl_buffer(id<MTLDevice> device, uint64_t bytes);
void dsm_mtl_set_checked(id<MTLComputeCommandEncoder> encoder, id<MTLBuffer> buffer,
                         NSUInteger offset, NSUInteger need, NSUInteger index);
int dsm_mtl_qwen35_arch(const dsm_model *model);
int dsm_mtl_tensor_view(const dsm_metal *state, const gguf_tensor *tensor,
                        id<MTLBuffer> *buffer_out, uint64_t *offset_out);
void dsm_mtl_dispatch_1d(id<MTLComputeCommandEncoder> encoder,
                         id<MTLComputePipelineState> pipeline, uint32_t count);
int dsm_mtl_prefill_enabled(void);
int dsm_mtl_hybrid_prefill_enabled(void);
int dsm_mtl_attention_decode_enabled(void);
int dsm_mtl_attention_prefill_enabled(void);
int dsm_mtl_attention_flash_enabled(void);
int dsm_mtl_mul_mm_id_enabled(void);

int dsm_mtl_expert_gemv_supported(enum ggml_type type);
id<MTLComputePipelineState> dsm_mtl_expert_gemv_pipeline(
    const dsm_metal *state, enum ggml_type type);
void dsm_mtl_encode_gemv_at(dsm_metal *state,
                            id<MTLComputeCommandEncoder> encoder,
                            const gguf_tensor *weights, id<MTLBuffer> input,
                            NSUInteger input_offset, id<MTLBuffer> output,
                            NSUInteger output_offset);
void dsm_mtl_encode_gemv(dsm_metal *state, id<MTLComputeCommandEncoder> encoder,
                         const gguf_tensor *weights, id<MTLBuffer> input,
                         id<MTLBuffer> output);
void dsm_mtl_encode_f32_rows(dsm_metal *state,
                             id<MTLComputeCommandEncoder> encoder,
                             const gguf_tensor *weights, id<MTLBuffer> input,
                             id<MTLBuffer> output, uint32_t rows,
                             uint32_t input_stride, uint32_t output_stride);
void dsm_mtl_encode_expert_gemv_batch_row(
    dsm_metal *state, id<MTLComputeCommandEncoder> encoder,
    const gguf_tensor *weights, id<MTLBuffer> input, NSUInteger input_offset,
    id<MTLBuffer> output, NSUInteger output_offset, uint32_t row,
    uint32_t input_stride, uint32_t output_stride);
void dsm_mtl_encode_rope(dsm_metal *state,
                         id<MTLComputeCommandEncoder> encoder,
                         id<MTLBuffer> input, uint32_t heads);
void dsm_mtl_encode_moe_router(dsm_metal *state,
                               id<MTLComputeCommandEncoder> encoder,
                               const dsm_layer *layer);
void dsm_mtl_encode_moe_topk(dsm_metal *state,
                             id<MTLComputeCommandEncoder> encoder);
void dsm_mtl_encode_moe_gate_up(dsm_metal *state,
                                id<MTLComputeCommandEncoder> encoder,
                                const dsm_layer *layer);
void dsm_mtl_encode_moe_down(dsm_metal *state,
                             id<MTLComputeCommandEncoder> encoder,
                             const dsm_layer *layer);
void dsm_mtl_encode_moe_accumulate(dsm_metal *state,
                                   id<MTLComputeCommandEncoder> encoder);
void dsm_mtl_encode_moe_topk_batch(dsm_metal *state,
                                   id<MTLComputeCommandEncoder> encoder,
                                   uint32_t rows);
void dsm_mtl_encode_norm_at(dsm_metal *state,
                            id<MTLComputeCommandEncoder> encoder,
                            id<MTLBuffer> input, NSUInteger input_offset,
                            const gguf_tensor *weights, id<MTLBuffer> output,
                            NSUInteger output_offset, uint32_t width,
                            uint32_t groups);
void dsm_mtl_encode_norm(dsm_metal *state,
                         id<MTLComputeCommandEncoder> encoder,
                         id<MTLBuffer> input, const gguf_tensor *weights,
                         id<MTLBuffer> output, uint32_t width, uint32_t groups);
void dsm_mtl_encode_simple(id<MTLComputeCommandEncoder> encoder,
                           id<MTLComputePipelineState> pipeline,
                           id<MTLBuffer> input, id<MTLBuffer> output,
                           uint32_t count);
void dsm_mtl_encode_attention_decode_at(
    dsm_metal *state, id<MTLComputeCommandEncoder> encoder, uint32_t layer,
    uint32_t position, id<MTLBuffer> query, NSUInteger query_offset,
    id<MTLBuffer> output, NSUInteger output_offset);
void dsm_mtl_encode_attention_decode(dsm_metal *state,
                                     id<MTLComputeCommandEncoder> encoder,
                                     uint32_t layer, uint32_t position);

id<MTLComputePipelineState> dsm_mtl_matmul_pipeline(const dsm_metal *state,
                                                     enum ggml_type type);
id<MTLComputePipelineState> dsm_mtl_matmul_id_pipeline(const dsm_metal *state,
                                                        enum ggml_type type);
uint32_t dsm_mtl_matmul_id_max_tiles(uint32_t experts, uint32_t slots,
                                     uint32_t rows);
int dsm_mtl_encode_expert_matmul(dsm_metal *state,
                                 id<MTLComputeCommandEncoder> encoder,
                                 const gguf_tensor *weights, id<MTLBuffer> input,
                                 id<MTLBuffer> output, uint32_t rows,
                                 uint32_t input_rows, uint32_t input_stride,
                                 uint32_t output_stride, int build_map);
void dsm_mtl_encode_matmul(dsm_metal *state,
                           id<MTLComputeCommandEncoder> encoder,
                           const gguf_tensor *weights, id<MTLBuffer> input,
                           id<MTLBuffer> output, uint32_t rows);

int dsm_mtl_profile_token(dsm_metal *state, float *logits);
void dsm_mtl_dump_hybrid_layer(const dsm_metal *state, uint32_t layer);
void dsm_mtl_profile_print(const dsm_metal *state);

#endif
