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

static void set_error(char *out, uint32_t cap, const char *text) { if (cap) snprintf(out, cap, "%s", text); }
static NSString *shader_path(void) {
    const char *env = getenv("DSM_METAL_DIR"); if (env && *env) return [NSString stringWithFormat:@"%s/dsm.metal", env];
    char exe[PATH_MAX]; uint32_t n = sizeof(exe); if (_NSGetExecutablePath(exe, &n) == 0) {
        char path[PATH_MAX]; snprintf(path, sizeof(path), "%s", exe); return [NSString stringWithFormat:@"%s/metal/dsm.metal", dirname(path)];
    }
    return @"metal/dsm.metal";
}
static id<MTLComputePipelineState> pipeline(id<MTLDevice> dev, id<MTLLibrary> lib, NSString *name, char *error, uint32_t cap) {
    NSError *ns_error = nil; id<MTLFunction> fn = [lib newFunctionWithName:name];
    if (!fn) {
        set_error(error, cap, [[NSString stringWithFormat:@"required Metal kernel is missing: %@",
                                                         name] UTF8String]);
        return nil;
    }
    id<MTLComputePipelineState> state = [dev newComputePipelineStateWithFunction:fn error:&ns_error];
    if (!state) set_error(error, cap, ns_error.localizedDescription.UTF8String);
    return state;
}
typedef struct {
    size_t offset;
    NSString *name;
} pipeline_spec;

static int create_pipelines(dsm_metal *state, id<MTLLibrary> library,
                            const pipeline_spec *specs, size_t count,
                            char *error, uint32_t error_size) {
    for (size_t i = 0; i < count; ++i) {
        id<MTLComputePipelineState> *field =
                (id<MTLComputePipelineState> *)((char *)state + specs[i].offset);
        *field = pipeline(state->device, library, specs[i].name, error, error_size);
        if (!*field) return 0;
    }
    return 1;
}

static id<MTLBuffer> buffer(id<MTLDevice> device, uint64_t bytes) {
    return [device newBufferWithLength:(NSUInteger)bytes options:MTLResourceStorageModeShared];
}
static void set_checked(id<MTLComputeCommandEncoder> e, id<MTLBuffer> b,
                        NSUInteger off, NSUInteger need, NSUInteger index) {
    if (!b || off > b.length || need > b.length - off) {
        fprintf(stderr, "dsm: buffer overrun index %lu off %lu need %lu len %lu\n",
                (unsigned long)index, (unsigned long)off, (unsigned long)need,
                (unsigned long)b.length);
        abort();
    }
    [e setBuffer:b offset:off atIndex:index];
}
static uint32_t prefill_chunk_size(const dsm_model *m) {
    const char *value = getenv("DSM_PREFILL_CHUNK");
    char *end = NULL;
    unsigned long chunk = value ? strtoul(value, &end, 10) : 1024;
    if (!value || !*value || *end || !chunk || chunk > UINT32_MAX) chunk = 1024;
    if (!m->is_moe && m->hidden > 16384 && chunk > 512) chunk = 512;
    if (chunk > m->context) chunk = m->context;
    return (uint32_t)chunk;
}
static int prefill_enabled(void) {
    const char *value = getenv("DSM_PREFILL");
    return !value || strcmp(value, "0");
}
static int hybrid_prefill_enabled(void) {
    const char *value = getenv("DSM_HYBRID_PREFILL");
    return !value || strcmp(value, "0");
}
static int f32_weight_matmul_enabled(void) {
    const char *value = getenv("DSM_MM_F32W");
    return !value || strcmp(value, "0");
}
static int mul_mm_id_enabled(void) {
    const char *value = getenv("DSM_MUL_MM_ID");
    return !value || strcmp(value, "0");
}
static int attention_decode_enabled(void) {
    const char *value = getenv("DSM_ATTN_DECODE");
    return !value || strcmp(value, "0");
}
static int attention_prefill_enabled(void) {
    const char *value = getenv("DSM_ATTN_PREFILL");
    return !value || strcmp(value, "0");
}
static int attention_flash_enabled(void) {
    const char *value = getenv("DSM_ATTN_FA");
    return !value || strcmp(value, "0");
}
static int qwen35_arch(const dsm_model *m) {
    return m->arch == DSM_ARCH_QWEN35 || m->arch == DSM_ARCH_QWEN35MOE;
}
static uint32_t checkpoint_ring_size(void) {
    const char *value = getenv("DSM_CKPT");
    char *end = NULL;
    unsigned long count = value ? strtoul(value, &end, 10) : 3;
    if (!value || !*value || *end || count < 2 || count > 4) count = 3;
    return (uint32_t)count;
}
static int checkpoint_buffers(dsm_metal *s) {
    if (!qwen35_arch(s->m) || s->checkpoints) return 1;
    s->checkpoint_count = checkpoint_ring_size();
    s->checkpoints = calloc(s->checkpoint_count, sizeof(*s->checkpoints));
    if (!s->checkpoints) return 0;
    for (uint32_t i = 0; i < s->checkpoint_count; ++i) {
        s->checkpoints[i].conv_state = buffer(s->device, s->conv_state.length);
        s->checkpoints[i].ssm_state = buffer(s->device, s->ssm_state.length);
        if (!s->checkpoints[i].conv_state || !s->checkpoints[i].ssm_state) return 0;
    }
    return 1;
}
static void invalidate_checkpoints_after(dsm_metal *s, uint32_t pos) {
    for (uint32_t i = 0; i < s->checkpoint_count; ++i)
        if (s->checkpoints[i].valid && s->checkpoints[i].pos > pos)
            s->checkpoints[i].valid = 0;
}
static void dump_hybrid_layer(const dsm_metal *s, uint32_t layer) {
    const char *dir = getenv("DSM_DUMP_DIR");
    if (!dir || !*dir) return;
    char path[PATH_MAX];
    snprintf(path, sizeof(path), "%s/l_out-%u-%u.f32", dir, layer, s->pos);
    FILE *file = fopen(path, "wb");
    if (!file) return;
    fwrite(s->x.contents, sizeof(float), s->m->dim, file);
    fclose(file);
}
static double monotonic_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}
static id<MTLComputeCommandEncoder> profile_encoder(dsm_metal *s, id<MTLCommandBuffer> *cb, double *start) {
    *start = monotonic_seconds();
    *cb = [s->queue commandBuffer];
    return [*cb computeCommandEncoder];
}
static int profile_complete(dsm_metal *s, profile_category category, id<MTLCommandBuffer> cb,
                            id<MTLComputeCommandEncoder> e, uint32_t dispatches,
                            uint64_t weight_bytes, double start) {
    [e endEncoding];
    [cb commit];
    s->profile.cpu_encode_seconds += monotonic_seconds() - start;
    [cb waitUntilCompleted];
    if (cb.status != MTLCommandBufferStatusCompleted) return 0;
    double elapsed = cb.GPUEndTime - cb.GPUStartTime;
    if (elapsed > 0) s->profile.gpu_seconds[category] += elapsed;
    s->profile.dispatches += dispatches;
    s->profile.weight_bytes[category] += weight_bytes;
    return 1;
}
static void dsm_metal_profile_print(const dsm_metal *s) {
    static const char *names[PROFILE_CATEGORY_COUNT] = {
        "qkv_gemv", "qk_norm_rope", "kv_store", "attention", "o_gemv",
        "moe_router", "moe_topk", "moe_expert_gate_up", "moe_expert_down", "moe_accumulate",
        "ffn_gate_up_gemv", "swiglu", "ffn_down_gemv", "rmsnorm", "residual",
        "output_gemv", "other",
    };
    double total = 0;
    for (uint32_t i = 0; i < PROFILE_CATEGORY_COUNT; ++i) total += s->profile.gpu_seconds[i];
    fprintf(stderr, "Metal decode profile: %llu tokens, %.1f dispatches/token, CPU encode %.3f ms/token\n",
            (unsigned long long)s->profile.tokens,
            s->profile.tokens ? (double)s->profile.dispatches / s->profile.tokens : 0.0,
            s->profile.tokens ? s->profile.cpu_encode_seconds * 1e3 / s->profile.tokens : 0.0);
    fprintf(stderr, "category                 total ms   ms/token  GPU %%      GB/s\n");
    for (uint32_t i = 0; i < PROFILE_CATEGORY_COUNT; ++i) {
        double seconds = s->profile.gpu_seconds[i];
        double gbps = seconds > 0 && s->profile.weight_bytes[i]
            ? (double)s->profile.weight_bytes[i] / seconds / 1e9 : 0.0;
        if (s->profile.weight_bytes[i]) {
            fprintf(stderr, "%-24s %9.3f %10.3f %6.2f %9.2f\n", names[i], seconds * 1e3,
                    s->profile.tokens ? seconds * 1e3 / s->profile.tokens : 0.0,
                    total ? seconds * 100.0 / total : 0.0, gbps);
        } else {
            fprintf(stderr, "%-24s %9.3f %10.3f %6.2f         -\n", names[i], seconds * 1e3,
                    s->profile.tokens ? seconds * 1e3 / s->profile.tokens : 0.0,
                    total ? seconds * 100.0 / total : 0.0);
        }
    }
    fprintf(stderr, "total                    %9.3f %10.3f 100.00         -\n", total * 1e3,
            s->profile.tokens ? total * 1e3 / s->profile.tokens : 0.0);
}
static int tensor_view(const dsm_metal *s, const gguf_tensor *t, id<MTLBuffer> *buffer_out, uint64_t *offset_out) {
    uint64_t offset = (const uint8_t *)t->data - s->m->gguf.map;
    for (uint32_t i = 0; i < s->n_views; ++i) if (offset >= s->views[i].file_offset &&
        offset - s->views[i].file_offset <= s->views[i].bytes &&
        t->bytes <= s->views[i].bytes - (offset - s->views[i].file_offset)) {
        *buffer_out = s->views[i].buffer; *offset_out = offset - s->views[i].file_offset; return 1;
    }
    return 0;
}
static void dispatch_1d(id<MTLComputeCommandEncoder> e, id<MTLComputePipelineState> p, uint32_t n) {
    NSUInteger w = MIN((NSUInteger)256, p.maxTotalThreadsPerThreadgroup);
    [e dispatchThreads:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(w, 1, 1)];
}
static void encode_simple(id<MTLComputeCommandEncoder> e,id<MTLComputePipelineState> p,id<MTLBuffer> a,id<MTLBuffer> b,uint32_t n);
static id<MTLComputePipelineState> matmul_pipeline(const dsm_metal *s,
                                                    enum ggml_type type);
static id<MTLComputePipelineState> matmul_id_pipeline(const dsm_metal *s,
                                                       enum ggml_type type);
static void encode_matmul(dsm_metal *s, id<MTLComputeCommandEncoder> e,
                          const gguf_tensor *w, id<MTLBuffer> x,
                          id<MTLBuffer> y, uint32_t rows);
static uint32_t matmul_id_max_tiles(uint32_t experts, uint32_t slots,
                                    uint32_t rows);
static int encode_expert_matmul(dsm_metal *s, id<MTLComputeCommandEncoder> e,
                                const gguf_tensor *w, id<MTLBuffer> x,
                                id<MTLBuffer> y, uint32_t rows,
                                uint32_t input_rows, uint32_t input_stride,
                                uint32_t output_stride, int build_map);
static void encode_attention_decode(dsm_metal *s,
                                    id<MTLComputeCommandEncoder> e,
                                    uint32_t layer, uint32_t pos);
static void encode_gemv_at(dsm_metal *s, id<MTLComputeCommandEncoder> e,
                           const gguf_tensor *w, id<MTLBuffer> x,
                           NSUInteger x_off, id<MTLBuffer> y,
                           NSUInteger y_off) {
    id<MTLBuffer> wb; uint64_t off; if (!tensor_view(s, w, &wb, &off)) return;
    /* A 1-D weight (qwen35moe ffn_gate_inp_shexp [dim]) is a one-row matrix;
       the GGUF loader leaves dims[1] = 0 for it. */
    uint64_t rows = w->ndim > 1 ? w->dims[1] : 1;
    gemv_params p = {(uint32_t)w->dims[0], (uint32_t)rows};
    gemv_mul_mv_params mul_mv = {
        .ne00 = (int32_t)w->dims[0], .ne01 = (int32_t)rows, .ne02 = 1,
        .nb00 = 0, .nb01 = w->bytes / rows, .nb02 = 0, .nb03 = 0,
        .ne10 = (int32_t)w->dims[0], .ne11 = 1, .ne12 = 1,
        .nb10 = sizeof(float), .nb11 = 0, .nb12 = 0, .nb13 = 0,
        .ne0 = (int32_t)rows, .ne1 = 1, .nr0 = 0, .r2 = 1, .r3 = 1,
    };
    id<MTLComputePipelineState> state = s->f32;
    switch (w->type) {
        case GGML_TYPE_Q8_0: state = s->q8; break;
        case GGML_TYPE_Q2_K: state = s->q2k; break;
        case GGML_TYPE_Q3_K: state = s->q3k; break;
        case GGML_TYPE_Q4_K: state = s->q4k; break;
        case GGML_TYPE_Q5_K: state = s->q5k; break;
        case GGML_TYPE_Q6_K: state = s->q6k; break;
    }
    [e setComputePipelineState:state]; [e setBuffer:wb offset:(NSUInteger)off atIndex:0]; [e setBuffer:x offset:x_off atIndex:1]; [e setBuffer:y offset:y_off atIndex:2];
    if (w->type == GGML_TYPE_Q8_0) {
        [e setBytes:&mul_mv length:sizeof(mul_mv) atIndex:3];
        [e setThreadgroupMemoryLength:32 * sizeof(float) * 2 atIndex:0];
        [e dispatchThreadgroups:MTLSizeMake((p.out + 1) / 2, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 4, 1)];
    } else if (w->type == GGML_TYPE_Q2_K) {
        [e setBytes:&mul_mv length:sizeof(mul_mv) atIndex:3];
        [e dispatchThreadgroups:MTLSizeMake((p.out + 7) / 8, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 2, 1)];
    } else if (w->type == GGML_TYPE_Q3_K) {
        [e setBytes:&mul_mv length:sizeof(mul_mv) atIndex:3];
        [e dispatchThreadgroups:MTLSizeMake((p.out + 3) / 4, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 2, 1)];
    } else if (w->type == GGML_TYPE_Q5_K) {
        [e setBytes:&mul_mv length:sizeof(mul_mv) atIndex:3];
        [e dispatchThreadgroups:MTLSizeMake((p.out + 1) / 2, 1, 1)
  threadsPerThreadgroup:MTLSizeMake(32, 2, 1)];
    } else if (w->type == GGML_TYPE_Q4_K || w->type == GGML_TYPE_Q6_K) {
        [e setBytes:&mul_mv length:sizeof(mul_mv) atIndex:3];
        [e dispatchThreadgroups:MTLSizeMake((p.out + 3) / 4, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 2, 1)];
    } else {
        [e setBytes:&p length:sizeof(p) atIndex:3];
        [e setThreadgroupMemoryLength:256 * sizeof(float) atIndex:0]; [e dispatchThreadgroups:MTLSizeMake(p.out, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    }
}
static void encode_f32_rows(dsm_metal *s, id<MTLComputeCommandEncoder> e,
                            const gguf_tensor *w, id<MTLBuffer> x,
                            id<MTLBuffer> y, uint32_t rows,
                            uint32_t x_stride, uint32_t y_stride) {
    id<MTLBuffer> wb;
    uint64_t off;
    if (!tensor_view(s, w, &wb, &off)) return;
    gemv_rows_params p = {(uint32_t)w->dims[0],
                          (uint32_t)(w->ndim > 1 ? w->dims[1] : 1), rows,
                          x_stride, y_stride};
    [e setComputePipelineState:s->f32_rows];
    set_checked(e, wb, (NSUInteger)off, (NSUInteger)w->bytes, 0);
    set_checked(e, x, 0, (NSUInteger)rows * x_stride * sizeof(float), 1);
    set_checked(e, y, 0, (NSUInteger)rows * y_stride * sizeof(float), 2);
    [e setBytes:&p length:sizeof(p) atIndex:3];
    [e setThreadgroupMemoryLength:256 * sizeof(float) atIndex:0];
    [e dispatchThreadgroups:MTLSizeMake(p.out, rows, 1)
  threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
}
static void encode_gemv(dsm_metal *s, id<MTLComputeCommandEncoder> e,
                         const gguf_tensor *w, id<MTLBuffer> x,
                         id<MTLBuffer> y) {
    encode_gemv_at(s, e, w, x, 0, y, 0);
}
static id<MTLComputePipelineState> expert_gemv_pipeline(const dsm_metal *s,
                                                         enum ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q8_0: return s->expert_q8_fast;
        case GGML_TYPE_Q3_K: return s->expert_q3k;
        case GGML_TYPE_Q4_K: return s->expert_q4k;
        case GGML_TYPE_Q5_K: return s->expert_q5k;
        case GGML_TYPE_Q6_K: return s->expert_q6k;
        default: return nil;
    }
}
static int expert_gemv_supported(enum ggml_type type) {
    return type == GGML_TYPE_Q8_0 || type == GGML_TYPE_Q3_K ||
            type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q5_K || type == GGML_TYPE_Q6_K;
}
static void encode_expert_gemv(dsm_metal *s, id<MTLComputeCommandEncoder> e,
                               const gguf_tensor *w, id<MTLBuffer> x,
                               id<MTLBuffer> y) {
    id<MTLBuffer> wb; uint64_t off;
    if (!tensor_view(s, w, &wb, &off)) return;
    id<MTLComputePipelineState> state = expert_gemv_pipeline(s, w->type);
    if (!state) return;
    expert_gemv_params p = {(uint32_t)w->dims[0], (uint32_t)w->dims[1],
                             (uint32_t)(w->bytes / w->dims[2]), (uint32_t)w->dims[1], 0};
    [e setComputePipelineState:state]; [e setBuffer:wb offset:(NSUInteger)off atIndex:0];
    [e setBuffer:x offset:0 atIndex:1]; [e setBuffer:y offset:0 atIndex:2];
    [e setBytes:&p length:sizeof(p) atIndex:3]; [e setBuffer:s->expert_ids offset:0 atIndex:4];
    if (w->type == GGML_TYPE_Q8_0) {
        [e setThreadgroupMemoryLength:2 * 32 * sizeof(float) atIndex:0];
        [e dispatchThreadgroups:MTLSizeMake((p.out + 1) / 2, 1, s->m->experts_used)
 threadsPerThreadgroup:MTLSizeMake(32, 4, 1)];
    } else if (w->type == GGML_TYPE_Q3_K || w->type == GGML_TYPE_Q4_K || w->type == GGML_TYPE_Q5_K) {
        [e dispatchThreadgroups:MTLSizeMake((p.out + 3) / 4, 1, s->m->experts_used) threadsPerThreadgroup:MTLSizeMake(32, 2, 1)];
    } else {
        [e setThreadgroupMemoryLength:256 * sizeof(float) atIndex:0];
        [e dispatchThreadgroups:MTLSizeMake(p.out, 1, s->m->experts_used) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    }
}
static void encode_moe_router(dsm_metal *s, id<MTLComputeCommandEncoder> e, const dsm_layer *w) {
    encode_gemv(s, e, w->gate_inp, s->norm_x, s->route);
}
static void encode_expert_gemv_batch_row(dsm_metal *s,
                                         id<MTLComputeCommandEncoder> e,
                                         const gguf_tensor *w, id<MTLBuffer> x,
                                         NSUInteger x_off, id<MTLBuffer> y,
                                         NSUInteger y_off, uint32_t row,
                                         uint32_t input_stride,
                                         uint32_t output_stride) {
    id<MTLBuffer> wb;
    uint64_t off;
    if (!tensor_view(s, w, &wb, &off)) return;
    id<MTLComputePipelineState> state = expert_gemv_pipeline(s, w->type);
    if (!state) return;
    expert_gemv_params p = {(uint32_t)w->dims[0], (uint32_t)w->dims[1],
                             (uint32_t)(w->bytes / w->dims[2]), output_stride,
                             input_stride};
    [e setComputePipelineState:state];
    [e setBuffer:wb offset:(NSUInteger)off atIndex:0];
    [e setBuffer:x offset:x_off atIndex:1];
    [e setBuffer:y offset:y_off atIndex:2];
    [e setBytes:&p length:sizeof(p) atIndex:3];
    [e setBuffer:s->bids offset:(NSUInteger)row * s->m->experts_used * sizeof(uint32_t) atIndex:4];
    if (w->type == GGML_TYPE_Q8_0) {
        [e setThreadgroupMemoryLength:2 * 32 * sizeof(float) atIndex:0];
        [e dispatchThreadgroups:MTLSizeMake((p.out + 1) / 2, 1, s->m->experts_used)
 threadsPerThreadgroup:MTLSizeMake(32, 4, 1)];
    } else if (w->type == GGML_TYPE_Q3_K || w->type == GGML_TYPE_Q4_K || w->type == GGML_TYPE_Q5_K) {
        [e dispatchThreadgroups:MTLSizeMake((p.out + 3) / 4, 1, s->m->experts_used)
  threadsPerThreadgroup:MTLSizeMake(32, 2, 1)];
    } else {
        [e setThreadgroupMemoryLength:256 * sizeof(float) atIndex:0];
        [e dispatchThreadgroups:MTLSizeMake(p.out, 1, s->m->experts_used)
  threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    }
}
static void encode_moe_topk(dsm_metal *s, id<MTLComputeCommandEncoder> e) {
    const dsm_model *m = s->m;
    [e setComputePipelineState:s->route_topk]; [e setBuffer:s->route offset:0 atIndex:0];
    [e setBuffer:s->expert_ids offset:0 atIndex:1]; [e setBuffer:s->expert_weights offset:0 atIndex:2];
    [e setBytes:&m->experts length:sizeof(m->experts) atIndex:3];
    [e setBytes:&m->experts_used length:sizeof(m->experts_used) atIndex:4];
    [e setThreadgroupMemoryLength:(m->experts + 8) * sizeof(float) atIndex:0];
    [e dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
}
static void encode_moe_topk_batch(dsm_metal *s, id<MTLComputeCommandEncoder> e,
                                   uint32_t rows) {
    const dsm_model *m = s->m;
    [e setComputePipelineState:s->route_topk];
    [e setBuffer:s->broute offset:0 atIndex:0];
    [e setBuffer:s->bids offset:0 atIndex:1];
    [e setBuffer:s->bw offset:0 atIndex:2];
    [e setBytes:&m->experts length:sizeof(m->experts) atIndex:3];
    [e setBytes:&m->experts_used length:sizeof(m->experts_used) atIndex:4];
    [e setThreadgroupMemoryLength:(m->experts + 8) * sizeof(float) atIndex:0];
    [e dispatchThreadgroups:MTLSizeMake(rows, 1, 1)
 threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
}
static void encode_moe_gate_up(dsm_metal *s, id<MTLComputeCommandEncoder> e, const dsm_layer *w) {
    const dsm_model *m = s->m;
    encode_expert_gemv(s, e, w->gate_exps, s->norm_x, s->gate);
    encode_expert_gemv(s, e, w->up_exps, s->norm_x, s->up);
    uint32_t rows = 1;
    [e setComputePipelineState:s->expert_swiglu]; [e setBuffer:s->gate offset:0 atIndex:0];
    [e setBuffer:s->up offset:0 atIndex:1]; [e setBytes:&m->expert_hidden length:sizeof(m->expert_hidden) atIndex:2];
    [e setBytes:&m->experts_used length:sizeof(m->experts_used) atIndex:3]; [e setBytes:&rows length:sizeof(rows) atIndex:4];
    dispatch_1d(e, s->expert_swiglu, m->expert_hidden * m->experts_used);
}
static void encode_moe_down(dsm_metal *s, id<MTLComputeCommandEncoder> e, const dsm_layer *w) {
    const dsm_model *m = s->m;
    id<MTLBuffer> wb; uint64_t off;
    if (!tensor_view(s, w->down_exps, &wb, &off)) return;
    expert_gemv_params p = {(uint32_t)w->down_exps->dims[0], (uint32_t)w->down_exps->dims[1],
                             (uint32_t)(w->down_exps->bytes / w->down_exps->dims[2]), m->dim, m->expert_hidden};
    id<MTLComputePipelineState> state = expert_gemv_pipeline(s, w->down_exps->type);
    if (!state) return;
    [e setComputePipelineState:state]; [e setBuffer:wb offset:(NSUInteger)off atIndex:0];
    [e setBuffer:s->gate offset:0 atIndex:1]; [e setBuffer:s->tmp offset:0 atIndex:2];
    [e setBytes:&p length:sizeof(p) atIndex:3]; [e setBuffer:s->expert_ids offset:0 atIndex:4];
    if (w->down_exps->type == GGML_TYPE_Q8_0) {
        [e setThreadgroupMemoryLength:2 * 32 * sizeof(float) atIndex:0];
        [e dispatchThreadgroups:MTLSizeMake((p.out + 1) / 2, 1, m->experts_used) threadsPerThreadgroup:MTLSizeMake(32, 4, 1)];
    } else if (w->down_exps->type == GGML_TYPE_Q3_K || w->down_exps->type == GGML_TYPE_Q4_K || w->down_exps->type == GGML_TYPE_Q5_K) {
        [e dispatchThreadgroups:MTLSizeMake((p.out + 3) / 4, 1, m->experts_used) threadsPerThreadgroup:MTLSizeMake(32, 2, 1)];
    } else {
        [e setThreadgroupMemoryLength:256 * sizeof(float) atIndex:0];
        [e dispatchThreadgroups:MTLSizeMake(p.out, 1, m->experts_used) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    }
}
static void encode_moe_accumulate(dsm_metal *s, id<MTLComputeCommandEncoder> e) {
    const dsm_model *m = s->m;
    [e setComputePipelineState:s->moe_add]; [e setBuffer:s->x offset:0 atIndex:0];
    [e setBuffer:s->tmp offset:0 atIndex:1]; [e setBuffer:s->expert_weights offset:0 atIndex:2];
    [e setBytes:&m->experts_used length:sizeof(m->experts_used) atIndex:3]; [e setBytes:&m->dim length:sizeof(m->dim) atIndex:4];
    dispatch_1d(e, s->moe_add, m->dim);
}
static void encode_moe(dsm_metal *s, id<MTLComputeCommandEncoder> e, const dsm_layer *w) {
    encode_moe_router(s, e, w); encode_moe_topk(s, e);
    encode_moe_gate_up(s, e, w); encode_moe_down(s, e, w); encode_moe_accumulate(s, e);
}
static void encode_norm_at(dsm_metal *s, id<MTLComputeCommandEncoder> e,
                           id<MTLBuffer> x, NSUInteger x_off,
                           const gguf_tensor *w, id<MTLBuffer> out,
                           NSUInteger out_off, uint32_t width,
                           uint32_t groups) {
    id<MTLBuffer> wb; uint64_t off; if (!tensor_view(s,w,&wb,&off)) return; norm_params p={width,groups,s->m->rms_eps};
    [e setComputePipelineState:s->norm];[e setBuffer:x offset:x_off atIndex:0];[e setBuffer:wb offset:(NSUInteger)off atIndex:1];[e setBuffer:out offset:out_off atIndex:2];[e setBytes:&p length:sizeof(p) atIndex:3];[e setThreadgroupMemoryLength:256 * sizeof(float) atIndex:0];[e dispatchThreadgroups:MTLSizeMake(groups,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
}
static void encode_norm(dsm_metal *s, id<MTLComputeCommandEncoder> e,
                        id<MTLBuffer> x, const gguf_tensor *w,
                        id<MTLBuffer> out, uint32_t width, uint32_t groups) {
    encode_norm_at(s, e, x, 0, w, out, 0, width, groups);
}
static void encode_rope(dsm_metal *s,id<MTLComputeCommandEncoder> e,id<MTLBuffer> x,uint32_t heads) { rope_params p={heads,s->m->key_dim,s->pos,s->m->rope_base};[e setComputePipelineState:s->rope];[e setBuffer:x offset:0 atIndex:0];[e setBytes:&p length:sizeof(p) atIndex:1];dispatch_1d(e,s->rope,heads*(s->m->key_dim/2)); }
static void encode_simple(id<MTLComputeCommandEncoder> e,id<MTLComputePipelineState> p,id<MTLBuffer> a,id<MTLBuffer> b,uint32_t n) { [e setComputePipelineState:p];[e setBuffer:a offset:0 atIndex:0];[e setBuffer:b offset:0 atIndex:1];[e setBytes:&n length:4 atIndex:2];dispatch_1d(e,p,n); }
static void encode_qk_norm_rope(dsm_metal *s, id<MTLComputeCommandEncoder> e,
                                const dsm_layer *w, uint32_t layer) {
    id<MTLBuffer> qw, kw;
    uint64_t qoff, koff;
    if (!tensor_view(s, w->q_norm, &qw, &qoff) ||
        !tensor_view(s, w->k_norm, &kw, &koff)) return;
    qk_norm_rope_params p = {s->m->heads, s->m->kv_heads, s->m->key_dim,
                             s->pos, s->context, layer, s->m->rope_base,
                             s->m->rms_eps};
    [e setComputePipelineState:s->qk_norm_rope];
    [e setBuffer:s->q offset:0 atIndex:0];
    [e setBuffer:s->k offset:0 atIndex:1];
    [e setBuffer:qw offset:(NSUInteger)qoff atIndex:2];
    [e setBuffer:kw offset:(NSUInteger)koff atIndex:3];
    [e setBuffer:s->keys offset:0 atIndex:4];
    [e setBytes:&p length:sizeof(p) atIndex:5];
    [e setThreadgroupMemoryLength:256 * sizeof(float) atIndex:0];
    [e dispatchThreadgroups:MTLSizeMake(s->m->heads + s->m->kv_heads, 1, 1)
  threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
}
static void encode_add_rmsnorm(dsm_metal *s, id<MTLComputeCommandEncoder> e,
                               const gguf_tensor *weight) {
    id<MTLBuffer> wb;
    uint64_t off;
    if (!tensor_view(s, weight, &wb, &off)) return;
    norm_params p = {s->m->dim, 1, s->m->rms_eps};
    [e setComputePipelineState:s->add_rmsnorm];
    [e setBuffer:s->x offset:0 atIndex:0];
    [e setBuffer:s->tmp offset:0 atIndex:1];
    [e setBuffer:wb offset:(NSUInteger)off atIndex:2];
    [e setBuffer:s->norm_x offset:0 atIndex:3];
    [e setBytes:&p length:sizeof(p) atIndex:4];
    [e setThreadgroupMemoryLength:256 * sizeof(float) atIndex:0];
    [e dispatchThreadgroups:MTLSizeMake(1, 1, 1)
  threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
}

int dsm_metal_available(void) { return MTLCreateSystemDefaultDevice() != nil; }
static uint64_t max_buffer_bytes(id<MTLDevice> device) {
    uint64_t max = [device maxBufferLength];
    /* Test hook: smaller views make tensors straddle view boundaries (make check-views). */
    const char *value = getenv("DSM_MAX_BUFFER_BYTES");
    char *end = NULL;
    unsigned long long cap = value ? strtoull(value, &end, 10) : 0;
    if (value && *value && !*end && cap && cap < max) max = cap;
    return max;
}
/* Wrap the mmapped GGUF in no-copy Metal buffers. A file can be larger than
   maxBufferLength, so consecutive views overlap by more than the largest tensor:
   every tensor then lies entirely inside at least one view. */
static int create_weight_views(dsm_metal *s, char *error, uint32_t error_size) {
    const dsm_model *m = s->m;
    const uint64_t page = (uint64_t)getpagesize();
    const uint64_t start = m->gguf.tensor_data_offset & ~(page - 1), end = m->gguf.size;
    const uint64_t max = max_buffer_bytes(s->device) & ~(page - 1);
    if (!max) { set_error(error, error_size, "Metal max buffer length is zero"); return 0; }
    uint64_t max_tensor = 0;
    for (uint64_t i = 0; i < m->gguf.n_tensors; ++i)
        if (m->gguf.tensors[i].bytes > max_tensor) max_tensor = m->gguf.tensors[i].bytes;
    const uint64_t overlap = ((max_tensor + page - 1) & ~(page - 1)) + page;
    if (overlap >= max) {
        set_error(error, error_size, "tensor larger than Metal max buffer length");
        return 0;
    }
    const uint64_t step = end - start <= max ? max : max - overlap;
    s->n_views = end - start <= max ? 1 : (uint32_t)((end - start - overlap + step - 1) / step);
    s->views = calloc(s->n_views, sizeof(*s->views));
    if (!s->views) { set_error(error, error_size, "out of memory"); return 0; }
    for (uint32_t i = 0; i < s->n_views; ++i) {
        uint64_t off = start + (uint64_t)i * step, bytes = end - off < max ? end - off : max;
        bytes = (bytes + page - 1) & ~(page - 1);
        s->views[i].buffer = [s->device newBufferWithBytesNoCopy:(void *)(m->gguf.map + off)
                                                          length:(NSUInteger)bytes
                                                         options:MTLResourceStorageModeShared
                                                     deallocator:nil];
        s->views[i].file_offset = off;
        s->views[i].bytes = bytes;
        if (!s->views[i].buffer) {
            set_error(error, error_size, "cannot wrap GGUF mmap in Metal buffer");
            return 0;
        }
    }
    for (uint64_t i = 0; i < m->gguf.n_tensors; ++i) {
        id<MTLBuffer> buffer;
        uint64_t offset;
        if (!tensor_view(s, &m->gguf.tensors[i], &buffer, &offset)) {
            set_error(error, error_size, "GGUF tensor not mappable to Metal view");
            return 0;
        }
    }
    if (getenv("DSM_MAX_BUFFER_BYTES"))
        fprintf(stderr, "dsm: %u Metal weight views of up to %llu bytes\n", s->n_views,
                (unsigned long long)max);
    if (@available(macOS 15.0, *)) {
        MTLResidencySetDescriptor *descriptor = [MTLResidencySetDescriptor new];
        descriptor.initialCapacity = s->n_views;
        NSError *ns_error = nil;
        s->residency = [s->device newResidencySetWithDescriptor:descriptor error:&ns_error];
        if (s->residency) {
            for (uint32_t i = 0; i < s->n_views; ++i) [s->residency addAllocation:s->views[i].buffer];
            [s->residency commit];
            [s->residency requestResidency];
            [s->queue addResidencySet:s->residency];
        }
    }
    return 1;
}
int dsm_metal_init(dsm_metal **out,const dsm_model *m,uint32_t context,char *error,uint32_t error_size) {
    NSError *ns_error = nil;
    NSString *source = nil;
    MTLCompileOptions *options = nil;
    id<MTLLibrary> lib = nil;
    *out=NULL; if (!context || context > m->context) { set_error(error,error_size,"invalid Metal context"); return 0; }
    dsm_metal *s=calloc(1,sizeof(*s)); if(!s){set_error(error,error_size,"out of memory");return 0;} s->m=m;s->context=context;s->profile.enabled=getenv("DSM_PROFILE") && !strcmp(getenv("DSM_PROFILE"),"1");s->device=MTLCreateSystemDefaultDevice(); if(!s->device){set_error(error,error_size,"no Metal device");goto bad;}s->queue=[s->device newCommandQueue];
    source=[NSString stringWithContentsOfFile:shader_path() encoding:NSUTF8StringEncoding error:&ns_error];if(!source){set_error(error,error_size,ns_error.localizedDescription.UTF8String);goto bad;}options=[MTLCompileOptions new];if (@available(macOS 15.0,*)) options.mathMode=MTLMathModeSafe;
    lib=[s->device newLibraryWithSource:source options:options error:&ns_error];if(!lib){set_error(error,error_size,ns_error.localizedDescription.UTF8String);goto bad;}
    if (m->is_moe) for (uint32_t layer = 0; layer < m->layers; ++layer) {
        const dsm_layer *w = &m->layer[layer];
        if (!expert_gemv_supported(w->gate_exps->type) ||
            !expert_gemv_supported(w->up_exps->type) ||
            !expert_gemv_supported(w->down_exps->type)) {
            set_error(error, error_size, "unsupported Metal MoE expert quantization"); goto bad;
        }
    }
    if (m->is_moe && (m->experts > 256 || (m->experts + 31) / 32 > 8)) {
        set_error(error, error_size, "Metal route_topk supports at most 256 experts"); goto bad;
    }
    static const pipeline_spec specs[] = {
        {offsetof(struct dsm_metal, q8), @"q8_gemv"},
        {offsetof(struct dsm_metal, q2k), @"q2k_gemv"},
        {offsetof(struct dsm_metal, q3k), @"q3k_gemv"},
        {offsetof(struct dsm_metal, q4k), @"q4k_gemv"},
        {offsetof(struct dsm_metal, q5k), @"q5k_gemv"},
        {offsetof(struct dsm_metal, q6k), @"q6k_gemv"},
        {offsetof(struct dsm_metal, f32), @"f32_gemv"},
        {offsetof(struct dsm_metal, norm), @"rmsnorm"},
        {offsetof(struct dsm_metal, rope), @"rope_neox"},
        {offsetof(struct dsm_metal, rope_partial), @"rope_neox_partial"},
        {offsetof(struct dsm_metal, split_q_gate), @"split_q_gate"},
        {offsetof(struct dsm_metal, split_gdn_qk), @"split_gdn_qk"},
        {offsetof(struct dsm_metal, l2norm), @"l2norm_heads"},
        {offsetof(struct dsm_metal, ssm_conv), @"ssm_conv_step"},
        {offsetof(struct dsm_metal, gdn_gates), @"gdn_gates"},
        {offsetof(struct dsm_metal, gdn), @"gated_delta_net"},
        {offsetof(struct dsm_metal, gated_rmsnorm), @"gated_rmsnorm_silu"},
        {offsetof(struct dsm_metal, sigmoid_mul), @"sigmoid_mul"},
        {offsetof(struct dsm_metal, shexp_add), @"shexp_add"},
        {offsetof(struct dsm_metal, ssm_conv_batch), @"ssm_conv_batch"},
        {offsetof(struct dsm_metal, split_gdn_qk_batch), @"split_gdn_qk_batch"},
        {offsetof(struct dsm_metal, gdn_gates_batch), @"gdn_gates_batch"},
        {offsetof(struct dsm_metal, gdn_batch), @"gated_delta_net_batch"},
        {offsetof(struct dsm_metal, cache), @"cache_write"},
        {offsetof(struct dsm_metal, attention), @"attention_decode"},
        {offsetof(struct dsm_metal, attention_split), @"attention_decode_split"},
        {offsetof(struct dsm_metal, attention_merge), @"attention_decode_merge"},
        {offsetof(struct dsm_metal, add), @"add_inplace"},
        {offsetof(struct dsm_metal, add_rmsnorm), @"add_rmsnorm"},
        {offsetof(struct dsm_metal, qk_norm_rope), @"qk_norm_rope"},
        {offsetof(struct dsm_metal, swiglu), @"swiglu"},
        {offsetof(struct dsm_metal, route_topk), @"route_topk"},
        {offsetof(struct dsm_metal, expert_q8), @"expert_q8_0_gemv"},
        {offsetof(struct dsm_metal, expert_q8_fast), @"expert_q8_0_gemv_fast"},
        {offsetof(struct dsm_metal, expert_q3k), @"expert_q3k_gemv"},
        {offsetof(struct dsm_metal, expert_q4k), @"expert_q4k_gemv"},
        {offsetof(struct dsm_metal, expert_q5k), @"expert_q5k_gemv"},
        {offsetof(struct dsm_metal, expert_q6k), @"expert_q6k_gemv"},
        {offsetof(struct dsm_metal, expert_swiglu), @"expert_swiglu"},
        {offsetof(struct dsm_metal, moe_add), @"moe_add"},
        {offsetof(struct dsm_metal, rope_batch), @"rope_neox_batch"},
        {offsetof(struct dsm_metal, rope_partial_batch), @"rope_neox_partial_batch"},
        {offsetof(struct dsm_metal, cache_batch), @"cache_write_batch"},
        {offsetof(struct dsm_metal, attention_prefill), @"attention_prefill"},
        {offsetof(struct dsm_metal, attention_prefill_d256), @"attention_prefill_d256"},
        {offsetof(struct dsm_metal, attention_prefill_d128_tiled), @"attention_prefill_d128_tiled"},
        {offsetof(struct dsm_metal, attention_prefill_fa128), @"attention_prefill_fa128"},
        {offsetof(struct dsm_metal, moe_add_batch), @"moe_add_batch"},
        {offsetof(struct dsm_metal, mm_q8), @"mul_mm_q8_0"},
        {offsetof(struct dsm_metal, mm_q2k), @"mul_mm_q2_K"},
        {offsetof(struct dsm_metal, mm_q3k), @"mul_mm_q3_K"},
        {offsetof(struct dsm_metal, mm_q4k), @"mul_mm_q4_K"},
        {offsetof(struct dsm_metal, mm_q5k), @"mul_mm_q5_K"},
        {offsetof(struct dsm_metal, mm_q6k), @"mul_mm_q6_K"},
        {offsetof(struct dsm_metal, mm_id_map0), @"mul_mm_id_map0"},
        {offsetof(struct dsm_metal, mm_id_q8), @"mul_mm_id_q8_0"},
        {offsetof(struct dsm_metal, mm_id_q2k), @"mul_mm_id_q2_K"},
        {offsetof(struct dsm_metal, mm_id_q3k), @"mul_mm_id_q3_K"},
        {offsetof(struct dsm_metal, mm_id_q4k), @"mul_mm_id_q4_K"},
        {offsetof(struct dsm_metal, mm_id_q5k), @"mul_mm_id_q5_K"},
        {offsetof(struct dsm_metal, mm_id_q6k), @"mul_mm_id_q6_K"},
        {offsetof(struct dsm_metal, f32_rows), @"f32_gemv_rows"},
        {offsetof(struct dsm_metal, mm_f32w), @"mul_mm_f32w_f32"},
        {offsetof(struct dsm_metal, shexp_add_batch), @"shexp_add_batch"},
        {offsetof(struct dsm_metal, mmf[0]), @"mul_mm_q8_0_f32"},
        {offsetof(struct dsm_metal, mmf[1]), @"mul_mm_q2_K_f32"},
        {offsetof(struct dsm_metal, mmf[2]), @"mul_mm_q3_K_f32"},
        {offsetof(struct dsm_metal, mmf[3]), @"mul_mm_q4_K_f32"},
        {offsetof(struct dsm_metal, mmf[4]), @"mul_mm_q5_K_f32"},
        {offsetof(struct dsm_metal, mmf[5]), @"mul_mm_q6_K_f32"},
        {offsetof(struct dsm_metal, mmf_id[0]), @"mul_mm_id_q8_0_f32"},
        {offsetof(struct dsm_metal, mmf_id[1]), @"mul_mm_id_q2_K_f32"},
        {offsetof(struct dsm_metal, mmf_id[2]), @"mul_mm_id_q3_K_f32"},
        {offsetof(struct dsm_metal, mmf_id[3]), @"mul_mm_id_q4_K_f32"},
        {offsetof(struct dsm_metal, mmf_id[4]), @"mul_mm_id_q5_K_f32"},
        {offsetof(struct dsm_metal, mmf_id[5]), @"mul_mm_id_q6_K_f32"},
    };
    if (!create_pipelines(s, lib, specs, sizeof(specs) / sizeof(*specs),
                          error, error_size)) {
        goto bad;
    }
    const char *mmf_env = getenv("DSM_MM_F32");
    s->mm_f32 = mmf_env ? strcmp(mmf_env, "0") != 0 : qwen35_arch(m);
    s->mm_f32w_enabled = f32_weight_matmul_enabled();
    const char *fuse_env = getenv("DSM_DECODE_FUSE");
    s->decode_fuse = !fuse_env || strcmp(fuse_env, "0") != 0;
    if (!create_weight_views(s, error, error_size)) goto bad;
    uint32_t ffn=m->is_moe?m->expert_hidden:m->hidden, slots=m->is_moe?m->experts_used:1, context_alloc=(context+63)&~UINT32_C(63);uint64_t kv=(uint64_t)(qwen35_arch(m) ? m->n_attn_layers : m->layers)*context_alloc*m->kv_heads*m->key_dim*4;s->x=buffer(s->device,m->dim*4);s->norm_x=buffer(s->device,m->dim*4);s->q=buffer(s->device,m->q_width*4);s->k=buffer(s->device,(qwen35_arch(m) ? (m->ssm_groups*m->ssm_state > m->kv_heads*m->key_dim ? m->ssm_groups*m->ssm_state : m->kv_heads*m->key_dim) : m->kv_heads*m->key_dim)*4);s->v=buffer(s->device,m->kv_heads*m->value_dim*4);s->att=buffer(s->device,m->attn_width*4);s->tmp=buffer(s->device,m->dim*slots*4);s->gate=buffer(s->device,ffn*slots*4);s->up=buffer(s->device,ffn*slots*4);s->route=m->is_moe?buffer(s->device,m->experts*4):NULL;s->expert_ids=m->is_moe?buffer(s->device,m->experts_used*4):NULL;s->expert_weights=m->is_moe?buffer(s->device,m->experts_used*4):NULL;s->scores=buffer(s->device,(uint64_t)context*m->heads*4);s->keys=buffer(s->device,kv);s->values=buffer(s->device,kv);s->logits=buffer(s->device,m->vocab*4);s->attention_partial=buffer(s->device,(uint64_t)m->heads*((context+63)/64)*(m->key_dim+2)*sizeof(float));s->qg=qwen35_arch(m)?buffer(s->device,(uint64_t)m->q_width*2*4):NULL;s->qkv_mixed=qwen35_arch(m)?buffer(s->device,(m->ssm_inner+2*m->ssm_groups*m->ssm_state)*sizeof(float)):NULL;s->conv_out=qwen35_arch(m)?buffer(s->device,(m->ssm_inner+2*m->ssm_groups*m->ssm_state)*sizeof(float)):NULL;s->z=qwen35_arch(m)?buffer(s->device,m->ssm_inner*4):NULL;s->alpha=qwen35_arch(m)?buffer(s->device,m->ssm_vheads*4):NULL;s->beta=qwen35_arch(m)?buffer(s->device,m->ssm_vheads*4):NULL;s->conv_state=qwen35_arch(m)?buffer(s->device,(uint64_t)m->layers*(m->ssm_inner+2*m->ssm_groups*m->ssm_state)*(m->ssm_conv-1)*4):NULL;s->ssm_state=qwen35_arch(m)?buffer(s->device,(uint64_t)m->layers*m->ssm_vheads*m->ssm_state*m->ssm_state*4):NULL;s->shared_gate=qwen35_arch(m)?buffer(s->device,4):NULL;s->batch_chunk=prefill_chunk_size(m);s->bx=buffer(s->device,(uint64_t)s->batch_chunk*m->dim*4);s->bnorm=buffer(s->device,(uint64_t)s->batch_chunk*m->dim*4);s->bq=buffer(s->device,(uint64_t)s->batch_chunk*m->q_width*4);s->bk=buffer(s->device,(uint64_t)s->batch_chunk*m->kv_heads*m->key_dim*4);s->bv=buffer(s->device,(uint64_t)s->batch_chunk*m->kv_heads*m->value_dim*4);s->batt=buffer(s->device,(uint64_t)s->batch_chunk*m->attn_width*4);s->btmp=buffer(s->device,(uint64_t)s->batch_chunk*m->dim*slots*4);s->bgate=buffer(s->device,(uint64_t)s->batch_chunk*ffn*slots*4);s->bup=buffer(s->device,(uint64_t)s->batch_chunk*ffn*slots*4);s->broute=m->is_moe?buffer(s->device,(uint64_t)s->batch_chunk*m->experts*4):NULL;s->bids=m->is_moe?buffer(s->device,(uint64_t)s->batch_chunk*m->experts_used*4):NULL;s->bw=m->is_moe?buffer(s->device,(uint64_t)s->batch_chunk*m->experts_used*4):NULL;uint32_t max_tiles=m->is_moe?matmul_id_max_tiles(m->experts,m->experts_used,s->batch_chunk):0;s->mm_id_tpe=m->is_moe?buffer(s->device,(uint64_t)(1+3*max_tiles)*sizeof(uint32_t)):NULL;s->mm_id_hids=m->is_moe?buffer(s->device,(uint64_t)(m->experts_used*s->batch_chunk+m->experts)*sizeof(int32_t)):NULL;if(!s->x||!s->norm_x||!s->q||!s->k||!s->v||!s->att||!s->tmp||!s->gate||!s->up||(m->is_moe&&(!s->route||!s->expert_ids||!s->expert_weights))||!s->scores||!s->keys||!s->values||!s->logits||!s->attention_partial||(qwen35_arch(m)&&(!s->qg||!s->qkv_mixed||!s->conv_out||!s->z||!s->alpha||!s->beta||!s->conv_state||!s->ssm_state||!s->shared_gate))||!s->bx||!s->bnorm||!s->bq||!s->bk||!s->bv||!s->batt||!s->btmp||!s->bgate||!s->bup||(m->is_moe&&(!s->broute||!s->bids||!s->bw||!s->mm_id_tpe||!s->mm_id_hids))){set_error(error,error_size,"cannot allocate Metal activations");goto bad;}
    if (qwen35_arch(m)) {
        uint32_t qk = m->ssm_groups * m->ssm_state;
        uint32_t mixed = m->ssm_inner + 2 * qk;
        uint32_t mix = m->ssm_inner > m->attn_width ? m->ssm_inner : m->attn_width;
        uint64_t rows = s->batch_chunk;
        s->hb_qkv = buffer(s->device, rows * mixed * sizeof(float));
        s->hb_conv = buffer(s->device, rows * mixed * sizeof(float));
        s->hb_dq = buffer(s->device, rows * qk * sizeof(float));
        s->hb_dk = buffer(s->device, rows * qk * sizeof(float));
        s->hb_z = buffer(s->device, rows * m->ssm_inner * sizeof(float));
        s->hb_g = buffer(s->device, rows * m->ssm_vheads * sizeof(float));
        s->hb_beta = buffer(s->device, rows * m->ssm_vheads * sizeof(float));
        s->hb_mix = buffer(s->device, rows * mix * sizeof(float));
        s->hb_qg = buffer(s->device, rows * 2 * m->q_width * sizeof(float));
        s->hb_aq = buffer(s->device, rows * m->q_width * sizeof(float));
        s->hb_agate = buffer(s->device, rows * m->q_width * sizeof(float));
        s->hb_ak = buffer(s->device, rows * m->kv_heads * m->key_dim * sizeof(float));
        s->hb_av = buffer(s->device, rows * m->kv_heads * m->value_dim * sizeof(float));
        s->hb_out = buffer(s->device, rows * m->dim * sizeof(float));
        uint32_t ffn_width = m->is_moe ? m->shared_hidden : m->hidden;
        s->hb_shg = m->is_moe ? buffer(s->device, rows * sizeof(float)) : NULL;
        s->hb_sgate = buffer(s->device, rows * ffn_width * sizeof(float));
        s->hb_sup = buffer(s->device, rows * ffn_width * sizeof(float));
        s->hb_sdown = buffer(s->device, rows * m->dim * sizeof(float));
        if (!s->hb_qkv || !s->hb_conv || !s->hb_dq || !s->hb_dk || !s->hb_z ||
            !s->hb_g || !s->hb_beta || !s->hb_mix || !s->hb_qg || !s->hb_aq ||
            !s->hb_agate || !s->hb_ak || !s->hb_av || !s->hb_out ||
            !s->hb_sgate || !s->hb_sup || !s->hb_sdown) {
            set_error(error, error_size, "cannot allocate hybrid Metal activations");
            goto bad;
        }
    }
    *out=s;return 1;
bad:dsm_metal_free(s);return 0;
}
void dsm_metal_free(dsm_metal *s){if(!s)return;if(s->profile.enabled)dsm_metal_profile_print(s);free(s->checkpoints);free(s->views);free(s);}
uint32_t dsm_metal_pos(const dsm_metal *s){return s->pos;}
int dsm_metal_checkpoint(dsm_metal *s) {
    if (!s || !qwen35_arch(s->m)) return s != NULL;
    if (!checkpoint_buffers(s)) return 0;
    for (uint32_t i = 0; i < s->checkpoint_count; ++i)
        if (s->checkpoints[i].valid && s->checkpoints[i].pos == s->pos) return 1;
    double started = CFAbsoluteTimeGetCurrent();
    uint32_t slot = s->checkpoint_next;
    for (uint32_t i = 0; i < s->checkpoint_count; ++i) {
        uint32_t candidate = (s->checkpoint_next + i) % s->checkpoint_count;
        if (!s->checkpoints[candidate].valid) { slot = candidate; break; }
    }
    recurrent_checkpoint *checkpoint = &s->checkpoints[slot];
    id<MTLCommandBuffer> cb = [s->queue commandBuffer];
    id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
    [blit copyFromBuffer:s->conv_state sourceOffset:0 toBuffer:checkpoint->conv_state
      destinationOffset:0 size:s->conv_state.length];
    [blit copyFromBuffer:s->ssm_state sourceOffset:0 toBuffer:checkpoint->ssm_state
      destinationOffset:0 size:s->ssm_state.length];
    [blit endEncoding]; [cb commit]; [cb waitUntilCompleted];
    if (cb.status != MTLCommandBufferStatusCompleted) return 0;
    checkpoint->pos = s->pos;
    checkpoint->valid = 1;
    s->checkpoint_next = (slot + 1) % s->checkpoint_count;
    if (getenv("DSM_CKPT_PROFILE")) fprintf(stderr, "dsm: checkpoint %u %.2f MiB %.3f ms\n",
        s->pos, (s->conv_state.length + s->ssm_state.length) / (1024.0 * 1024.0),
        (CFAbsoluteTimeGetCurrent() - started) * 1000.0);
    return 1;
}
uint32_t dsm_metal_rewind(dsm_metal *s, uint32_t pos) {
    if (!s || pos > s->pos) return UINT32_MAX;
    if (!qwen35_arch(s->m)) { s->pos = pos; return pos; }
    recurrent_checkpoint *best = NULL;
    for (uint32_t i = 0; i < s->checkpoint_count; ++i) {
        recurrent_checkpoint *checkpoint = &s->checkpoints[i];
        if (checkpoint->valid && checkpoint->pos <= pos &&
            (!best || checkpoint->pos > best->pos)) best = checkpoint;
    }
    if (!best) {
        memset(s->conv_state.contents, 0, s->conv_state.length);
        memset(s->ssm_state.contents, 0, s->ssm_state.length);
        s->pos = 0;
    } else {
        id<MTLCommandBuffer> cb = [s->queue commandBuffer];
        id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
        [blit copyFromBuffer:best->conv_state sourceOffset:0 toBuffer:s->conv_state
          destinationOffset:0 size:s->conv_state.length];
        [blit copyFromBuffer:best->ssm_state sourceOffset:0 toBuffer:s->ssm_state
          destinationOffset:0 size:s->ssm_state.length];
        [blit endEncoding]; [cb commit]; [cb waitUntilCompleted];
        if (cb.status != MTLCommandBufferStatusCompleted) return UINT32_MAX;
        s->pos = best->pos;
    }
    invalidate_checkpoints_after(s, s->pos);
    return s->pos;
}
int dsm_metal_truncate(dsm_metal *s,uint32_t pos){return s&&dsm_metal_rewind(s,pos)==pos;}
int dsm_metal_matvec_test(dsm_metal *s,const gguf_tensor *w,const float *input,float *output){if(!s||!w||!dsm_quant_supported(w->type))return 0;id<MTLBuffer> x=buffer(s->device,w->dims[0]*sizeof(float)),y=buffer(s->device,w->dims[1]*sizeof(float));if(!x||!y)return 0;memcpy(x.contents,input,w->dims[0]*sizeof(float));id<MTLCommandBuffer> cb=[s->queue commandBuffer];id<MTLComputeCommandEncoder> e=[cb computeCommandEncoder];encode_gemv(s,e,w,x,y);[e endEncoding];[cb commit];[cb waitUntilCompleted];if(cb.status!=MTLCommandBufferStatusCompleted)return 0;memcpy(output,y.contents,w->dims[1]*sizeof(float));return 1;}
int dsm_metal_matmul_test(dsm_metal *s, const gguf_tensor *w, const float *input,
                          uint32_t rows, float *output) {
    if (!s || !w || !input || !output || !rows || w->ndim != 2 ||
        !matmul_pipeline(s, w->type) || w->dims[0] % 32) return 0;
    id<MTLBuffer> x = buffer(s->device, (uint64_t)rows * w->dims[0] * sizeof(float));
    id<MTLBuffer> y = buffer(s->device, (uint64_t)rows * w->dims[1] * sizeof(float));
    if (!x || !y) return 0;
    memcpy(x.contents, input, (uint64_t)rows * w->dims[0] * sizeof(float));
    id<MTLCommandBuffer> cb = [s->queue commandBuffer];
    id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
    encode_matmul(s, e, w, x, y, rows);
    [e endEncoding]; [cb commit]; [cb waitUntilCompleted];
    if (cb.status != MTLCommandBufferStatusCompleted) return 0;
    memcpy(output, y.contents, (uint64_t)rows * w->dims[1] * sizeof(float));
    return 1;
}
int dsm_metal_expert_matmul_test(dsm_metal *s, const gguf_tensor *w,
                                 const float *input, const uint32_t *ids,
                                 uint32_t rows, uint32_t input_rows,
                                 float *output) {
    if (!s || !w || !input || !ids || !output || !rows || rows > s->batch_chunk ||
        !input_rows || input_rows > s->m->experts_used || w->ndim != 3 ||
        w->dims[2] != s->m->experts || !matmul_id_pipeline(s, w->type)) return 0;
    id<MTLBuffer> x = buffer(s->device, (uint64_t)rows * input_rows * w->dims[0] * sizeof(float));
    id<MTLBuffer> y = buffer(s->device, (uint64_t)rows * s->m->experts_used * w->dims[1] * sizeof(float));
    if (!x || !y) return 0;
    memcpy(x.contents, input, (uint64_t)rows * input_rows * w->dims[0] * sizeof(float));
    memcpy(s->bids.contents, ids, (uint64_t)rows * s->m->experts_used * sizeof(*ids));
    id<MTLCommandBuffer> cb = [s->queue commandBuffer];
    id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
    if (!encode_expert_matmul(s, e, w, x, y, rows, input_rows, (uint32_t)w->dims[0],
                              (uint32_t)w->dims[1], 1)) return 0;
    [e endEncoding]; [cb commit]; [cb waitUntilCompleted];
    if (cb.status != MTLCommandBufferStatusCompleted) return 0;
    memcpy(output, y.contents,
           (uint64_t)rows * s->m->experts_used * w->dims[1] * sizeof(*output));
    return 1;
}
int dsm_metal_expert_matvec_test(dsm_metal *s,const gguf_tensor *w,uint32_t expert,const float *input,float *output){
    if(!s||!w||!expert_gemv_supported(w->type)||w->ndim!=3||expert>=w->dims[2])return 0;
    id<MTLBuffer> wb,x,y,ids;uint64_t off;if(!tensor_view(s,w,&wb,&off))return 0;
    x=buffer(s->device,w->dims[0]*sizeof(float));y=buffer(s->device,w->dims[1]*sizeof(float));ids=buffer(s->device,sizeof(expert));if(!x||!y||!ids)return 0;
    memcpy(x.contents,input,w->dims[0]*sizeof(float));memcpy(ids.contents,&expert,sizeof(expert));
    expert_gemv_params p={(uint32_t)w->dims[0],(uint32_t)w->dims[1],(uint32_t)(w->bytes/w->dims[2]),(uint32_t)w->dims[1],0};
    id<MTLCommandBuffer> cb=[s->queue commandBuffer];id<MTLComputeCommandEncoder> e=[cb computeCommandEncoder];
    [e setComputePipelineState:expert_gemv_pipeline(s,w->type)];[e setBuffer:wb offset:(NSUInteger)off atIndex:0];[e setBuffer:x offset:0 atIndex:1];[e setBuffer:y offset:0 atIndex:2];[e setBytes:&p length:sizeof(p) atIndex:3];[e setBuffer:ids offset:0 atIndex:4];if(w->type==GGML_TYPE_Q8_0){[e setThreadgroupMemoryLength:2*32*sizeof(float) atIndex:0];[e dispatchThreadgroups:MTLSizeMake((p.out+1)/2,1,1) threadsPerThreadgroup:MTLSizeMake(32,4,1)];}else if(w->type==GGML_TYPE_Q3_K||w->type==GGML_TYPE_Q4_K){[e dispatchThreadgroups:MTLSizeMake((p.out+3)/4,1,1) threadsPerThreadgroup:MTLSizeMake(32,2,1)];}else{[e setThreadgroupMemoryLength:256*sizeof(float) atIndex:0];[e dispatchThreadgroups:MTLSizeMake(p.out,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];}
    [e endEncoding];[cb commit];[cb waitUntilCompleted];if(cb.status!=MTLCommandBufferStatusCompleted)return 0;memcpy(output,y.contents,w->dims[1]*sizeof(float));return 1;
}
static float hybrid_test_random(uint32_t *seed) {
    *seed = *seed * 1664525u + 1013904223u;
    return (float)(*seed >> 8) / 8388608.0f - 1.0f;
}
int dsm_metal_hybrid_kernels_test(dsm_metal *s, uint32_t layer, uint32_t rows,
                                  int nonzero_state, float *max_abs,
                                  float *max_rel) {
    const dsm_model *m;
    const dsm_layer *w;
    uint32_t qk_width, mixed_width, history, v_stride, qk_count, gate_count;
    uint64_t conv_state_bytes, ssm_state_bytes;
    id<MTLBuffer> conv_weight, dt_bias, ssm_a;
    uint64_t conv_weight_off, dt_bias_off, ssm_a_off;
    id<MTLBuffer> input, conv_decode, conv_batch, q_decode, k_decode, q_batch,
        k_batch, alpha_decode, beta_decode, alpha_batch, beta_batch, out_decode,
        out_batch, history_decode, history_batch, state_decode, state_batch;
    uint32_t seed = 1;

    if (max_abs) *max_abs = 0.0f;
    if (max_rel) *max_rel = 0.0f;
    if (!s || !max_abs || !max_rel || !rows || !qwen35_arch(s->m) ||
        layer >= s->m->layers || !s->m->is_recurrent[layer]) return 1;
    m = s->m;
    w = &m->layer[layer];
    if (!w->conv1d || !w->dt_bias || !w->ssm_a || !m->ssm_conv ||
        m->ssm_conv < 2 || !m->ssm_state || m->ssm_state > 128 ||
        m->ssm_state % 32 || !m->ssm_groups || !m->ssm_vheads) return 1;
    if (!tensor_view(s, w->conv1d, &conv_weight, &conv_weight_off) ||
        !tensor_view(s, w->dt_bias, &dt_bias, &dt_bias_off) ||
        !tensor_view(s, w->ssm_a, &ssm_a, &ssm_a_off)) return 1;

    qk_width = m->ssm_groups * m->ssm_state;
    mixed_width = m->ssm_inner + 2 * qk_width;
    history = m->ssm_conv - 1;
    v_stride = m->ssm_vheads * m->ssm_state;
    qk_count = rows * qk_width;
    gate_count = rows * m->ssm_vheads;
    conv_state_bytes = (uint64_t)mixed_width * history * sizeof(float);
    ssm_state_bytes = (uint64_t)m->ssm_vheads * m->ssm_state * m->ssm_state * sizeof(float);
    input = buffer(s->device, (uint64_t)rows * mixed_width * sizeof(float));
    conv_decode = buffer(s->device, (uint64_t)rows * mixed_width * sizeof(float));
    conv_batch = buffer(s->device, (uint64_t)rows * mixed_width * sizeof(float));
    q_decode = buffer(s->device, qk_width * sizeof(float));
    k_decode = buffer(s->device, qk_width * sizeof(float));
    q_batch = buffer(s->device, (uint64_t)qk_count * sizeof(float));
    k_batch = buffer(s->device, (uint64_t)qk_count * sizeof(float));
    alpha_decode = buffer(s->device, (uint64_t)gate_count * sizeof(float));
    beta_decode = buffer(s->device, (uint64_t)gate_count * sizeof(float));
    alpha_batch = buffer(s->device, (uint64_t)gate_count * sizeof(float));
    beta_batch = buffer(s->device, (uint64_t)gate_count * sizeof(float));
    out_decode = buffer(s->device, (uint64_t)rows * v_stride * sizeof(float));
    out_batch = buffer(s->device, (uint64_t)rows * v_stride * sizeof(float));
    history_decode = buffer(s->device, conv_state_bytes);
    history_batch = buffer(s->device, conv_state_bytes);
    state_decode = buffer(s->device, ssm_state_bytes);
    state_batch = buffer(s->device, ssm_state_bytes);
    if (!input || !conv_decode || !conv_batch || !q_decode || !k_decode ||
        !q_batch || !k_batch || !alpha_decode || !beta_decode || !alpha_batch ||
        !beta_batch || !out_decode || !out_batch || !history_decode ||
        !history_batch || !state_decode || !state_batch) return 1;

    float *input_data = input.contents;
    float *alpha_data = alpha_decode.contents;
    float *beta_data = beta_decode.contents;
    for (uint64_t i = 0; i < (uint64_t)rows * mixed_width; ++i)
        input_data[i] = 0.1f * hybrid_test_random(&seed);
    for (uint32_t i = 0; i < gate_count; ++i) {
        alpha_data[i] = 0.1f * hybrid_test_random(&seed);
        beta_data[i] = 0.1f * hybrid_test_random(&seed);
    }
    memcpy(alpha_batch.contents, alpha_decode.contents,
           (size_t)gate_count * sizeof(float));
    memcpy(beta_batch.contents, beta_decode.contents,
           (size_t)gate_count * sizeof(float));
    memset(history_decode.contents, 0, (size_t)conv_state_bytes);
    memset(state_decode.contents, 0, (size_t)ssm_state_bytes);
    if (nonzero_state) {
        float *h = history_decode.contents, *st = state_decode.contents;
        for (uint64_t i = 0; i < conv_state_bytes / sizeof(*h); ++i)
            h[i] = 0.01f * hybrid_test_random(&seed);
        for (uint64_t i = 0; i < ssm_state_bytes / sizeof(*st); ++i)
            st[i] = 0.01f * hybrid_test_random(&seed);
    }
    memcpy(history_batch.contents, history_decode.contents, (size_t)conv_state_bytes);
    memcpy(state_batch.contents, state_decode.contents, (size_t)ssm_state_bytes);

    id<MTLCommandBuffer> cb = [s->queue commandBuffer];
    id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
    ssm_conv_params conv = {mixed_width, history};
    norm_params l2 = {m->ssm_state, m->ssm_groups * 2, m->rms_eps};
    gdn_params gdn = {m->ssm_groups, m->ssm_vheads, m->ssm_state, 0, m->rms_eps};
    for (uint32_t row = 0; row < rows; ++row) {
        NSUInteger mixed_off = (NSUInteger)row * mixed_width * sizeof(float);
        NSUInteger gate_off = (NSUInteger)row * m->ssm_vheads * sizeof(float);
        NSUInteger out_off = (NSUInteger)row * v_stride * sizeof(float);
        [e setComputePipelineState:s->ssm_conv];
        [e setBuffer:input offset:mixed_off atIndex:0]; [e setBuffer:history_decode offset:0 atIndex:1];
        [e setBuffer:conv_weight offset:(NSUInteger)conv_weight_off atIndex:2]; [e setBuffer:conv_decode offset:mixed_off atIndex:3];
        [e setBytes:&conv length:sizeof(conv) atIndex:4]; dispatch_1d(e, s->ssm_conv, mixed_width);
        [e setComputePipelineState:s->split_gdn_qk]; [e setBuffer:conv_decode offset:mixed_off atIndex:0];
        [e setBuffer:q_decode offset:0 atIndex:1]; [e setBuffer:k_decode offset:0 atIndex:2];
        [e setBytes:&qk_width length:sizeof(qk_width) atIndex:3]; dispatch_1d(e, s->split_gdn_qk, qk_width);
        [e setComputePipelineState:s->l2norm]; [e setBuffer:q_decode offset:0 atIndex:0]; [e setBytes:&l2 length:sizeof(l2) atIndex:1];
        [e setThreadgroupMemoryLength:256 * sizeof(float) atIndex:0]; [e dispatchThreadgroups:MTLSizeMake(m->ssm_groups, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [e setBuffer:k_decode offset:0 atIndex:0]; [e dispatchThreadgroups:MTLSizeMake(m->ssm_groups, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [e setComputePipelineState:s->gdn_gates]; [e setBuffer:alpha_decode offset:gate_off atIndex:0]; [e setBuffer:beta_decode offset:gate_off atIndex:1];
        [e setBuffer:dt_bias offset:(NSUInteger)dt_bias_off atIndex:2]; [e setBuffer:ssm_a offset:(NSUInteger)ssm_a_off atIndex:3]; dispatch_1d(e, s->gdn_gates, m->ssm_vheads);
        [e setComputePipelineState:s->gdn]; [e setBuffer:q_decode offset:0 atIndex:0]; [e setBuffer:k_decode offset:0 atIndex:1];
        [e setBuffer:conv_decode offset:mixed_off + (NSUInteger)2 * qk_width * sizeof(float) atIndex:2]; [e setBuffer:alpha_decode offset:gate_off atIndex:3]; [e setBuffer:beta_decode offset:gate_off atIndex:4];
        [e setBuffer:state_decode offset:0 atIndex:5]; [e setBuffer:out_decode offset:out_off atIndex:6]; [e setBytes:&gdn length:sizeof(gdn) atIndex:7];
        [e dispatchThreadgroups:MTLSizeMake((m->ssm_state + 3) / 4, m->ssm_vheads, 1) threadsPerThreadgroup:MTLSizeMake(32, 4, 1)];
    }
    ssm_conv_batch_params conv_batch_p = {mixed_width, history, rows};
    gdn_batch_params gdn_batch_p = {m->ssm_groups, m->ssm_vheads, m->ssm_state, 0, rows, qk_width, mixed_width, v_stride};
    [e setComputePipelineState:s->ssm_conv_batch]; [e setBuffer:input offset:0 atIndex:0]; [e setBuffer:history_batch offset:0 atIndex:1];
    [e setBuffer:conv_weight offset:(NSUInteger)conv_weight_off atIndex:2]; [e setBuffer:conv_batch offset:0 atIndex:3]; [e setBytes:&conv_batch_p length:sizeof(conv_batch_p) atIndex:4]; dispatch_1d(e, s->ssm_conv_batch, mixed_width);
    [e setComputePipelineState:s->split_gdn_qk_batch]; [e setBuffer:conv_batch offset:0 atIndex:0]; [e setBuffer:q_batch offset:0 atIndex:1]; [e setBuffer:k_batch offset:0 atIndex:2];
    [e setBytes:&qk_width length:sizeof(qk_width) atIndex:3]; [e setBytes:&mixed_width length:sizeof(mixed_width) atIndex:4]; [e setBytes:&qk_count length:sizeof(qk_count) atIndex:5]; dispatch_1d(e, s->split_gdn_qk_batch, qk_count);
    l2.groups = rows * m->ssm_groups;
    [e setComputePipelineState:s->l2norm]; [e setBuffer:q_batch offset:0 atIndex:0]; [e setBytes:&l2 length:sizeof(l2) atIndex:1]; [e setThreadgroupMemoryLength:256 * sizeof(float) atIndex:0]; [e dispatchThreadgroups:MTLSizeMake(l2.groups, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [e setBuffer:k_batch offset:0 atIndex:0]; [e dispatchThreadgroups:MTLSizeMake(l2.groups, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [e setComputePipelineState:s->gdn_gates_batch]; [e setBuffer:alpha_batch offset:0 atIndex:0]; [e setBuffer:beta_batch offset:0 atIndex:1]; [e setBuffer:dt_bias offset:(NSUInteger)dt_bias_off atIndex:2]; [e setBuffer:ssm_a offset:(NSUInteger)ssm_a_off atIndex:3]; [e setBytes:&m->ssm_vheads length:sizeof(m->ssm_vheads) atIndex:4]; [e setBytes:&gate_count length:sizeof(gate_count) atIndex:5]; dispatch_1d(e, s->gdn_gates_batch, gate_count);
    [e setComputePipelineState:s->gdn_batch]; [e setBuffer:q_batch offset:0 atIndex:0]; [e setBuffer:k_batch offset:0 atIndex:1]; [e setBuffer:conv_batch offset:(NSUInteger)2 * qk_width * sizeof(float) atIndex:2]; [e setBuffer:alpha_batch offset:0 atIndex:3]; [e setBuffer:beta_batch offset:0 atIndex:4]; [e setBuffer:state_batch offset:0 atIndex:5]; [e setBuffer:out_batch offset:0 atIndex:6]; [e setBytes:&gdn_batch_p length:sizeof(gdn_batch_p) atIndex:7]; [e dispatchThreadgroups:MTLSizeMake((m->ssm_state + 3) / 4, m->ssm_vheads, 1) threadsPerThreadgroup:MTLSizeMake(32, 4, 1)];
    [e endEncoding]; [cb commit]; [cb waitUntilCompleted];
    if (cb.status != MTLCommandBufferStatusCompleted) return 1;
    const float *left[] = {out_decode.contents, history_decode.contents, state_decode.contents};
    const float *right[] = {out_batch.contents, history_batch.contents, state_batch.contents};
    const uint64_t counts[] = {(uint64_t)rows * v_stride, conv_state_bytes / sizeof(float), ssm_state_bytes / sizeof(float)};
    for (uint32_t part = 0; part < 3; ++part) for (uint64_t i = 0; i < counts[part]; ++i) {
        float abs = fabsf(left[part][i] - right[part][i]);
        float rel = abs / fmaxf(fabsf(left[part][i]), 1e-6f);
        if (!isfinite(abs) || !isfinite(rel)) { *max_abs = INFINITY; *max_rel = INFINITY; return 0; }
        if (abs > *max_abs) *max_abs = abs;
        if (rel > *max_rel) *max_rel = rel;
    }
    return 0;
}
static int profile_layer(dsm_metal *s, const dsm_layer *w, uint32_t layer, uint32_t kvw) {
    const dsm_model *m=s->m; id<MTLCommandBuffer> cb; id<MTLComputeCommandEncoder> e; double start;
#define PROFILE_ENCODE(category, count, bytes, body) do { \
    e=profile_encoder(s,&cb,&start); body; \
    if(!profile_complete(s,category,cb,e,count,bytes,start))return 0; \
} while(0)
    PROFILE_ENCODE(PROFILE_RMSNORM,1,0,encode_norm(s,e,s->x,w->attn_norm,s->norm_x,m->dim,1));
    PROFILE_ENCODE(PROFILE_QKV_GEMV,3,w->q->bytes+w->k->bytes+w->v->bytes,encode_gemv(s,e,w->q,s->norm_x,s->q);encode_gemv(s,e,w->k,s->norm_x,s->k);encode_gemv(s,e,w->v,s->norm_x,s->v));
    PROFILE_ENCODE(PROFILE_QK_NORM_ROPE,4,0,encode_norm(s,e,s->q,w->q_norm,s->q,m->key_dim,m->heads);encode_norm(s,e,s->k,w->k_norm,s->k,m->key_dim,m->kv_heads);encode_rope(s,e,s->q,m->heads);encode_rope(s,e,s->k,m->kv_heads));
    cache_params cp={layer,s->pos,s->context,kvw};
    PROFILE_ENCODE(PROFILE_KV_STORE,2,0,[e setComputePipelineState:s->cache];[e setBuffer:s->k offset:0 atIndex:0];[e setBuffer:s->keys offset:0 atIndex:1];[e setBytes:&cp length:sizeof(cp) atIndex:2];dispatch_1d(e,s->cache,kvw);[e setBuffer:s->v offset:0 atIndex:0];[e setBuffer:s->values offset:0 atIndex:1];[e setBytes:&cp length:sizeof(cp) atIndex:2];dispatch_1d(e,s->cache,kvw));
    PROFILE_ENCODE(PROFILE_ATTENTION,attention_decode_enabled()?2:1,0,encode_attention_decode(s,e,layer,s->pos));
    PROFILE_ENCODE(PROFILE_O_GEMV,1,w->o->bytes,encode_gemv(s,e,w->o,s->att,s->tmp));
    PROFILE_ENCODE(PROFILE_RESIDUAL,1,0,encode_simple(e,s->add,s->x,s->tmp,m->dim));
    PROFILE_ENCODE(PROFILE_RMSNORM,1,0,encode_norm(s,e,s->x,w->ffn_norm,s->norm_x,m->dim,1));
    if (m->is_moe) {
        PROFILE_ENCODE(PROFILE_MOE_ROUTER, 1, w->gate_inp->bytes, encode_moe_router(s, e, w));
        PROFILE_ENCODE(PROFILE_MOE_TOPK, 1, 0, encode_moe_topk(s, e));
        PROFILE_ENCODE(PROFILE_MOE_EXPERT_GATE_UP, 3, m->experts_used * (w->gate_exps->bytes + w->up_exps->bytes) / m->experts, encode_moe_gate_up(s, e, w));
        PROFILE_ENCODE(PROFILE_MOE_EXPERT_DOWN, 1, m->experts_used * w->down_exps->bytes / m->experts, encode_moe_down(s, e, w));
        PROFILE_ENCODE(PROFILE_MOE_ACCUMULATE, 1, 0, encode_moe_accumulate(s, e));
    } else {
        PROFILE_ENCODE(PROFILE_FFN_GATE_UP_GEMV,2,w->gate->bytes+w->up->bytes,encode_gemv(s,e,w->gate,s->norm_x,s->gate);encode_gemv(s,e,w->up,s->norm_x,s->up));
        PROFILE_ENCODE(PROFILE_SWIGLU,1,0,encode_simple(e,s->swiglu,s->gate,s->up,m->hidden));
        PROFILE_ENCODE(PROFILE_FFN_DOWN_GEMV,1,w->down->bytes,encode_gemv(s,e,w->down,s->gate,s->tmp));
        PROFILE_ENCODE(PROFILE_RESIDUAL,1,0,encode_simple(e,s->add,s->x,s->tmp,m->dim));
    }
#undef PROFILE_ENCODE
    return 1;
}
static int profile_token(dsm_metal *s, float *logits) {
    const dsm_model *m=s->m; uint32_t kvw=m->kv_heads*m->key_dim;
    for(uint32_t l=0;l<m->layers;++l)if(!profile_layer(s,&m->layer[l],l,kvw))return 0;
    id<MTLCommandBuffer> cb; double start; id<MTLComputeCommandEncoder> e=profile_encoder(s,&cb,&start);
    encode_norm(s,e,s->x,m->output_norm,s->norm_x,m->dim,1);
    if(!profile_complete(s,PROFILE_RMSNORM,cb,e,1,0,start))return 0;
    e=profile_encoder(s,&cb,&start);encode_gemv(s,e,m->output,s->norm_x,s->logits);
    if(!profile_complete(s,PROFILE_OUTPUT_GEMV,cb,e,1,m->output->bytes,start))return 0;
    memcpy(logits,s->logits.contents,m->vocab*4);++s->pos;++s->profile.tokens;return 1;
}
static void encode_kv_store(dsm_metal *s, id<MTLComputeCommandEncoder> e,
                            uint32_t layer, uint32_t pos) {
    uint32_t kvw = s->m->kv_heads * s->m->key_dim;
    cache_params cp = {layer, pos, s->context, kvw};
    [e setComputePipelineState:s->cache];
    [e setBuffer:s->k offset:0 atIndex:0];
    [e setBuffer:s->keys offset:0 atIndex:1];
    [e setBytes:&cp length:sizeof(cp) atIndex:2];
    dispatch_1d(e, s->cache, kvw);
    [e setBuffer:s->v offset:0 atIndex:0];
    [e setBuffer:s->values offset:0 atIndex:1];
    [e setBytes:&cp length:sizeof(cp) atIndex:2];
    dispatch_1d(e, s->cache, kvw);
}
static void encode_v_store(dsm_metal *s, id<MTLComputeCommandEncoder> e,
                           uint32_t layer, uint32_t pos) {
    uint32_t kvw = s->m->kv_heads * s->m->key_dim;
    cache_params cp = {layer, pos, s->context, kvw};
    [e setComputePipelineState:s->cache];
    [e setBuffer:s->v offset:0 atIndex:0];
    [e setBuffer:s->values offset:0 atIndex:1];
    [e setBytes:&cp length:sizeof(cp) atIndex:2];
    dispatch_1d(e, s->cache, kvw);
}
static void encode_attention_decode_at(dsm_metal *s,
                                       id<MTLComputeCommandEncoder> e,
                                       uint32_t layer, uint32_t pos,
                                       id<MTLBuffer> q, NSUInteger q_off,
                                       id<MTLBuffer> out, NSUInteger out_off) {
    const dsm_model *m = s->m;
    attn_params ap = {m->heads, m->kv_heads, m->key_dim, pos, s->context,
                       layer};
    if (attention_decode_enabled()) {
        uint32_t splits = (pos + 64) / 64;
        attn_split_params split = {m->heads, m->kv_heads, m->key_dim, pos,
                                   s->context, layer, splits};
        [e setComputePipelineState:s->attention_split];
        [e setBuffer:q offset:q_off atIndex:0];
        [e setBuffer:s->keys offset:0 atIndex:1];
        [e setBuffer:s->values offset:0 atIndex:2];
        [e setBuffer:s->attention_partial offset:0 atIndex:3];
        [e setBytes:&split length:sizeof(split) atIndex:4];
        [e setThreadgroupMemoryLength:(64 + 256) * sizeof(float) atIndex:0];
        [e dispatchThreadgroups:MTLSizeMake(splits, m->heads, 1)
 threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [e setComputePipelineState:s->attention_merge];
        [e setBuffer:s->attention_partial offset:0 atIndex:0];
        [e setBuffer:out offset:out_off atIndex:1];
        [e setBytes:&split length:sizeof(split) atIndex:2];
        [e setThreadgroupMemoryLength:256 * sizeof(float) atIndex:0];
        [e dispatchThreadgroups:MTLSizeMake(m->heads, 1, 1)
 threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        return;
    }
    [e setComputePipelineState:s->attention];
    [e setBuffer:q offset:q_off atIndex:0];
    [e setBuffer:s->keys offset:0 atIndex:1];
    [e setBuffer:s->values offset:0 atIndex:2];
    [e setBuffer:out offset:out_off atIndex:3];
    [e setBuffer:s->scores offset:0 atIndex:4];
    [e setBytes:&ap length:sizeof(ap) atIndex:5];
    [e setThreadgroupMemoryLength:256 * sizeof(float) atIndex:0];
    [e dispatchThreadgroups:MTLSizeMake(m->heads, 1, 1)
 threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
}
static void encode_attention_decode(dsm_metal *s,
                                    id<MTLComputeCommandEncoder> e,
                                    uint32_t layer, uint32_t pos) {
    encode_attention_decode_at(s, e, layer, pos, s->q, 0, s->att, 0);
}
static void encode_partial_rope(dsm_metal *s, id<MTLComputeCommandEncoder> e,
                                id<MTLBuffer> x, uint32_t heads) {
    rope_partial_params p = {heads, s->m->key_dim, s->m->rope_dims, s->pos,
                             s->m->rope_base};
    [e setComputePipelineState:s->rope_partial];
    [e setBuffer:x offset:0 atIndex:0];
    [e setBytes:&p length:sizeof(p) atIndex:1];
    dispatch_1d(e, s->rope_partial, heads * (s->m->rope_dims / 2));
}
static void encode_hybrid_moe(dsm_metal *s, id<MTLComputeCommandEncoder> e,
                              const dsm_layer *w) {
    const dsm_model *m = s->m;
    encode_moe(s, e, w);
    encode_gemv(s, e, w->gate_shexp, s->norm_x, s->gate);
    encode_gemv(s, e, w->up_shexp, s->norm_x, s->up);
    encode_simple(e, s->swiglu, s->gate, s->up, m->shared_hidden);
    encode_gemv(s, e, w->down_shexp, s->gate, s->tmp);
    encode_gemv(s, e, w->gate_inp_shexp, s->norm_x, s->shared_gate);
    [e setComputePipelineState:s->shexp_add];
    [e setBuffer:s->x offset:0 atIndex:0];
    [e setBuffer:s->tmp offset:0 atIndex:1];
    [e setBuffer:s->shared_gate offset:0 atIndex:2];
    [e setBytes:&m->dim length:sizeof(m->dim) atIndex:3];
    dispatch_1d(e, s->shexp_add, m->dim);
}
static void encode_hybrid_dense_ffn(dsm_metal *s,
                                    id<MTLComputeCommandEncoder> e,
                                    const dsm_layer *w) {
    const dsm_model *m = s->m;
    encode_gemv(s, e, w->gate, s->norm_x, s->gate);
    encode_gemv(s, e, w->up, s->norm_x, s->up);
    encode_simple(e, s->swiglu, s->gate, s->up, m->hidden);
    encode_gemv(s, e, w->down, s->gate, s->tmp);
    encode_simple(e, s->add, s->x, s->tmp, m->dim);
}
static void encode_layer_hybrid(dsm_metal *s, id<MTLComputeCommandEncoder> e,
                                uint32_t layer) {
    const dsm_model *m = s->m;
    const dsm_layer *w = &m->layer[layer];
    encode_norm(s, e, s->x, w->attn_norm, s->norm_x, m->dim, 1);
    if (m->is_recurrent[layer]) {
        uint32_t qk_width = m->ssm_groups * m->ssm_state;
        uint32_t mixed_width = m->ssm_inner + 2 * qk_width;
        encode_gemv(s, e, w->qkv, s->norm_x, s->qkv_mixed);
        encode_gemv(s, e, w->z, s->norm_x, s->z);
        encode_gemv(s, e, w->alpha, s->norm_x, s->alpha);
        encode_gemv(s, e, w->beta, s->norm_x, s->beta);
        ssm_conv_params conv = {mixed_width, m->ssm_conv - 1};
        [e setComputePipelineState:s->ssm_conv];
        set_checked(e, s->qkv_mixed, 0, (NSUInteger)mixed_width * sizeof(float), 0);
        set_checked(e, s->conv_state,
                    (NSUInteger)layer * mixed_width * (m->ssm_conv - 1) * sizeof(float),
                    (NSUInteger)mixed_width * (m->ssm_conv - 1) * sizeof(float), 1);
        id<MTLBuffer> wb; uint64_t off;
        if (!tensor_view(s, w->conv1d, &wb, &off)) return;
        set_checked(e, wb, (NSUInteger)off, (NSUInteger)w->conv1d->bytes, 2);
        set_checked(e, s->conv_out, 0, (NSUInteger)mixed_width * sizeof(float), 3);
        [e setBytes:&conv length:sizeof(conv) atIndex:4];
        dispatch_1d(e, s->ssm_conv, mixed_width);
        [e setComputePipelineState:s->split_gdn_qk];
        set_checked(e, s->conv_out, 0, (NSUInteger)mixed_width * sizeof(float), 0);
        set_checked(e, s->q, 0, (NSUInteger)qk_width * sizeof(float), 1);
        set_checked(e, s->k, 0, (NSUInteger)qk_width * sizeof(float), 2);
        [e setBytes:&qk_width length:sizeof(qk_width) atIndex:3];
        dispatch_1d(e, s->split_gdn_qk, qk_width);
        norm_params l2 = {m->ssm_state, m->ssm_groups * 2, m->rms_eps};
        [e setComputePipelineState:s->l2norm];
        set_checked(e, s->q, 0, (NSUInteger)qk_width * sizeof(float), 0); [e setBytes:&l2 length:sizeof(l2) atIndex:1];
        [e setThreadgroupMemoryLength:256 * sizeof(float) atIndex:0];
        [e dispatchThreadgroups:MTLSizeMake(m->ssm_groups, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [e setComputePipelineState:s->l2norm];
        set_checked(e, s->k, 0, (NSUInteger)qk_width * sizeof(float), 0); [e setBytes:&l2 length:sizeof(l2) atIndex:1];
        [e setThreadgroupMemoryLength:256 * sizeof(float) atIndex:0];
        [e dispatchThreadgroups:MTLSizeMake(m->ssm_groups, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        if (!tensor_view(s, w->dt_bias, &wb, &off)) return;
        [e setComputePipelineState:s->gdn_gates];
        [e setBuffer:s->alpha offset:0 atIndex:0]; [e setBuffer:s->beta offset:0 atIndex:1];
        [e setBuffer:wb offset:(NSUInteger)off atIndex:2];
        if (!tensor_view(s, w->ssm_a, &wb, &off)) return;
        [e setBuffer:wb offset:(NSUInteger)off atIndex:3];
        dispatch_1d(e, s->gdn_gates, m->ssm_vheads);
        gdn_params gdn = {m->ssm_groups, m->ssm_vheads, m->ssm_state, layer, m->rms_eps};
        [e setComputePipelineState:s->gdn];
        [e setBuffer:s->q offset:0 atIndex:0]; [e setBuffer:s->k offset:0 atIndex:1];
        [e setBuffer:s->conv_out offset:(NSUInteger)(2 * qk_width) * sizeof(float) atIndex:2];
        [e setBuffer:s->alpha offset:0 atIndex:3]; [e setBuffer:s->beta offset:0 atIndex:4];
        [e setBuffer:s->ssm_state offset:0 atIndex:5]; [e setBuffer:s->att offset:0 atIndex:6];
        [e setBytes:&gdn length:sizeof(gdn) atIndex:7];
        [e dispatchThreadgroups:MTLSizeMake((m->ssm_state + 3) / 4, m->ssm_vheads, 1) threadsPerThreadgroup:MTLSizeMake(32, 4, 1)];
        norm_params gated = {m->ssm_state, m->ssm_vheads, m->rms_eps};
        if (!tensor_view(s, w->ssm_norm, &wb, &off)) return;
        [e setComputePipelineState:s->gated_rmsnorm];
        [e setBuffer:s->att offset:0 atIndex:0]; [e setBuffer:s->z offset:0 atIndex:1];
        [e setBuffer:wb offset:(NSUInteger)off atIndex:2]; [e setBytes:&gated length:sizeof(gated) atIndex:3];
        [e setThreadgroupMemoryLength:256 * sizeof(float) atIndex:0];
        [e dispatchThreadgroups:MTLSizeMake(m->ssm_vheads, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        encode_gemv(s, e, w->ssm_out, s->att, s->tmp);
    } else {
        encode_gemv(s, e, w->q, s->norm_x, s->qg);
        encode_gemv(s, e, w->k, s->norm_x, s->k);
        encode_gemv(s, e, w->v, s->norm_x, s->v);
        [e setComputePipelineState:s->split_q_gate];
        [e setBuffer:s->qg offset:0 atIndex:0]; [e setBuffer:s->q offset:0 atIndex:1];
        [e setBuffer:s->gate offset:0 atIndex:2]; [e setBytes:&m->key_dim length:sizeof(m->key_dim) atIndex:3];
        dispatch_1d(e, s->split_q_gate, m->q_width);
        encode_norm(s, e, s->q, w->q_norm, s->q, m->key_dim, m->heads);
        encode_norm(s, e, s->k, w->k_norm, s->k, m->key_dim, m->kv_heads);
        encode_partial_rope(s, e, s->q, m->heads);
        encode_partial_rope(s, e, s->k, m->kv_heads);
        encode_kv_store(s, e, m->attn_index[layer], s->pos);
        encode_attention_decode(s, e, m->attn_index[layer], s->pos);
        [e setComputePipelineState:s->sigmoid_mul];
        [e setBuffer:s->att offset:0 atIndex:0]; [e setBuffer:s->gate offset:0 atIndex:1];
        [e setBytes:&m->attn_width length:sizeof(m->attn_width) atIndex:2];
        dispatch_1d(e, s->sigmoid_mul, m->attn_width);
        encode_gemv(s, e, w->o, s->att, s->tmp);
    }
    encode_simple(e, s->add, s->x, s->tmp, m->dim);
    encode_norm(s, e, s->x, w->ffn_norm, s->norm_x, m->dim, 1);
    if (m->is_moe) encode_hybrid_moe(s, e, w);
    else encode_hybrid_dense_ffn(s, e, w);
}
static void encode_layer(dsm_metal *s, id<MTLComputeCommandEncoder> e,
                         uint32_t layer) {
    const dsm_model *m = s->m;
    const dsm_layer *w = &m->layer[layer];
    encode_norm(s, e, s->x, w->attn_norm, s->norm_x, m->dim, 1);
    encode_gemv(s, e, w->q, s->norm_x, s->q);
    encode_gemv(s, e, w->k, s->norm_x, s->k);
    encode_gemv(s, e, w->v, s->norm_x, s->v);
    if (s->decode_fuse) {
        encode_qk_norm_rope(s, e, w, layer);
        encode_v_store(s, e, layer, s->pos);
    } else {
        encode_norm(s, e, s->q, w->q_norm, s->q, m->key_dim, m->heads);
        encode_norm(s, e, s->k, w->k_norm, s->k, m->key_dim, m->kv_heads);
        encode_rope(s, e, s->q, m->heads);
        encode_rope(s, e, s->k, m->kv_heads);
        encode_kv_store(s, e, layer, s->pos);
    }
    encode_attention_decode(s, e, layer, s->pos);
    encode_gemv(s, e, w->o, s->att, s->tmp);
    if (s->decode_fuse) encode_add_rmsnorm(s, e, w->ffn_norm);
    else {
        encode_simple(e, s->add, s->x, s->tmp, m->dim);
        encode_norm(s, e, s->x, w->ffn_norm, s->norm_x, m->dim, 1);
    }
    if (m->is_moe) {
        encode_moe(s, e, w);
    } else {
        encode_gemv(s, e, w->gate, s->norm_x, s->gate);
        encode_gemv(s, e, w->up, s->norm_x, s->up);
        encode_simple(e, s->swiglu, s->gate, s->up, m->hidden);
        encode_gemv(s, e, w->down, s->gate, s->tmp);
        if (s->decode_fuse) {
            const gguf_tensor *weight = layer + 1 < m->layers
                    ? m->layer[layer + 1].attn_norm : m->output_norm;
            encode_add_rmsnorm(s, e, weight);
        } else {
            encode_simple(e, s->add, s->x, s->tmp, m->dim);
        }
    }
}
static int mm_type_index(enum ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q8_0: return 0;
        case GGML_TYPE_Q2_K: return 1;
        case GGML_TYPE_Q3_K: return 2;
        case GGML_TYPE_Q4_K: return 3;
        case GGML_TYPE_Q5_K: return 4;
        case GGML_TYPE_Q6_K: return 5;
        default: return -1;
    }
}
/* FP32 staging needs 2048 + 1024 floats of threadgroup memory; half needs 8 KiB
   (also the size of the partial-tile output staging). */
static NSUInteger matmul_tg_bytes(const dsm_metal *s) {
    return s->mm_f32 || s->mm_f32w_enabled ? 12288 : 8192;
}
static id<MTLComputePipelineState> matmul_pipeline(const dsm_metal *s, enum ggml_type type) {
    if (type == GGML_TYPE_F32) return s->mm_f32w_enabled ? s->mm_f32w : nil;
    if (s->mm_f32) { int i = mm_type_index(type); return i < 0 ? nil : s->mmf[i]; }
    switch (type) {
        case GGML_TYPE_Q8_0: return s->mm_q8;
        case GGML_TYPE_Q2_K: return s->mm_q2k;
        case GGML_TYPE_Q3_K: return s->mm_q3k;
        case GGML_TYPE_Q4_K: return s->mm_q4k;
        case GGML_TYPE_Q5_K: return s->mm_q5k;
        case GGML_TYPE_Q6_K: return s->mm_q6k;
        default: return nil;
    }
}
static id<MTLComputePipelineState> matmul_id_pipeline(const dsm_metal *s, enum ggml_type type) {
    if (s->mm_f32) { int i = mm_type_index(type); return i < 0 ? nil : s->mmf_id[i]; }
    switch (type) {
        case GGML_TYPE_Q8_0: return s->mm_id_q8;
        case GGML_TYPE_Q2_K: return s->mm_id_q2k;
        case GGML_TYPE_Q3_K: return s->mm_id_q3k;
        case GGML_TYPE_Q4_K: return s->mm_id_q4k;
        case GGML_TYPE_Q5_K: return s->mm_id_q5k;
        case GGML_TYPE_Q6_K: return s->mm_id_q6k;
        default: return nil;
    }
}
static uint32_t matmul_id_max_tiles(uint32_t experts, uint32_t slots, uint32_t rows) {
    uint32_t entries = slots * rows;
    uint32_t tiles = (entries + 31) / 32 + (experts < entries ? experts : entries) - 1;
    return entries < tiles ? entries : tiles;
}
static int encode_expert_matmul(dsm_metal *s, id<MTLComputeCommandEncoder> e,
                                const gguf_tensor *w, id<MTLBuffer> x,
                                id<MTLBuffer> y, uint32_t rows,
                                uint32_t input_rows, uint32_t input_stride,
                                uint32_t output_stride, int build_map) {
    const dsm_model *m = s->m;
    (void)output_stride;
    id<MTLComputePipelineState> p = rows > 8 && w->dims[0] % 32 == 0
                                         ? matmul_id_pipeline(s, w->type) : nil;
    if (!p) return 0;
    id<MTLBuffer> wb;
    uint64_t off;
    if (!tensor_view(s, w, &wb, &off)) return 0;
    mul_mm_id_map0_params map = {
        .ne10 = (int32_t)m->experts, .ne11 = (int32_t)input_rows,
        .nb11 = input_stride * sizeof(float),
        .nb12 = (uint64_t)input_rows * input_stride * sizeof(float),
        .ne21 = (int32_t)rows, .ne20 = (int32_t)m->experts_used,
        .nb21 = m->experts_used * sizeof(uint32_t),
    };
    /* The token->expert tile map depends only on the routing ids, so callers
       build it once per layer and reuse it for gate, up and down. */
    if (!build_map) goto skip_map0;
    [e setComputePipelineState:s->mm_id_map0];
    [e setBytes:&map length:sizeof(map) atIndex:0];
    [e setBuffer:s->bids offset:0 atIndex:1];
    [e setBuffer:s->mm_id_tpe offset:0 atIndex:2];
    [e setBuffer:s->mm_id_hids offset:0 atIndex:3];
    [e setThreadgroupMemoryLength:(NSUInteger)(2 * m->experts * sizeof(uint32_t)) atIndex:0];
    [e dispatchThreadgroups:MTLSizeMake(1, 1, 1)
 threadsPerThreadgroup:MTLSizeMake(m->experts, 1, 1)];
skip_map0:;
    mul_mm_id_params args = {
        .ne00 = (int32_t)w->dims[0], .ne02 = (int32_t)m->experts,
        .nb01 = w->bytes / w->dims[1] / m->experts,
        .nb02 = w->bytes / m->experts, .nb03 = 0,
        .ne11 = (int32_t)input_rows,
        .nb10 = sizeof(float), .nb11 = input_stride * sizeof(float),
        .nb12 = (uint64_t)input_rows * input_stride * sizeof(float), .nb13 = 0,
        .ne20 = (int32_t)m->experts_used,
        .ne0 = (int32_t)w->dims[1], .ne1 = (int32_t)m->experts_used,
        .r2 = 1, .r3 = 1,
    };
    [e setComputePipelineState:p];
    [e setBytes:&args length:sizeof(args) atIndex:0];
    [e setBuffer:wb offset:(NSUInteger)off atIndex:1];
    [e setBuffer:x offset:0 atIndex:2];
    [e setBuffer:s->mm_id_tpe offset:0 atIndex:3];
    [e setBuffer:s->mm_id_hids offset:0 atIndex:4];
    [e setBuffer:y offset:0 atIndex:5];
    [e setThreadgroupMemoryLength:matmul_tg_bytes(s) atIndex:0];
    [e dispatchThreadgroups:MTLSizeMake(matmul_id_max_tiles(m->experts, m->experts_used, rows),
                                         ((uint32_t)w->dims[1] + 63) / 64, 1)
 threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    return 1;
}
static void encode_matmul(dsm_metal *s, id<MTLComputeCommandEncoder> e,
                          const gguf_tensor *w, id<MTLBuffer> x,
                          id<MTLBuffer> y, uint32_t rows) {
    id<MTLComputePipelineState> p = rows > 8 && w->ndim == 2 && w->dims[0] % 32 == 0
                                        ? matmul_pipeline(s, w->type) : nil;
    if (p) {
        assert(w->dims[0] % 32 == 0);
        id<MTLBuffer> wb;
        uint64_t off;
        if (!tensor_view(s, w, &wb, &off)) return;
        mul_mm_params args = {
            .ne00 = (int32_t)w->dims[0], .ne02 = 1,
            .nb01 = w->bytes / w->dims[1], .nb02 = 0, .nb03 = 0,
            .ne12 = 1,
            .nb10 = sizeof(float), .nb11 = w->dims[0] * sizeof(float),
            .nb12 = 0, .nb13 = 0,
            .ne0 = (int32_t)w->dims[1], .ne1 = (int32_t)rows,
            .r2 = 1, .r3 = 1,
        };
        [e setComputePipelineState:p];
        [e setBytes:&args length:sizeof(args) atIndex:0];
        [e setBuffer:wb offset:(NSUInteger)off atIndex:1];
        [e setBuffer:x offset:0 atIndex:2];
        [e setBuffer:y offset:0 atIndex:3];
        [e setThreadgroupMemoryLength:matmul_tg_bytes(s) atIndex:0];
        [e dispatchThreadgroups:MTLSizeMake((rows + 31) / 32,
                                             ((uint32_t)w->dims[1] + 63) / 64, 1)
 threadsPerThreadgroup:MTLSizeMake(32, 4, 1)];
        return;
    }
    for (uint32_t row = 0; row < rows; ++row) {
        encode_gemv_at(s, e, w, x, (NSUInteger)row * w->dims[0] * sizeof(float), y,
                       (NSUInteger)row * w->dims[1] * sizeof(float));
    }
}
static void encode_layer_batch(dsm_metal *s, id<MTLComputeCommandEncoder> e,
                               uint32_t layer, uint32_t pos0, uint32_t rows) {
    const dsm_model *m = s->m;
    const dsm_layer *w = &m->layer[layer];
    uint32_t kvw = m->kv_heads * m->key_dim;
    encode_norm_at(s, e, s->bx, 0, w->attn_norm, s->bnorm, 0, m->dim, rows);
    encode_matmul(s, e, w->q, s->bnorm, s->bq, rows);
    encode_matmul(s, e, w->k, s->bnorm, s->bk, rows);
    encode_matmul(s, e, w->v, s->bnorm, s->bv, rows);
    encode_norm_at(s, e, s->bq, 0, w->q_norm, s->bq, 0, m->key_dim, rows * m->heads);
    encode_norm_at(s, e, s->bk, 0, w->k_norm, s->bk, 0, m->key_dim, rows * m->kv_heads);
    rope_batch_params rope = {m->heads, m->key_dim, pos0, rows, m->rope_base};
    [e setComputePipelineState:s->rope_batch]; [e setBuffer:s->bq offset:0 atIndex:0]; [e setBytes:&rope length:sizeof(rope) atIndex:1]; dispatch_1d(e, s->rope_batch, rows * m->heads * (m->key_dim / 2));
    rope.heads = m->kv_heads;
    [e setComputePipelineState:s->rope_batch]; [e setBuffer:s->bk offset:0 atIndex:0]; [e setBytes:&rope length:sizeof(rope) atIndex:1]; dispatch_1d(e, s->rope_batch, rows * m->kv_heads * (m->key_dim / 2));
    cache_batch_params cache = {layer, pos0, s->context, kvw, rows};
    [e setComputePipelineState:s->cache_batch]; [e setBuffer:s->bk offset:0 atIndex:0]; [e setBuffer:s->keys offset:0 atIndex:1]; [e setBytes:&cache length:sizeof(cache) atIndex:2]; dispatch_1d(e, s->cache_batch, rows * kvw);
    [e setComputePipelineState:s->cache_batch]; [e setBuffer:s->bv offset:0 atIndex:0]; [e setBuffer:s->values offset:0 atIndex:1]; [e setBytes:&cache length:sizeof(cache) atIndex:2]; dispatch_1d(e, s->cache_batch, rows * kvw);
    attn_prefill_params attention = {m->heads, m->kv_heads, m->key_dim, pos0,
                                     s->context, layer, rows};
    uint32_t q_per_kv = m->heads / m->kv_heads;
    id<MTLComputePipelineState> attention_pipeline =
        rows > 8 && attention_prefill_enabled() && m->key_dim == 128 && q_per_kv <= 8
            ? (attention_flash_enabled() ? s->attention_prefill_fa128
                                         : s->attention_prefill_d128_tiled)
            : s->attention_prefill;
    [e setComputePipelineState:attention_pipeline];
    [e setBuffer:s->bq offset:0 atIndex:0];
    [e setBuffer:s->keys offset:0 atIndex:1];
    [e setBuffer:s->values offset:0 atIndex:2];
    [e setBuffer:s->batt offset:0 atIndex:3];
    [e setBytes:&attention length:sizeof(attention) atIndex:4];
    NSUInteger attention_tgmem = attention_pipeline == s->attention_prefill_d128_tiled
        ? 32768 : (attention_pipeline == s->attention_prefill_fa128
            ? 12288 : (NSUInteger)(2 * 16 * 128 + 4 * q_per_kv * 128) * sizeof(float));
    [e setThreadgroupMemoryLength:attention_tgmem atIndex:0];
    if (attention_pipeline == s->attention_prefill_fa128) {
        /* 8 query slots = (8 / G) rows x G heads sharing one KV head */
        uint32_t group = 8 % q_per_kv == 0 ? q_per_kv : 1, group_rows = 8 / group;
        [e dispatchThreadgroups:MTLSizeMake((rows + group_rows - 1) / group_rows, m->heads / group, 1)
         threadsPerThreadgroup:MTLSizeMake(32, 4, 1)];
    } else {
        [e dispatchThreadgroups:MTLSizeMake((rows + 3) / 4, m->kv_heads, 1)
         threadsPerThreadgroup:MTLSizeMake(32, q_per_kv, 1)];
    }
    encode_matmul(s, e, w->o, s->batt, s->btmp, rows);
    [e setComputePipelineState:s->add]; [e setBuffer:s->bx offset:0 atIndex:0]; [e setBuffer:s->btmp offset:0 atIndex:1]; { uint32_t n = rows * m->dim; [e setBytes:&n length:sizeof(n) atIndex:2]; dispatch_1d(e, s->add, n); }
    encode_norm_at(s, e, s->bx, 0, w->ffn_norm, s->bnorm, 0, m->dim, rows);
    if (!m->is_moe) {
        encode_matmul(s, e, w->gate, s->bnorm, s->bgate, rows);
        encode_matmul(s, e, w->up, s->bnorm, s->bup, rows);
        [e setComputePipelineState:s->swiglu]; [e setBuffer:s->bgate offset:0 atIndex:0]; [e setBuffer:s->bup offset:0 atIndex:1]; { uint32_t n = rows * m->hidden; [e setBytes:&n length:sizeof(n) atIndex:2]; dispatch_1d(e, s->swiglu, n); }
        encode_matmul(s, e, w->down, s->bgate, s->btmp, rows);
        [e setComputePipelineState:s->add]; [e setBuffer:s->bx offset:0 atIndex:0]; [e setBuffer:s->btmp offset:0 atIndex:1]; { uint32_t n = rows * m->dim; [e setBytes:&n length:sizeof(n) atIndex:2]; dispatch_1d(e, s->add, n); }
    } else {
        encode_matmul(s, e, w->gate_inp, s->bnorm, s->broute, rows);
        encode_moe_topk_batch(s, e, rows);
        int use_mm_id = mul_mm_id_enabled() && rows > 8 &&
                        encode_expert_matmul(s, e, w->gate_exps, s->bnorm, s->bgate,
                                             rows, 1, m->dim, m->expert_hidden, 1) &&
                        encode_expert_matmul(s, e, w->up_exps, s->bnorm, s->bup,
                                             rows, 1, m->dim, m->expert_hidden, 0);
        if (use_mm_id) {
            [e setComputePipelineState:s->expert_swiglu]; [e setBuffer:s->bgate offset:0 atIndex:0]; [e setBuffer:s->bup offset:0 atIndex:1]; [e setBytes:&m->expert_hidden length:sizeof(m->expert_hidden) atIndex:2]; [e setBytes:&m->experts_used length:sizeof(m->experts_used) atIndex:3]; [e setBytes:&rows length:sizeof(rows) atIndex:4]; dispatch_1d(e, s->expert_swiglu, rows * m->expert_hidden * m->experts_used);
            use_mm_id = encode_expert_matmul(s, e, w->down_exps, s->bgate, s->btmp,
                                              rows, m->experts_used, m->expert_hidden, m->dim, 0);
        }
        if (!use_mm_id) {
            for (uint32_t row = 0; row < rows; ++row) {
                NSUInteger x_off = (NSUInteger)row * m->dim * sizeof(float);
                NSUInteger gate_off = (NSUInteger)row * m->experts_used * m->expert_hidden * sizeof(float);
                NSUInteger tmp_off = (NSUInteger)row * m->experts_used * m->dim * sizeof(float);
                encode_expert_gemv_batch_row(s, e, w->gate_exps, s->bnorm, x_off, s->bgate, gate_off, row, 0, m->expert_hidden);
                encode_expert_gemv_batch_row(s, e, w->up_exps, s->bnorm, x_off, s->bup, gate_off, row, 0, m->expert_hidden);
                uint32_t single_row = 1;
                [e setComputePipelineState:s->expert_swiglu]; [e setBuffer:s->bgate offset:gate_off atIndex:0]; [e setBuffer:s->bup offset:gate_off atIndex:1]; [e setBytes:&m->expert_hidden length:sizeof(m->expert_hidden) atIndex:2]; [e setBytes:&m->experts_used length:sizeof(m->experts_used) atIndex:3]; [e setBytes:&single_row length:sizeof(single_row) atIndex:4]; dispatch_1d(e, s->expert_swiglu, m->expert_hidden * m->experts_used);
                encode_expert_gemv_batch_row(s, e, w->down_exps, s->bgate, gate_off, s->btmp, tmp_off, row, m->expert_hidden, m->dim);
            }
        }
        [e setComputePipelineState:s->moe_add_batch]; [e setBuffer:s->bx offset:0 atIndex:0]; [e setBuffer:s->btmp offset:0 atIndex:1]; [e setBuffer:s->bw offset:0 atIndex:2]; [e setBytes:&m->experts_used length:sizeof(m->experts_used) atIndex:3]; [e setBytes:&m->dim length:sizeof(m->dim) atIndex:4]; [e setBytes:&rows length:sizeof(rows) atIndex:5]; dispatch_1d(e, s->moe_add_batch, rows * m->dim);
    }
}
/* PROFILING (scratch only): 0 = all; else only that stage is encoded.
   1 norm+mixer in-proj, 2 mixer elementwise (conv/l2/gates/gnorm | split/norm/rope/cache/sigmoid),
   3 out-proj+add+ffn_norm, 4 router+topk, 5 routed experts+moe_add, 6 shared expert / dense FFN,
   7 gdn recurrence, 8 attention row loop */
static int dbg_stage;
double dsm_stage_gpu[9][2];
#define ST(n) (dbg_stage == 0 || dbg_stage == (n))
static void encode_hybrid_moe_batch(dsm_metal *s, id<MTLComputeCommandEncoder> e,
                                    const dsm_layer *w, uint32_t rows) {
    const dsm_model *m = s->m;
    if (ST(4)) {
    if (s->mm_f32w_enabled) encode_matmul(s, e, w->gate_inp, s->bnorm, s->broute, rows);
    else encode_f32_rows(s, e, w->gate_inp, s->bnorm, s->broute, rows, m->dim,
                         m->experts);
    encode_moe_topk_batch(s, e, rows); }
    if (ST(5)) {
    int use_mm_id = mul_mm_id_enabled() && rows > 8;
    /* The first expert matmul builds the token-to-expert map; the others reuse it. */
    if (use_mm_id)
        use_mm_id = encode_expert_matmul(s, e, w->gate_exps, s->bnorm, s->bgate, rows, 1,
                                         m->dim, m->expert_hidden, 1);
    if (use_mm_id)
        use_mm_id = encode_expert_matmul(s, e, w->up_exps, s->bnorm, s->bup, rows, 1,
                                         m->dim, m->expert_hidden, 0);
    if (use_mm_id) {
        [e setComputePipelineState:s->expert_swiglu];
        [e setBuffer:s->bgate offset:0 atIndex:0]; [e setBuffer:s->bup offset:0 atIndex:1];
        [e setBytes:&m->expert_hidden length:sizeof(m->expert_hidden) atIndex:2];
        [e setBytes:&m->experts_used length:sizeof(m->experts_used) atIndex:3];
        [e setBytes:&rows length:sizeof(rows) atIndex:4];
        dispatch_1d(e, s->expert_swiglu, rows * m->expert_hidden * m->experts_used);
    }
    if (use_mm_id)
        use_mm_id = encode_expert_matmul(s, e, w->down_exps, s->bgate, s->btmp, rows,
                                         m->experts_used, m->expert_hidden, m->dim, 0);
    if (!use_mm_id) for (uint32_t row = 0; row < rows; ++row) {
        NSUInteger x_off = (NSUInteger)row * m->dim * sizeof(float);
        NSUInteger gate_off = (NSUInteger)row * m->experts_used * m->expert_hidden * sizeof(float);
        NSUInteger tmp_off = (NSUInteger)row * m->experts_used * m->dim * sizeof(float);
        encode_expert_gemv_batch_row(s, e, w->gate_exps, s->bnorm, x_off, s->bgate,
                                     gate_off, row, 0, m->expert_hidden);
        encode_expert_gemv_batch_row(s, e, w->up_exps, s->bnorm, x_off, s->bup,
                                     gate_off, row, 0, m->expert_hidden);
        uint32_t one = 1;
        [e setComputePipelineState:s->expert_swiglu]; [e setBuffer:s->bgate offset:gate_off atIndex:0];
        [e setBuffer:s->bup offset:gate_off atIndex:1]; [e setBytes:&m->expert_hidden length:sizeof(m->expert_hidden) atIndex:2];
        [e setBytes:&m->experts_used length:sizeof(m->experts_used) atIndex:3]; [e setBytes:&one length:sizeof(one) atIndex:4];
        dispatch_1d(e, s->expert_swiglu, m->expert_hidden * m->experts_used);
        encode_expert_gemv_batch_row(s, e, w->down_exps, s->bgate, gate_off, s->btmp,
                                     tmp_off, row, m->expert_hidden, m->dim);
    }
    [e setComputePipelineState:s->moe_add_batch]; [e setBuffer:s->bx offset:0 atIndex:0]; [e setBuffer:s->btmp offset:0 atIndex:1]; [e setBuffer:s->bw offset:0 atIndex:2];
    [e setBytes:&m->experts_used length:sizeof(m->experts_used) atIndex:3]; [e setBytes:&m->dim length:sizeof(m->dim) atIndex:4]; [e setBytes:&rows length:sizeof(rows) atIndex:5]; dispatch_1d(e, s->moe_add_batch, rows * m->dim);
    }
    if (!ST(6)) return;
    encode_f32_rows(s, e, w->gate_inp_shexp, s->bnorm, s->hb_shg, rows, m->dim, 1);
    encode_matmul(s, e, w->gate_shexp, s->bnorm, s->hb_sgate, rows);
    encode_matmul(s, e, w->up_shexp, s->bnorm, s->hb_sup, rows);
    { uint32_t n = rows * m->shared_hidden; [e setComputePipelineState:s->swiglu]; [e setBuffer:s->hb_sgate offset:0 atIndex:0]; [e setBuffer:s->hb_sup offset:0 atIndex:1]; [e setBytes:&n length:sizeof(n) atIndex:2]; dispatch_1d(e, s->swiglu, n); }
    encode_matmul(s, e, w->down_shexp, s->hb_sgate, s->hb_sdown, rows);
    { uint32_t n = rows * m->dim; [e setComputePipelineState:s->shexp_add_batch]; [e setBuffer:s->bx offset:0 atIndex:0]; [e setBuffer:s->hb_sdown offset:0 atIndex:1]; [e setBuffer:s->hb_shg offset:0 atIndex:2]; [e setBytes:&m->dim length:sizeof(m->dim) atIndex:3]; [e setBytes:&n length:sizeof(n) atIndex:4]; dispatch_1d(e, s->shexp_add_batch, n); }
}
static void encode_layer_hybrid_batch(dsm_metal *s, id<MTLComputeCommandEncoder> e,
                                      uint32_t layer, uint32_t pos0, uint32_t rows) {
    const dsm_model *m = s->m;
    const dsm_layer *w = &m->layer[layer];
    uint32_t qk = m->ssm_groups * m->ssm_state;
    uint32_t mixed = m->ssm_inner + 2 * qk;
    if (ST(1)) encode_norm_at(s, e, s->bx, 0, w->attn_norm, s->bnorm, 0, m->dim, rows);
    if (m->is_recurrent[layer]) {
        if (ST(1)) {
        encode_matmul(s, e, w->qkv, s->bnorm, s->hb_qkv, rows);
        encode_matmul(s, e, w->z, s->bnorm, s->hb_z, rows);
        /* qwen35moe ships F32 alpha/beta, dense qwen35 ships Q8_0. */
        if (w->alpha->type == GGML_TYPE_F32 && !s->mm_f32w_enabled)
            encode_f32_rows(s, e, w->alpha, s->bnorm, s->hb_g, rows, m->dim,
                            m->ssm_vheads);
        else encode_matmul(s, e, w->alpha, s->bnorm, s->hb_g, rows);
        if (w->beta->type == GGML_TYPE_F32 && !s->mm_f32w_enabled)
            encode_f32_rows(s, e, w->beta, s->bnorm, s->hb_beta, rows, m->dim,
                            m->ssm_vheads);
        else encode_matmul(s, e, w->beta, s->bnorm, s->hb_beta, rows);
        }
        id<MTLBuffer> wb; uint64_t off;
        if (ST(2)) {
        ssm_conv_batch_params conv = {mixed, m->ssm_conv - 1, rows};
        if (!tensor_view(s, w->conv1d, &wb, &off)) return;
        [e setComputePipelineState:s->ssm_conv_batch]; set_checked(e, s->hb_qkv, 0, (NSUInteger)rows * mixed * sizeof(float), 0); set_checked(e, s->conv_state, (NSUInteger)layer * mixed * (m->ssm_conv - 1) * sizeof(float), (NSUInteger)mixed * (m->ssm_conv - 1) * sizeof(float), 1); set_checked(e, wb, (NSUInteger)off, (NSUInteger)w->conv1d->bytes, 2); set_checked(e, s->hb_conv, 0, (NSUInteger)rows * mixed * sizeof(float), 3); [e setBytes:&conv length:sizeof(conv) atIndex:4]; dispatch_1d(e, s->ssm_conv_batch, mixed);
        uint32_t n = rows * qk;
        [e setComputePipelineState:s->split_gdn_qk_batch]; set_checked(e, s->hb_conv, 0, (NSUInteger)rows * mixed * sizeof(float), 0); set_checked(e, s->hb_dq, 0, (NSUInteger)n * sizeof(float), 1); set_checked(e, s->hb_dk, 0, (NSUInteger)n * sizeof(float), 2); [e setBytes:&qk length:sizeof(qk) atIndex:3]; [e setBytes:&mixed length:sizeof(mixed) atIndex:4]; [e setBytes:&n length:sizeof(n) atIndex:5]; dispatch_1d(e, s->split_gdn_qk_batch, n);
        norm_params l2 = {m->ssm_state, rows * m->ssm_groups, m->rms_eps};
        [e setComputePipelineState:s->l2norm]; set_checked(e, s->hb_dq, 0, (NSUInteger)n * sizeof(float), 0); [e setBytes:&l2 length:sizeof(l2) atIndex:1]; [e setThreadgroupMemoryLength:256 * sizeof(float) atIndex:0]; [e dispatchThreadgroups:MTLSizeMake(l2.groups, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        set_checked(e, s->hb_dk, 0, (NSUInteger)n * sizeof(float), 0); [e dispatchThreadgroups:MTLSizeMake(l2.groups, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        n = rows * m->ssm_vheads; if (!tensor_view(s, w->dt_bias, &wb, &off)) return;
        [e setComputePipelineState:s->gdn_gates_batch]; set_checked(e, s->hb_g, 0, (NSUInteger)n * sizeof(float), 0); set_checked(e, s->hb_beta, 0, (NSUInteger)n * sizeof(float), 1); set_checked(e, wb, (NSUInteger)off, (NSUInteger)w->dt_bias->bytes, 2); if (!tensor_view(s, w->ssm_a, &wb, &off)) return; set_checked(e, wb, (NSUInteger)off, (NSUInteger)w->ssm_a->bytes, 3); [e setBytes:&m->ssm_vheads length:sizeof(m->ssm_vheads) atIndex:4]; [e setBytes:&n length:sizeof(n) atIndex:5]; dispatch_1d(e, s->gdn_gates_batch, n);
        }
        if (ST(7)) {
        gdn_batch_params gdn = {m->ssm_groups, m->ssm_vheads, m->ssm_state, layer, rows, qk, mixed, m->ssm_inner};
        [e setComputePipelineState:s->gdn_batch]; set_checked(e, s->hb_dq, 0, (NSUInteger)rows * qk * sizeof(float), 0); set_checked(e, s->hb_dk, 0, (NSUInteger)rows * qk * sizeof(float), 1); set_checked(e, s->hb_conv, (NSUInteger)2 * qk * sizeof(float), (NSUInteger)(rows - 1) * mixed * sizeof(float) + m->ssm_inner * sizeof(float), 2); set_checked(e, s->hb_g, 0, (NSUInteger)rows * m->ssm_vheads * sizeof(float), 3); set_checked(e, s->hb_beta, 0, (NSUInteger)rows * m->ssm_vheads * sizeof(float), 4); set_checked(e, s->ssm_state, 0, s->ssm_state.length, 5); set_checked(e, s->hb_mix, 0, (NSUInteger)rows * m->ssm_inner * sizeof(float), 6); [e setBytes:&gdn length:sizeof(gdn) atIndex:7]; [e dispatchThreadgroups:MTLSizeMake((m->ssm_state + 3) / 4, m->ssm_vheads, 1) threadsPerThreadgroup:MTLSizeMake(32, 4, 1)];
        }
        if (ST(2)) {
        norm_params gated = {m->ssm_state, rows * m->ssm_vheads, m->rms_eps}; if (!tensor_view(s, w->ssm_norm, &wb, &off)) return;
        [e setComputePipelineState:s->gated_rmsnorm]; set_checked(e, s->hb_mix, 0, (NSUInteger)rows * m->ssm_inner * sizeof(float), 0); set_checked(e, s->hb_z, 0, (NSUInteger)rows * m->ssm_inner * sizeof(float), 1); set_checked(e, wb, (NSUInteger)off, (NSUInteger)w->ssm_norm->bytes, 2); [e setBytes:&gated length:sizeof(gated) atIndex:3]; [e setThreadgroupMemoryLength:256 * sizeof(float) atIndex:0]; [e dispatchThreadgroups:MTLSizeMake(gated.groups, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        }
        if (ST(3)) encode_matmul(s, e, w->ssm_out, s->hb_mix, s->hb_out, rows);
    } else {
        if (ST(1)) { encode_matmul(s, e, w->q, s->bnorm, s->hb_qg, rows); encode_matmul(s, e, w->k, s->bnorm, s->hb_ak, rows); encode_matmul(s, e, w->v, s->bnorm, s->hb_av, rows); }
        if (ST(2)) {
        { uint32_t n = rows * m->q_width;
          [e setComputePipelineState:s->split_q_gate];
          set_checked(e, s->hb_qg, 0, (NSUInteger)rows * 2 * m->q_width * sizeof(float), 0);
          set_checked(e, s->hb_aq, 0, (NSUInteger)n * sizeof(float), 1);
          set_checked(e, s->hb_agate, 0, (NSUInteger)n * sizeof(float), 2);
          [e setBytes:&m->key_dim length:sizeof(m->key_dim) atIndex:3];
          dispatch_1d(e, s->split_q_gate, n); }
        encode_norm_at(s, e, s->hb_aq, 0, w->q_norm, s->hb_aq, 0, m->key_dim, rows * m->heads); encode_norm_at(s, e, s->hb_ak, 0, w->k_norm, s->hb_ak, 0, m->key_dim, rows * m->kv_heads);
        rope_partial_batch_params rope = {m->heads, m->key_dim, m->rope_dims,
                                          pos0, rows, m->rope_base};
        [e setComputePipelineState:s->rope_partial_batch];
        set_checked(e, s->hb_aq, 0, (NSUInteger)rows * m->q_width * sizeof(float), 0);
        [e setBytes:&rope length:sizeof(rope) atIndex:1];
        dispatch_1d(e, s->rope_partial_batch, rows * m->heads * (m->rope_dims / 2));
        rope.heads = m->kv_heads;
        [e setComputePipelineState:s->rope_partial_batch];
        set_checked(e, s->hb_ak, 0, (NSUInteger)rows * m->kv_heads * m->key_dim * sizeof(float), 0);
        [e setBytes:&rope length:sizeof(rope) atIndex:1];
        dispatch_1d(e, s->rope_partial_batch, rows * m->kv_heads * (m->rope_dims / 2));
        cache_batch_params cache = {m->attn_index[layer], pos0, s->context,
                                    m->kv_heads * m->key_dim, rows};
        [e setComputePipelineState:s->cache_batch]; set_checked(e, s->hb_ak, 0, (NSUInteger)rows * cache.width * sizeof(float), 0); set_checked(e, s->keys, 0, s->keys.length, 1); [e setBytes:&cache length:sizeof(cache) atIndex:2]; dispatch_1d(e, s->cache_batch, rows * cache.width);
        [e setComputePipelineState:s->cache_batch]; set_checked(e, s->hb_av, 0, (NSUInteger)rows * cache.width * sizeof(float), 0); set_checked(e, s->values, 0, s->values.length, 1); [e setBytes:&cache length:sizeof(cache) atIndex:2]; dispatch_1d(e, s->cache_batch, rows * cache.width);
        }
        if (ST(8) && rows > 8 && attention_prefill_enabled() &&
            m->key_dim == 256 && m->heads / m->kv_heads <= 8) {
            attn_prefill_params attention = {m->heads, m->kv_heads, m->key_dim,
                                              pos0, s->context, m->attn_index[layer], rows};
            [e setComputePipelineState:s->attention_prefill_d256];
            set_checked(e, s->hb_aq, 0, (NSUInteger)rows * m->q_width * sizeof(float), 0);
            set_checked(e, s->keys, 0, s->keys.length, 1);
            set_checked(e, s->values, 0, s->values.length, 2);
            set_checked(e, s->hb_mix, 0, (NSUInteger)rows * m->attn_width * sizeof(float), 3);
            [e setBytes:&attention length:sizeof(attention) atIndex:4];
            [e setThreadgroupMemoryLength:32768 atIndex:0];
            [e dispatchThreadgroups:MTLSizeMake((rows + 3) / 4, m->kv_heads, 1)
              threadsPerThreadgroup:MTLSizeMake(32, m->heads / m->kv_heads, 1)];
        } else if (ST(8)) for (uint32_t row = 0; row < rows; ++row) {
            encode_attention_decode_at(s, e, m->attn_index[layer], pos0 + row,
                                       s->hb_aq, (NSUInteger)row * m->q_width * sizeof(float),
                                       s->hb_mix, (NSUInteger)row * m->attn_width * sizeof(float));
        }
        if (ST(2)) { uint32_t n = rows * m->attn_width; [e setComputePipelineState:s->sigmoid_mul]; set_checked(e, s->hb_mix, 0, (NSUInteger)n * sizeof(float), 0); set_checked(e, s->hb_agate, 0, (NSUInteger)rows * m->q_width * sizeof(float), 1); [e setBytes:&n length:sizeof(n) atIndex:2]; dispatch_1d(e, s->sigmoid_mul, n); }
        if (ST(3)) encode_matmul(s, e, w->o, s->hb_mix, s->hb_out, rows);
    }
    if (ST(3)) { uint32_t n = rows * m->dim; [e setComputePipelineState:s->add]; [e setBuffer:s->bx offset:0 atIndex:0]; [e setBuffer:s->hb_out offset:0 atIndex:1]; [e setBytes:&n length:sizeof(n) atIndex:2]; dispatch_1d(e, s->add, n);
    encode_norm_at(s, e, s->bx, 0, w->ffn_norm, s->bnorm, 0, m->dim, rows); }
    if (m->is_moe) {
        encode_hybrid_moe_batch(s, e, w, rows);
    } else if (ST(6)) {
        encode_matmul(s, e, w->gate, s->bnorm, s->hb_sgate, rows);
        encode_matmul(s, e, w->up, s->bnorm, s->hb_sup, rows);
        { uint32_t n = rows * m->hidden; [e setComputePipelineState:s->swiglu];
          set_checked(e, s->hb_sgate, 0, (NSUInteger)n * sizeof(float), 0);
          set_checked(e, s->hb_sup, 0, (NSUInteger)n * sizeof(float), 1);
          [e setBytes:&n length:sizeof(n) atIndex:2]; dispatch_1d(e, s->swiglu, n); }
        encode_matmul(s, e, w->down, s->hb_sgate, s->hb_sdown, rows);
        { uint32_t n = rows * m->dim; [e setComputePipelineState:s->add];
          set_checked(e, s->bx, 0, (NSUInteger)n * sizeof(float), 0);
          set_checked(e, s->hb_sdown, 0, (NSUInteger)n * sizeof(float), 1);
          [e setBytes:&n length:sizeof(n) atIndex:2]; dispatch_1d(e, s->add, n); }
    }
}
static int prefill_chunk(dsm_metal *s, const int32_t *tokens, uint32_t rows,
                         float *logits) {
    const dsm_model *m = s->m;
    const ggml_type_info *info = gguf_type_info(m->token_embd->type);
    float *x = s->bx.contents;
    uint32_t pos0 = s->pos;
    for (uint32_t row = 0; row < rows; ++row) {
        int32_t token = tokens[row];
        if (token < 0 || (uint32_t)token >= m->vocab || pos0 + row >= s->context) return 0;
        const uint8_t *embedding = (const uint8_t *)m->token_embd->data + (uint64_t)token * (m->dim / info->block_elems) * info->block_bytes;
        dsm_dequant_row(m->token_embd->type, embedding, x + (uint64_t)row * m->dim, m->dim);
    }
    id<MTLCommandBuffer> cb = [s->queue commandBuffer];
    id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
    static int prof = -1; if (prof < 0) prof = getenv("DSM_STAGE_PROF") != NULL;
    if (prof && qwen35_arch(m)) {
        [e endEncoding]; [cb commit]; [cb waitUntilCompleted];
        for (uint32_t layer = 0; layer < m->layers; ++layer) for (int st = 1; st <= 8; ++st) {
            cb = [s->queue commandBuffer]; e = [cb computeCommandEncoder]; dbg_stage = st;
            encode_layer_hybrid_batch(s, e, layer, pos0, rows); dbg_stage = 0;
            [e endEncoding]; [cb commit]; [cb waitUntilCompleted];
            dsm_stage_gpu[st][m->is_recurrent[layer] ? 0 : 1] += cb.GPUEndTime - cb.GPUStartTime;
        }
        cb = [s->queue commandBuffer]; e = [cb computeCommandEncoder];
    } else
    for (uint32_t layer = 0; layer < m->layers; ++layer) {
        if (qwen35_arch(m)) encode_layer_hybrid_batch(s, e, layer, pos0, rows);
        else encode_layer_batch(s, e, layer, pos0, rows);
    }
    if (logits) {
        NSUInteger off = (NSUInteger)(rows - 1) * m->dim * sizeof(float);
        encode_norm_at(s, e, s->bx, off, m->output_norm, s->norm_x, 0, m->dim, 1);
        encode_gemv(s, e, m->output, s->norm_x, s->logits);
    }
    [e endEncoding]; [cb commit]; [cb waitUntilCompleted];
    if (cb.status != MTLCommandBufferStatusCompleted) return 0;
    s->pos += rows;
    if (logits) memcpy(logits, s->logits.contents, m->vocab * sizeof(*logits));
    return 1;
}
int dsm_metal_prefill(dsm_metal *s, const int32_t *tokens, uint32_t n,
                       float *logits) {
    if (!s || !tokens || !n || !logits || n > s->context - s->pos) return 0;
    if ((qwen35_arch(s->m) && (!hybrid_prefill_enabled() || getenv("DSM_DUMP_DIR"))) ||
        s->profile.enabled || !prefill_enabled()) {
        for (uint32_t i = 0; i < n; ++i) if (!dsm_metal_token(s, tokens[i], logits)) return 0;
        return 1;
    }
    for (uint32_t start = 0; start < n;) {
        uint32_t rows = n - start;
        if (rows > s->batch_chunk) rows = s->batch_chunk;
        if (!prefill_chunk(s, tokens + start, rows, start + rows == n ? logits : NULL)) return 0;
        start += rows;
    }
    return 1;
}
int dsm_metal_token(dsm_metal *s,int32_t token,float *logits){const dsm_model *m=s->m;if(token<0||(uint32_t)token>=m->vocab||s->pos>=s->context)return 0;const ggml_type_info *info=gguf_type_info(m->token_embd->type);const uint8_t *row=(const uint8_t *)m->token_embd->data+(uint64_t)token*(m->dim/info->block_elems)*info->block_bytes;float *x=s->x.contents;dsm_dequant_row(m->token_embd->type,row,x,m->dim);if(s->profile.enabled&&!qwen35_arch(m))return profile_token(s,logits);
    if(qwen35_arch(m)){if(getenv("DSM_DUMP_DIR")){for(uint32_t l=0;l<m->layers;++l){id<MTLCommandBuffer> cb=[s->queue commandBuffer];id<MTLComputeCommandEncoder> e=[cb computeCommandEncoder];encode_layer_hybrid(s,e,l);[e endEncoding];[cb commit];[cb waitUntilCompleted];if(cb.status!=MTLCommandBufferStatusCompleted)return 0;dump_hybrid_layer(s,l);}id<MTLCommandBuffer> cb=[s->queue commandBuffer];id<MTLComputeCommandEncoder> e=[cb computeCommandEncoder];encode_norm(s,e,s->x,m->output_norm,s->norm_x,m->dim,1);encode_gemv(s,e,m->output,s->norm_x,s->logits);[e endEncoding];[cb commit];[cb waitUntilCompleted];if(cb.status!=MTLCommandBufferStatusCompleted)return 0;}else{id<MTLCommandBuffer> cb=[s->queue commandBuffer];id<MTLComputeCommandEncoder> e=[cb computeCommandEncoder];for(uint32_t l=0;l<m->layers;++l)encode_layer_hybrid(s,e,l);encode_norm(s,e,s->x,m->output_norm,s->norm_x,m->dim,1);encode_gemv(s,e,m->output,s->norm_x,s->logits);[e endEncoding];[cb commit];[cb waitUntilCompleted];if(cb.status!=MTLCommandBufferStatusCompleted)return 0;}memcpy(logits,s->logits.contents,m->vocab*4);++s->pos;return 1;}
    id<MTLCommandBuffer> cb=[s->queue commandBuffer];id<MTLComputeCommandEncoder> e=[cb computeCommandEncoder];for(uint32_t l=0;l<m->layers;++l)encode_layer(s,e,l);if(!s->decode_fuse||m->is_moe)encode_norm(s,e,s->x,m->output_norm,s->norm_x,m->dim,1);encode_gemv(s,e,m->output,s->norm_x,s->logits);[e endEncoding];[cb commit];[cb waitUntilCompleted];if(cb.status!=MTLCommandBufferStatusCompleted)return 0;memcpy(logits,s->logits.contents,m->vocab*4);++s->pos;return 1;}

double dsm_metal_gemv_bench(dsm_metal *s, const gguf_tensor *w, int reps) {
    uint32_t rows = w->ndim > 1 ? (uint32_t)w->dims[1] : 1;
    id<MTLBuffer> x = buffer(s->device, w->dims[0] * sizeof(float));
    id<MTLBuffer> y = buffer(s->device, (uint64_t)rows * sizeof(float));
    if (!x || !y || reps < 1) return -1.0;
    float *input = x.contents;
    for (uint64_t i = 0; i < w->dims[0]; ++i) input[i] = 0.001f * (float)(i % 97);
    id<MTLCommandBuffer> cb = [s->queue commandBuffer];
    id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
    for (int rep = 0; rep < reps; ++rep) encode_gemv(s, e, w, x, y);
    [e endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    return cb.status == MTLCommandBufferStatusCompleted
                   ? cb.GPUEndTime - cb.GPUStartTime : -1.0;
}
