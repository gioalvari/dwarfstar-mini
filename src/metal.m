#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "metal.h"
#include "quants.h"

#include <assert.h>
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
typedef struct { uint32_t heads, kv_heads, dim, pos, context, layer; } attn_params;
typedef struct { uint32_t heads, dim, pos0, rows; float base; } rope_batch_params;
typedef struct { uint32_t layer, pos0, context, width, rows; } cache_batch_params;
typedef struct { uint32_t heads, kv_heads, dim, pos0, context, layer, rows; } attn_prefill_params;
typedef struct { uint32_t in, out, expert_stride, out_stride, input_stride; } expert_gemv_params;

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
struct dsm_metal {
    const dsm_model *m; uint32_t pos, context, n_views;
    id<MTLDevice> device; id<MTLCommandQueue> queue;
    id<MTLComputePipelineState> q8, q2k, q3k, q4k, q5k, q6k, f32, norm, rope, cache, attention, add, swiglu, route_topk, expert_q8, expert_q3k, expert_q4k, expert_swiglu, moe_add;
    id<MTLComputePipelineState> rope_batch, cache_batch, attention_prefill, moe_add_batch;
    id<MTLComputePipelineState> mm_q8, mm_q2k, mm_q3k, mm_q4k, mm_q6k, mm_id_map0, mm_id_q8, mm_id_q2k, mm_id_q3k, mm_id_q4k, mm_id_q6k;
    weight_view *views;
    id<MTLBuffer> x, norm_x, q, k, v, att, tmp, gate, up, route, expert_ids, expert_weights, scores, keys, values, logits;
    id<MTLBuffer> bx, bnorm, bq, bk, bv, batt, btmp, bgate, bup, broute, bids, bw, mm_id_tpe, mm_id_hids;
    uint32_t batch_chunk;
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
    if (!fn) { set_error(error, cap, "required Metal kernel is missing"); return nil; }
    id<MTLComputePipelineState> state = [dev newComputePipelineStateWithFunction:fn error:&ns_error];
    if (!state) set_error(error, cap, ns_error.localizedDescription.UTF8String);
    return state;
}
static id<MTLBuffer> buffer(id<MTLDevice> device, uint64_t bytes) {
    return [device newBufferWithLength:(NSUInteger)bytes options:MTLResourceStorageModeShared];
}
static uint32_t prefill_chunk_size(const dsm_model *m) {
    const char *value = getenv("DSM_PREFILL_CHUNK");
    char *end = NULL;
    unsigned long chunk = value ? strtoul(value, &end, 10) : 512;
    if (!value || !*value || *end || !chunk || chunk > UINT32_MAX) chunk = 512;
    if (chunk > m->context) chunk = m->context;
    return (uint32_t)chunk;
}
static int prefill_enabled(void) {
    const char *value = getenv("DSM_PREFILL");
    return !value || strcmp(value, "0");
}
static int mul_mm_id_enabled(void) {
    const char *value = getenv("DSM_MUL_MM_ID");
    return !value || strcmp(value, "0");
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
                                uint32_t output_stride);
static void encode_gemv_at(dsm_metal *s, id<MTLComputeCommandEncoder> e,
                           const gguf_tensor *w, id<MTLBuffer> x,
                           NSUInteger x_off, id<MTLBuffer> y,
                           NSUInteger y_off) {
    id<MTLBuffer> wb; uint64_t off; if (!tensor_view(s, w, &wb, &off)) return;
    gemv_params p = {(uint32_t)w->dims[0], (uint32_t)w->dims[1]};
    gemv_mul_mv_params mul_mv = {
        .ne00 = (int32_t)w->dims[0], .ne01 = (int32_t)w->dims[1], .ne02 = 1,
        .nb00 = 0, .nb01 = w->bytes / w->dims[1], .nb02 = 0, .nb03 = 0,
        .ne10 = (int32_t)w->dims[0], .ne11 = 1, .ne12 = 1,
        .nb10 = sizeof(float), .nb11 = 0, .nb12 = 0, .nb13 = 0,
        .ne0 = (int32_t)w->dims[1], .ne1 = 1, .nr0 = 0, .r2 = 1, .r3 = 1,
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
        [e setBytes:&p length:sizeof(p) atIndex:3];
        [e setThreadgroupMemoryLength:256 * sizeof(float) atIndex:0]; [e dispatchThreadgroups:MTLSizeMake(p.out, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    } else if (w->type == GGML_TYPE_Q4_K || w->type == GGML_TYPE_Q6_K) {
        [e setBytes:&mul_mv length:sizeof(mul_mv) atIndex:3];
        [e dispatchThreadgroups:MTLSizeMake((p.out + 3) / 4, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 2, 1)];
    } else {
        [e setBytes:&p length:sizeof(p) atIndex:3];
        [e setThreadgroupMemoryLength:256 * sizeof(float) atIndex:0]; [e dispatchThreadgroups:MTLSizeMake(p.out, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    }
}
static void encode_gemv(dsm_metal *s, id<MTLComputeCommandEncoder> e,
                         const gguf_tensor *w, id<MTLBuffer> x,
                         id<MTLBuffer> y) {
    encode_gemv_at(s, e, w, x, 0, y, 0);
}
static id<MTLComputePipelineState> expert_gemv_pipeline(const dsm_metal *s,
                                                         enum ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q8_0: return s->expert_q8;
        case GGML_TYPE_Q3_K: return s->expert_q3k;
        case GGML_TYPE_Q4_K: return s->expert_q4k;
        default: return nil;
    }
}
static int expert_gemv_supported(enum ggml_type type) {
    return type == GGML_TYPE_Q8_0 || type == GGML_TYPE_Q3_K ||
           type == GGML_TYPE_Q4_K;
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
    if (w->type == GGML_TYPE_Q3_K || w->type == GGML_TYPE_Q4_K) {
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
    if (w->type == GGML_TYPE_Q3_K || w->type == GGML_TYPE_Q4_K) {
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
    [e setThreadgroupMemoryLength:m->experts * sizeof(float) atIndex:0];
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
    [e setThreadgroupMemoryLength:m->experts * sizeof(float) atIndex:0];
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
    if (w->down_exps->type == GGML_TYPE_Q3_K || w->down_exps->type == GGML_TYPE_Q4_K) {
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

int dsm_metal_available(void) { return MTLCreateSystemDefaultDevice() != nil; }
int dsm_metal_init(dsm_metal **out,const dsm_model *m,uint32_t context,char *error,uint32_t error_size) {
    NSError *ns_error = nil;
    NSString *source = nil;
    MTLCompileOptions *options = nil;
    id<MTLLibrary> lib = nil;
    *out=NULL; if (!context || context > m->context) { set_error(error,error_size,"invalid Metal context"); return 0; }
    dsm_metal *s=calloc(1,sizeof(*s)); if(!s){set_error(error,error_size,"out of memory");return 0;} s->m=m;s->context=context;s->profile.enabled=getenv("DSM_PROFILE") && !strcmp(getenv("DSM_PROFILE"),"1");s->device=MTLCreateSystemDefaultDevice(); if(!s->device){set_error(error,error_size,"no Metal device");goto bad;}s->queue=[s->device newCommandQueue];
    source=[NSString stringWithContentsOfFile:shader_path() encoding:NSUTF8StringEncoding error:&ns_error];if(!source){set_error(error,error_size,ns_error.localizedDescription.UTF8String);goto bad;}options=[MTLCompileOptions new];if (@available(macOS 15.0,*)) options.mathMode=MTLMathModeSafe;
    lib=[s->device newLibraryWithSource:source options:options error:&ns_error];if(!lib){set_error(error,error_size,ns_error.localizedDescription.UTF8String);goto bad;}
    if (m->is_moe && (!expert_gemv_supported(m->layer[0].gate_exps->type) ||
                      !expert_gemv_supported(m->layer[0].up_exps->type) ||
                      !expert_gemv_supported(m->layer[0].down_exps->type))) {
        set_error(error, error_size, "unsupported Metal MoE expert quantization"); goto bad;
    }
    s->q8=pipeline(s->device,lib,@"q8_gemv",error,error_size);s->q2k=pipeline(s->device,lib,@"q2k_gemv",error,error_size);s->q3k=pipeline(s->device,lib,@"q3k_gemv",error,error_size);s->q4k=pipeline(s->device,lib,@"q4k_gemv",error,error_size);s->q5k=pipeline(s->device,lib,@"q5k_gemv",error,error_size);s->q6k=pipeline(s->device,lib,@"q6k_gemv",error,error_size);s->f32=pipeline(s->device,lib,@"f32_gemv",error,error_size);s->norm=pipeline(s->device,lib,@"rmsnorm",error,error_size);s->rope=pipeline(s->device,lib,@"rope_neox",error,error_size);s->cache=pipeline(s->device,lib,@"cache_write",error,error_size);s->attention=pipeline(s->device,lib,@"attention_decode",error,error_size);s->add=pipeline(s->device,lib,@"add_inplace",error,error_size);s->swiglu=pipeline(s->device,lib,@"swiglu",error,error_size);s->route_topk=pipeline(s->device,lib,@"route_topk",error,error_size);s->expert_q8=pipeline(s->device,lib,@"expert_q8_0_gemv",error,error_size);s->expert_q3k=pipeline(s->device,lib,@"expert_q3k_gemv",error,error_size);s->expert_q4k=pipeline(s->device,lib,@"expert_q4k_gemv",error,error_size);s->expert_swiglu=pipeline(s->device,lib,@"expert_swiglu",error,error_size);s->moe_add=pipeline(s->device,lib,@"moe_add",error,error_size);if(!s->q8||!s->q2k||!s->q3k||!s->q4k||!s->q5k||!s->q6k||!s->f32||!s->norm||!s->rope||!s->cache||!s->attention||!s->add||!s->swiglu||!s->route_topk||!s->expert_q8||!s->expert_q3k||!s->expert_q4k||!s->expert_swiglu||!s->moe_add)goto bad;
    s->rope_batch=pipeline(s->device,lib,@"rope_neox_batch",error,error_size);s->cache_batch=pipeline(s->device,lib,@"cache_write_batch",error,error_size);s->attention_prefill=pipeline(s->device,lib,@"attention_prefill",error,error_size);s->moe_add_batch=pipeline(s->device,lib,@"moe_add_batch",error,error_size);s->mm_q8=pipeline(s->device,lib,@"mul_mm_q8_0",error,error_size);s->mm_q2k=pipeline(s->device,lib,@"mul_mm_q2_K",error,error_size);s->mm_q3k=pipeline(s->device,lib,@"mul_mm_q3_K",error,error_size);s->mm_q4k=pipeline(s->device,lib,@"mul_mm_q4_K",error,error_size);s->mm_q6k=pipeline(s->device,lib,@"mul_mm_q6_K",error,error_size);s->mm_id_map0=pipeline(s->device,lib,@"mul_mm_id_map0",error,error_size);s->mm_id_q8=pipeline(s->device,lib,@"mul_mm_id_q8_0",error,error_size);s->mm_id_q2k=pipeline(s->device,lib,@"mul_mm_id_q2_K",error,error_size);s->mm_id_q3k=pipeline(s->device,lib,@"mul_mm_id_q3_K",error,error_size);s->mm_id_q4k=pipeline(s->device,lib,@"mul_mm_id_q4_K",error,error_size);s->mm_id_q6k=pipeline(s->device,lib,@"mul_mm_id_q6_K",error,error_size);if(!s->rope_batch||!s->cache_batch||!s->attention_prefill||!s->moe_add_batch||!s->mm_q8||!s->mm_q2k||!s->mm_q3k||!s->mm_q4k||!s->mm_q6k||!s->mm_id_map0||!s->mm_id_q8||!s->mm_id_q2k||!s->mm_id_q3k||!s->mm_id_q4k||!s->mm_id_q6k)goto bad;
    const uint64_t page=(uint64_t)getpagesize(), start=m->gguf.tensor_data_offset&~(page-1), end=m->gguf.size, max=[s->device maxBufferLength]&~(page-1);if(!max){set_error(error,error_size,"Metal max buffer length is zero");goto bad;}uint64_t max_tensor=0;for(uint64_t i=0;i<m->gguf.n_tensors;++i)if(m->gguf.tensors[i].bytes>max_tensor)max_tensor=m->gguf.tensors[i].bytes;uint64_t overlap=((max_tensor+page-1)&~(page-1))+page,step=end-start<=max?max:max-overlap;if(overlap>=max){set_error(error,error_size,"tensor larger than Metal max buffer length");goto bad;}s->n_views=end-start<=max?1:(uint32_t)((end-start-overlap+step-1)/step);s->views=calloc(s->n_views,sizeof(*s->views));if(!s->views){set_error(error,error_size,"out of memory");goto bad;}for(uint32_t i=0;i<s->n_views;++i){uint64_t off=start+(uint64_t)i*step,bytes=end-off<max?end-off:max;bytes=(bytes+page-1)&~(page-1);s->views[i].buffer=[s->device newBufferWithBytesNoCopy:(void *)(m->gguf.map+off) length:(NSUInteger)bytes options:MTLResourceStorageModeShared deallocator:nil];s->views[i].file_offset=off;s->views[i].bytes=bytes;if(!s->views[i].buffer){set_error(error,error_size,"cannot wrap GGUF mmap in Metal buffer");goto bad;}}for(uint64_t i=0;i<m->gguf.n_tensors;++i){id<MTLBuffer> wb;uint64_t off;if(!tensor_view(s,&m->gguf.tensors[i],&wb,&off)){set_error(error,error_size,"GGUF tensor not mappable to Metal view");goto bad;}}
    uint32_t ffn=m->is_moe?m->expert_hidden:m->hidden, slots=m->is_moe?m->experts_used:1;uint64_t kv=(uint64_t)m->layers*context*m->kv_heads*m->key_dim*4;s->x=buffer(s->device,m->dim*4);s->norm_x=buffer(s->device,m->dim*4);s->q=buffer(s->device,m->q_width*4);s->k=buffer(s->device,m->kv_heads*m->key_dim*4);s->v=buffer(s->device,m->kv_heads*m->value_dim*4);s->att=buffer(s->device,m->attn_width*4);s->tmp=buffer(s->device,m->dim*slots*4);s->gate=buffer(s->device,ffn*slots*4);s->up=buffer(s->device,ffn*slots*4);s->route=m->is_moe?buffer(s->device,m->experts*4):NULL;s->expert_ids=m->is_moe?buffer(s->device,m->experts_used*4):NULL;s->expert_weights=m->is_moe?buffer(s->device,m->experts_used*4):NULL;s->scores=buffer(s->device,(uint64_t)context*m->heads*4);s->keys=buffer(s->device,kv);s->values=buffer(s->device,kv);s->logits=buffer(s->device,m->vocab*4);s->batch_chunk=prefill_chunk_size(m);s->bx=buffer(s->device,(uint64_t)s->batch_chunk*m->dim*4);s->bnorm=buffer(s->device,(uint64_t)s->batch_chunk*m->dim*4);s->bq=buffer(s->device,(uint64_t)s->batch_chunk*m->q_width*4);s->bk=buffer(s->device,(uint64_t)s->batch_chunk*m->kv_heads*m->key_dim*4);s->bv=buffer(s->device,(uint64_t)s->batch_chunk*m->kv_heads*m->value_dim*4);s->batt=buffer(s->device,(uint64_t)s->batch_chunk*m->attn_width*4);s->btmp=buffer(s->device,(uint64_t)s->batch_chunk*m->dim*slots*4);s->bgate=buffer(s->device,(uint64_t)s->batch_chunk*ffn*slots*4);s->bup=buffer(s->device,(uint64_t)s->batch_chunk*ffn*slots*4);s->broute=m->is_moe?buffer(s->device,(uint64_t)s->batch_chunk*m->experts*4):NULL;s->bids=m->is_moe?buffer(s->device,(uint64_t)s->batch_chunk*m->experts_used*4):NULL;s->bw=m->is_moe?buffer(s->device,(uint64_t)s->batch_chunk*m->experts_used*4):NULL;uint32_t max_tiles=m->is_moe?matmul_id_max_tiles(m->experts,m->experts_used,s->batch_chunk):0;s->mm_id_tpe=m->is_moe?buffer(s->device,(uint64_t)(1+3*max_tiles)*sizeof(uint32_t)):NULL;s->mm_id_hids=m->is_moe?buffer(s->device,(uint64_t)(m->experts_used*s->batch_chunk+m->experts)*sizeof(int32_t)):NULL;if(!s->x||!s->norm_x||!s->q||!s->k||!s->v||!s->att||!s->tmp||!s->gate||!s->up||(m->is_moe&&(!s->route||!s->expert_ids||!s->expert_weights))||!s->scores||!s->keys||!s->values||!s->logits||!s->bx||!s->bnorm||!s->bq||!s->bk||!s->bv||!s->batt||!s->btmp||!s->bgate||!s->bup||(m->is_moe&&(!s->broute||!s->bids||!s->bw||!s->mm_id_tpe||!s->mm_id_hids))){set_error(error,error_size,"cannot allocate Metal activations");goto bad;}*out=s;return 1;
bad:dsm_metal_free(s);return 0;
}
void dsm_metal_free(dsm_metal *s){if(!s)return;if(s->profile.enabled)dsm_metal_profile_print(s);free(s->views);free(s);}
uint32_t dsm_metal_pos(const dsm_metal *s){return s->pos;}
int dsm_metal_truncate(dsm_metal *s,uint32_t pos){if(!s||pos>s->pos)return 0;s->pos=pos;return 1;}
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
                              (uint32_t)w->dims[1])) return 0;
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
    [e setComputePipelineState:expert_gemv_pipeline(s,w->type)];[e setBuffer:wb offset:(NSUInteger)off atIndex:0];[e setBuffer:x offset:0 atIndex:1];[e setBuffer:y offset:0 atIndex:2];[e setBytes:&p length:sizeof(p) atIndex:3];[e setBuffer:ids offset:0 atIndex:4];if(w->type==GGML_TYPE_Q3_K||w->type==GGML_TYPE_Q4_K){[e dispatchThreadgroups:MTLSizeMake((p.out+3)/4,1,1) threadsPerThreadgroup:MTLSizeMake(32,2,1)];}else{[e setThreadgroupMemoryLength:256*sizeof(float) atIndex:0];[e dispatchThreadgroups:MTLSizeMake(p.out,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];}
    [e endEncoding];[cb commit];[cb waitUntilCompleted];if(cb.status!=MTLCommandBufferStatusCompleted)return 0;memcpy(output,y.contents,w->dims[1]*sizeof(float));return 1;
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
    attn_params ap={m->heads,m->kv_heads,m->key_dim,s->pos,s->context,layer};
    PROFILE_ENCODE(PROFILE_ATTENTION,1,0,[e setComputePipelineState:s->attention];[e setBuffer:s->q offset:0 atIndex:0];[e setBuffer:s->keys offset:0 atIndex:1];[e setBuffer:s->values offset:0 atIndex:2];[e setBuffer:s->att offset:0 atIndex:3];[e setBuffer:s->scores offset:0 atIndex:4];[e setBytes:&ap length:sizeof(ap) atIndex:5];[e setThreadgroupMemoryLength:256*sizeof(float) atIndex:0];[e dispatchThreadgroups:MTLSizeMake(m->heads,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)]);
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
static void encode_attention_decode(dsm_metal *s,
                                    id<MTLComputeCommandEncoder> e,
                                    uint32_t layer, uint32_t pos) {
    const dsm_model *m = s->m;
    attn_params ap = {m->heads, m->kv_heads, m->key_dim, pos, s->context,
                      layer};
    [e setComputePipelineState:s->attention];
    [e setBuffer:s->q offset:0 atIndex:0];
    [e setBuffer:s->keys offset:0 atIndex:1];
    [e setBuffer:s->values offset:0 atIndex:2];
    [e setBuffer:s->att offset:0 atIndex:3];
    [e setBuffer:s->scores offset:0 atIndex:4];
    [e setBytes:&ap length:sizeof(ap) atIndex:5];
    [e setThreadgroupMemoryLength:256 * sizeof(float) atIndex:0];
    [e dispatchThreadgroups:MTLSizeMake(m->heads, 1, 1)
 threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
}
static void encode_layer(dsm_metal *s, id<MTLComputeCommandEncoder> e,
                         uint32_t layer) {
    const dsm_model *m = s->m;
    const dsm_layer *w = &m->layer[layer];
    encode_norm(s, e, s->x, w->attn_norm, s->norm_x, m->dim, 1);
    encode_gemv(s, e, w->q, s->norm_x, s->q);
    encode_gemv(s, e, w->k, s->norm_x, s->k);
    encode_gemv(s, e, w->v, s->norm_x, s->v);
    encode_norm(s, e, s->q, w->q_norm, s->q, m->key_dim, m->heads);
    encode_norm(s, e, s->k, w->k_norm, s->k, m->key_dim, m->kv_heads);
    encode_rope(s, e, s->q, m->heads);
    encode_rope(s, e, s->k, m->kv_heads);
    encode_kv_store(s, e, layer, s->pos);
    encode_attention_decode(s, e, layer, s->pos);
    encode_gemv(s, e, w->o, s->att, s->tmp);
    encode_simple(e, s->add, s->x, s->tmp, m->dim);
    encode_norm(s, e, s->x, w->ffn_norm, s->norm_x, m->dim, 1);
    if (m->is_moe) {
        encode_moe(s, e, w);
    } else {
        encode_gemv(s, e, w->gate, s->norm_x, s->gate);
        encode_gemv(s, e, w->up, s->norm_x, s->up);
        encode_simple(e, s->swiglu, s->gate, s->up, m->hidden);
        encode_gemv(s, e, w->down, s->gate, s->tmp);
        encode_simple(e, s->add, s->x, s->tmp, m->dim);
    }
}
static id<MTLComputePipelineState> matmul_pipeline(const dsm_metal *s, enum ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q8_0: return s->mm_q8;
        case GGML_TYPE_Q2_K: return s->mm_q2k;
        case GGML_TYPE_Q3_K: return s->mm_q3k;
        case GGML_TYPE_Q4_K: return s->mm_q4k;
        case GGML_TYPE_Q6_K: return s->mm_q6k;
        default: return nil;
    }
}
static id<MTLComputePipelineState> matmul_id_pipeline(const dsm_metal *s, enum ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q8_0: return s->mm_id_q8;
        case GGML_TYPE_Q2_K: return s->mm_id_q2k;
        case GGML_TYPE_Q3_K: return s->mm_id_q3k;
        case GGML_TYPE_Q4_K: return s->mm_id_q4k;
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
                                uint32_t output_stride) {
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
    [e setComputePipelineState:s->mm_id_map0];
    [e setBytes:&map length:sizeof(map) atIndex:0];
    [e setBuffer:s->bids offset:0 atIndex:1];
    [e setBuffer:s->mm_id_tpe offset:0 atIndex:2];
    [e setBuffer:s->mm_id_hids offset:0 atIndex:3];
    [e setThreadgroupMemoryLength:(NSUInteger)(2 * m->experts * sizeof(uint32_t)) atIndex:0];
    [e dispatchThreadgroups:MTLSizeMake(1, 1, 1)
 threadsPerThreadgroup:MTLSizeMake(m->experts, 1, 1)];
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
    [e setThreadgroupMemoryLength:8192 atIndex:0];
    [e dispatchThreadgroups:MTLSizeMake(matmul_id_max_tiles(m->experts, m->experts_used, rows),
                                         ((uint32_t)w->dims[1] + 63) / 64, 1)
 threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    return 1;
}
static void encode_matmul(dsm_metal *s, id<MTLComputeCommandEncoder> e,
                          const gguf_tensor *w, id<MTLBuffer> x,
                          id<MTLBuffer> y, uint32_t rows) {
    id<MTLComputePipelineState> p = rows > 8 && w->dims[0] % 32 == 0
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
        [e setThreadgroupMemoryLength:8192 atIndex:0];
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
    [e setComputePipelineState:s->attention_prefill];
    [e setBuffer:s->bq offset:0 atIndex:0];
    [e setBuffer:s->keys offset:0 atIndex:1];
    [e setBuffer:s->values offset:0 atIndex:2];
    [e setBuffer:s->batt offset:0 atIndex:3];
    [e setBytes:&attention length:sizeof(attention) atIndex:4];
    uint32_t q_per_kv = m->heads / m->kv_heads;
    [e setThreadgroupMemoryLength:(NSUInteger)(2 * 16 * 128 + 4 * q_per_kv * 128) * sizeof(float) atIndex:0];
    [e dispatchThreadgroups:MTLSizeMake((rows + 3) / 4, m->kv_heads, 1)
 threadsPerThreadgroup:MTLSizeMake(32, q_per_kv, 1)];
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
                                             rows, 1, m->dim, m->expert_hidden) &&
                        encode_expert_matmul(s, e, w->up_exps, s->bnorm, s->bup,
                                             rows, 1, m->dim, m->expert_hidden);
        if (use_mm_id) {
            [e setComputePipelineState:s->expert_swiglu]; [e setBuffer:s->bgate offset:0 atIndex:0]; [e setBuffer:s->bup offset:0 atIndex:1]; [e setBytes:&m->expert_hidden length:sizeof(m->expert_hidden) atIndex:2]; [e setBytes:&m->experts_used length:sizeof(m->experts_used) atIndex:3]; [e setBytes:&rows length:sizeof(rows) atIndex:4]; dispatch_1d(e, s->expert_swiglu, rows * m->expert_hidden * m->experts_used);
            use_mm_id = encode_expert_matmul(s, e, w->down_exps, s->bgate, s->btmp,
                                              rows, m->experts_used, m->expert_hidden, m->dim);
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
    for (uint32_t layer = 0; layer < m->layers; ++layer) encode_layer_batch(s, e, layer, pos0, rows);
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
    if (s->profile.enabled || !prefill_enabled()) {
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
int dsm_metal_token(dsm_metal *s,int32_t token,float *logits){const dsm_model *m=s->m;if(token<0||(uint32_t)token>=m->vocab||s->pos>=s->context)return 0;const ggml_type_info *info=gguf_type_info(m->token_embd->type);const uint8_t *row=(const uint8_t *)m->token_embd->data+(uint64_t)token*(m->dim/info->block_elems)*info->block_bytes;float *x=s->x.contents;dsm_dequant_row(m->token_embd->type,row,x,m->dim);if(s->profile.enabled)return profile_token(s,logits);
    id<MTLCommandBuffer> cb=[s->queue commandBuffer];id<MTLComputeCommandEncoder> e=[cb computeCommandEncoder];for(uint32_t l=0;l<m->layers;++l)encode_layer(s,e,l);encode_norm(s,e,s->x,m->output_norm,s->norm_x,m->dim,1);encode_gemv(s,e,m->output,s->norm_x,s->logits);[e endEncoding];[cb commit];[cb waitUntilCompleted];if(cb.status!=MTLCommandBufferStatusCompleted)return 0;memcpy(logits,s->logits.contents,m->vocab*4);++s->pos;return 1;}
