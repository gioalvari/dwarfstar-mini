#include "metal_internal.h"

void dsm_mtl_encode_gemv_at(dsm_metal *s, id<MTLComputeCommandEncoder> e, const gguf_tensor *w,
                            id<MTLBuffer> x, NSUInteger x_off, id<MTLBuffer> y, NSUInteger y_off) {
    id<MTLBuffer> wb;
    uint64_t off;
    if (!dsm_mtl_tensor_view(s, w, &wb, &off))
        return;
    /* A 1-D weight (qwen35moe ffn_gate_inp_shexp [dim]) is a one-row matrix;
       the GGUF loader leaves dims[1] = 0 for it. */
    uint64_t rows = w->ndim > 1 ? w->dims[1] : 1;
    gemv_params p = {(uint32_t)w->dims[0], (uint32_t)rows};
    gemv_mul_mv_params mul_mv = {
        .ne00 = (int32_t)w->dims[0],
        .ne01 = (int32_t)rows,
        .ne02 = 1,
        .nb00 = 0,
        .nb01 = w->bytes / rows,
        .nb02 = 0,
        .nb03 = 0,
        .ne10 = (int32_t)w->dims[0],
        .ne11 = 1,
        .ne12 = 1,
        .nb10 = sizeof(float),
        .nb11 = 0,
        .nb12 = 0,
        .nb13 = 0,
        .ne0 = (int32_t)rows,
        .ne1 = 1,
        .nr0 = 0,
        .r2 = 1,
        .r3 = 1,
    };
    id<MTLComputePipelineState> state = s->f32;
    switch (w->type) {
    case GGML_TYPE_Q8_0:
        state = s->q8;
        break;
    case GGML_TYPE_Q2_K:
        state = s->q2k;
        break;
    case GGML_TYPE_Q3_K:
        state = s->q3k;
        break;
    case GGML_TYPE_Q4_K:
        state = s->q4k;
        break;
    case GGML_TYPE_Q5_K:
        state = s->q5k;
        break;
    case GGML_TYPE_Q6_K:
        state = s->q6k;
        break;
    }
    [e setComputePipelineState:state];
    [e setBuffer:wb offset:(NSUInteger)off atIndex:0];
    [e setBuffer:x offset:x_off atIndex:1];
    [e setBuffer:y offset:y_off atIndex:2];
    if (w->type == GGML_TYPE_Q8_0) {
        [e setBytes:&mul_mv length:sizeof(mul_mv) atIndex:3];
        [e setThreadgroupMemoryLength:32 * sizeof(float) * 2 atIndex:0];
        [e dispatchThreadgroups:MTLSizeMake((p.out + 1) / 2, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(32, 4, 1)];
    } else if (w->type == GGML_TYPE_Q2_K) {
        [e setBytes:&mul_mv length:sizeof(mul_mv) atIndex:3];
        [e dispatchThreadgroups:MTLSizeMake((p.out + 7) / 8, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(32, 2, 1)];
    } else if (w->type == GGML_TYPE_Q3_K) {
        [e setBytes:&mul_mv length:sizeof(mul_mv) atIndex:3];
        [e dispatchThreadgroups:MTLSizeMake((p.out + 3) / 4, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(32, 2, 1)];
    } else if (w->type == GGML_TYPE_Q5_K) {
        [e setBytes:&mul_mv length:sizeof(mul_mv) atIndex:3];
        [e dispatchThreadgroups:MTLSizeMake((p.out + 1) / 2, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(32, 2, 1)];
    } else if (w->type == GGML_TYPE_Q4_K || w->type == GGML_TYPE_Q6_K) {
        [e setBytes:&mul_mv length:sizeof(mul_mv) atIndex:3];
        [e dispatchThreadgroups:MTLSizeMake((p.out + 3) / 4, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(32, 2, 1)];
    } else {
        [e setBytes:&p length:sizeof(p) atIndex:3];
        [e setThreadgroupMemoryLength:256 * sizeof(float) atIndex:0];
        [e dispatchThreadgroups:MTLSizeMake(p.out, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    }
}
void dsm_mtl_encode_f32_rows(dsm_metal *s, id<MTLComputeCommandEncoder> e, const gguf_tensor *w,
                             id<MTLBuffer> x, id<MTLBuffer> y, uint32_t rows, uint32_t x_stride,
                             uint32_t y_stride) {
    id<MTLBuffer> wb;
    uint64_t off;
    if (!dsm_mtl_tensor_view(s, w, &wb, &off))
        return;
    gemv_rows_params p = {(uint32_t)w->dims[0], (uint32_t)(w->ndim > 1 ? w->dims[1] : 1), rows,
                          x_stride, y_stride};
    [e setComputePipelineState:s->f32_rows];
    dsm_mtl_set_checked(e, wb, (NSUInteger)off, (NSUInteger)w->bytes, 0);
    dsm_mtl_set_checked(e, x, 0, (NSUInteger)rows * x_stride * sizeof(float), 1);
    dsm_mtl_set_checked(e, y, 0, (NSUInteger)rows * y_stride * sizeof(float), 2);
    [e setBytes:&p length:sizeof(p) atIndex:3];
    [e setThreadgroupMemoryLength:256 * sizeof(float) atIndex:0];
    [e dispatchThreadgroups:MTLSizeMake(p.out, rows, 1)
        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
}
void dsm_mtl_encode_gemv(dsm_metal *s, id<MTLComputeCommandEncoder> e, const gguf_tensor *w,
                         id<MTLBuffer> x, id<MTLBuffer> y) {
    dsm_mtl_encode_gemv_at(s, e, w, x, 0, y, 0);
}
id<MTLComputePipelineState> dsm_mtl_expert_gemv_pipeline(const dsm_metal *s, enum ggml_type type) {
    switch (type) {
    case GGML_TYPE_Q8_0:
        return s->expert_q8_fast;
    case GGML_TYPE_Q3_K:
        return s->expert_q3k;
    case GGML_TYPE_Q4_K:
        return s->expert_q4k;
    case GGML_TYPE_Q5_K:
        return s->expert_q5k;
    case GGML_TYPE_Q6_K:
        return s->expert_q6k;
    default:
        return nil;
    }
}
int dsm_mtl_expert_gemv_supported(enum ggml_type type) {
    return type == GGML_TYPE_Q8_0 || type == GGML_TYPE_Q3_K || type == GGML_TYPE_Q4_K ||
           type == GGML_TYPE_Q5_K || type == GGML_TYPE_Q6_K;
}
static void encode_expert_gemv(dsm_metal *s, id<MTLComputeCommandEncoder> e, const gguf_tensor *w,
                               id<MTLBuffer> x, id<MTLBuffer> y) {
    id<MTLBuffer> wb;
    uint64_t off;
    if (!dsm_mtl_tensor_view(s, w, &wb, &off))
        return;
    id<MTLComputePipelineState> state = dsm_mtl_expert_gemv_pipeline(s, w->type);
    if (!state)
        return;
    expert_gemv_params p = {(uint32_t)w->dims[0], (uint32_t)w->dims[1],
                            (uint32_t)(w->bytes / w->dims[2]), (uint32_t)w->dims[1], 0};
    [e setComputePipelineState:state];
    [e setBuffer:wb offset:(NSUInteger)off atIndex:0];
    [e setBuffer:x offset:0 atIndex:1];
    [e setBuffer:y offset:0 atIndex:2];
    [e setBytes:&p length:sizeof(p) atIndex:3];
    [e setBuffer:s->expert_ids offset:0 atIndex:4];
    if (w->type == GGML_TYPE_Q8_0) {
        [e setThreadgroupMemoryLength:2 * 32 * sizeof(float) atIndex:0];
        [e dispatchThreadgroups:MTLSizeMake((p.out + 1) / 2, 1, s->m->experts_used)
            threadsPerThreadgroup:MTLSizeMake(32, 4, 1)];
    } else if (w->type == GGML_TYPE_Q3_K || w->type == GGML_TYPE_Q4_K ||
               w->type == GGML_TYPE_Q5_K) {
        [e dispatchThreadgroups:MTLSizeMake((p.out + 3) / 4, 1, s->m->experts_used)
            threadsPerThreadgroup:MTLSizeMake(32, 2, 1)];
    } else {
        [e setThreadgroupMemoryLength:256 * sizeof(float) atIndex:0];
        [e dispatchThreadgroups:MTLSizeMake(p.out, 1, s->m->experts_used)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    }
}
void dsm_mtl_encode_moe_router(dsm_metal *s, id<MTLComputeCommandEncoder> e, const dsm_layer *w) {
    dsm_mtl_encode_gemv(s, e, w->gate_inp, s->norm_x, s->route);
}
void dsm_mtl_encode_expert_gemv_batch_row(dsm_metal *s, id<MTLComputeCommandEncoder> e,
                                          const gguf_tensor *w, id<MTLBuffer> x, NSUInteger x_off,
                                          id<MTLBuffer> y, NSUInteger y_off, uint32_t row,
                                          uint32_t input_stride, uint32_t output_stride) {
    id<MTLBuffer> wb;
    uint64_t off;
    if (!dsm_mtl_tensor_view(s, w, &wb, &off))
        return;
    id<MTLComputePipelineState> state = dsm_mtl_expert_gemv_pipeline(s, w->type);
    if (!state)
        return;
    expert_gemv_params p = {(uint32_t)w->dims[0], (uint32_t)w->dims[1],
                            (uint32_t)(w->bytes / w->dims[2]), output_stride, input_stride};
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
    } else if (w->type == GGML_TYPE_Q3_K || w->type == GGML_TYPE_Q4_K ||
               w->type == GGML_TYPE_Q5_K) {
        [e dispatchThreadgroups:MTLSizeMake((p.out + 3) / 4, 1, s->m->experts_used)
            threadsPerThreadgroup:MTLSizeMake(32, 2, 1)];
    } else {
        [e setThreadgroupMemoryLength:256 * sizeof(float) atIndex:0];
        [e dispatchThreadgroups:MTLSizeMake(p.out, 1, s->m->experts_used)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    }
}
void dsm_mtl_encode_moe_topk(dsm_metal *s, id<MTLComputeCommandEncoder> e) {
    const dsm_model *m = s->m;
    [e setComputePipelineState:s->route_topk];
    [e setBuffer:s->route offset:0 atIndex:0];
    [e setBuffer:s->expert_ids offset:0 atIndex:1];
    [e setBuffer:s->expert_weights offset:0 atIndex:2];
    [e setBytes:&m->experts length:sizeof(m->experts) atIndex:3];
    [e setBytes:&m->experts_used length:sizeof(m->experts_used) atIndex:4];
    [e setThreadgroupMemoryLength:(m->experts + 8) * sizeof(float) atIndex:0];
    [e dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
}
void dsm_mtl_encode_moe_topk_batch(dsm_metal *s, id<MTLComputeCommandEncoder> e, uint32_t rows) {
    const dsm_model *m = s->m;
    [e setComputePipelineState:s->route_topk];
    [e setBuffer:s->broute offset:0 atIndex:0];
    [e setBuffer:s->bids offset:0 atIndex:1];
    [e setBuffer:s->bw offset:0 atIndex:2];
    [e setBytes:&m->experts length:sizeof(m->experts) atIndex:3];
    [e setBytes:&m->experts_used length:sizeof(m->experts_used) atIndex:4];
    [e setThreadgroupMemoryLength:(m->experts + 8) * sizeof(float) atIndex:0];
    [e dispatchThreadgroups:MTLSizeMake(rows, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
}
void dsm_mtl_encode_moe_gate_up(dsm_metal *s, id<MTLComputeCommandEncoder> e, const dsm_layer *w) {
    const dsm_model *m = s->m;
    encode_expert_gemv(s, e, w->gate_exps, s->norm_x, s->gate);
    encode_expert_gemv(s, e, w->up_exps, s->norm_x, s->up);
    uint32_t rows = 1;
    [e setComputePipelineState:s->expert_swiglu];
    [e setBuffer:s->gate offset:0 atIndex:0];
    [e setBuffer:s->up offset:0 atIndex:1];
    [e setBytes:&m->expert_hidden length:sizeof(m->expert_hidden) atIndex:2];
    [e setBytes:&m->experts_used length:sizeof(m->experts_used) atIndex:3];
    [e setBytes:&rows length:sizeof(rows) atIndex:4];
    dsm_mtl_dispatch_1d(e, s->expert_swiglu, m->expert_hidden * m->experts_used);
}
void dsm_mtl_encode_moe_down(dsm_metal *s, id<MTLComputeCommandEncoder> e, const dsm_layer *w) {
    const dsm_model *m = s->m;
    id<MTLBuffer> wb;
    uint64_t off;
    if (!dsm_mtl_tensor_view(s, w->down_exps, &wb, &off))
        return;
    expert_gemv_params p = {(uint32_t)w->down_exps->dims[0], (uint32_t)w->down_exps->dims[1],
                            (uint32_t)(w->down_exps->bytes / w->down_exps->dims[2]), m->dim,
                            m->expert_hidden};
    id<MTLComputePipelineState> state = dsm_mtl_expert_gemv_pipeline(s, w->down_exps->type);
    if (!state)
        return;
    [e setComputePipelineState:state];
    [e setBuffer:wb offset:(NSUInteger)off atIndex:0];
    [e setBuffer:s->gate offset:0 atIndex:1];
    [e setBuffer:s->tmp offset:0 atIndex:2];
    [e setBytes:&p length:sizeof(p) atIndex:3];
    [e setBuffer:s->expert_ids offset:0 atIndex:4];
    if (w->down_exps->type == GGML_TYPE_Q8_0) {
        [e setThreadgroupMemoryLength:2 * 32 * sizeof(float) atIndex:0];
        [e dispatchThreadgroups:MTLSizeMake((p.out + 1) / 2, 1, m->experts_used)
            threadsPerThreadgroup:MTLSizeMake(32, 4, 1)];
    } else if (w->down_exps->type == GGML_TYPE_Q3_K || w->down_exps->type == GGML_TYPE_Q4_K ||
               w->down_exps->type == GGML_TYPE_Q5_K) {
        [e dispatchThreadgroups:MTLSizeMake((p.out + 3) / 4, 1, m->experts_used)
            threadsPerThreadgroup:MTLSizeMake(32, 2, 1)];
    } else {
        [e setThreadgroupMemoryLength:256 * sizeof(float) atIndex:0];
        [e dispatchThreadgroups:MTLSizeMake(p.out, 1, m->experts_used)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    }
}
void dsm_mtl_encode_moe_accumulate(dsm_metal *s, id<MTLComputeCommandEncoder> e) {
    const dsm_model *m = s->m;
    [e setComputePipelineState:s->moe_add];
    [e setBuffer:s->x offset:0 atIndex:0];
    [e setBuffer:s->tmp offset:0 atIndex:1];
    [e setBuffer:s->expert_weights offset:0 atIndex:2];
    [e setBytes:&m->experts_used length:sizeof(m->experts_used) atIndex:3];
    [e setBytes:&m->dim length:sizeof(m->dim) atIndex:4];
    dsm_mtl_dispatch_1d(e, s->moe_add, m->dim);
}
static void encode_moe(dsm_metal *s, id<MTLComputeCommandEncoder> e, const dsm_layer *w) {
    dsm_mtl_encode_moe_router(s, e, w);
    dsm_mtl_encode_moe_topk(s, e);
    dsm_mtl_encode_moe_gate_up(s, e, w);
    dsm_mtl_encode_moe_down(s, e, w);
    dsm_mtl_encode_moe_accumulate(s, e);
}
void dsm_mtl_encode_norm_at(dsm_metal *s, id<MTLComputeCommandEncoder> e, id<MTLBuffer> x,
                            NSUInteger x_off, const gguf_tensor *w, id<MTLBuffer> out,
                            NSUInteger out_off, uint32_t width, uint32_t groups) {
    id<MTLBuffer> wb;
    uint64_t off;
    if (!dsm_mtl_tensor_view(s, w, &wb, &off))
        return;
    norm_params p = {width, groups, s->m->rms_eps};
    [e setComputePipelineState:s->norm];
    [e setBuffer:x offset:x_off atIndex:0];
    [e setBuffer:wb offset:(NSUInteger)off atIndex:1];
    [e setBuffer:out offset:out_off atIndex:2];
    [e setBytes:&p length:sizeof(p) atIndex:3];
    [e setThreadgroupMemoryLength:256 * sizeof(float) atIndex:0];
    [e dispatchThreadgroups:MTLSizeMake(groups, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
}
void dsm_mtl_encode_norm(dsm_metal *s, id<MTLComputeCommandEncoder> e, id<MTLBuffer> x,
                         const gguf_tensor *w, id<MTLBuffer> out, uint32_t width, uint32_t groups) {
    dsm_mtl_encode_norm_at(s, e, x, 0, w, out, 0, width, groups);
}
void dsm_mtl_encode_rope(dsm_metal *s, id<MTLComputeCommandEncoder> e, id<MTLBuffer> x,
                         uint32_t heads) {
    rope_params p = {heads, s->m->key_dim, s->pos, s->m->rope_base};
    [e setComputePipelineState:s->rope];
    [e setBuffer:x offset:0 atIndex:0];
    [e setBytes:&p length:sizeof(p) atIndex:1];
    dsm_mtl_dispatch_1d(e, s->rope, heads * (s->m->key_dim / 2));
}
void dsm_mtl_encode_simple(id<MTLComputeCommandEncoder> e, id<MTLComputePipelineState> p,
                           id<MTLBuffer> a, id<MTLBuffer> b, uint32_t n) {
    [e setComputePipelineState:p];
    [e setBuffer:a offset:0 atIndex:0];
    [e setBuffer:b offset:0 atIndex:1];
    [e setBytes:&n length:4 atIndex:2];
    dsm_mtl_dispatch_1d(e, p, n);
}
static void encode_qk_norm_rope(dsm_metal *s, id<MTLComputeCommandEncoder> e, const dsm_layer *w,
                                uint32_t layer) {
    id<MTLBuffer> qw, kw;
    uint64_t qoff, koff;
    if (!dsm_mtl_tensor_view(s, w->q_norm, &qw, &qoff) ||
        !dsm_mtl_tensor_view(s, w->k_norm, &kw, &koff))
        return;
    qk_norm_rope_params p = {s->m->heads, s->m->kv_heads, s->m->key_dim,   s->pos,
                             s->context,  layer,          s->m->rope_base, s->m->rms_eps};
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
    if (!dsm_mtl_tensor_view(s, weight, &wb, &off))
        return;
    norm_params p = {s->m->dim, 1, s->m->rms_eps};
    [e setComputePipelineState:s->add_rmsnorm];
    [e setBuffer:s->x offset:0 atIndex:0];
    [e setBuffer:s->tmp offset:0 atIndex:1];
    [e setBuffer:wb offset:(NSUInteger)off atIndex:2];
    [e setBuffer:s->norm_x offset:0 atIndex:3];
    [e setBytes:&p length:sizeof(p) atIndex:4];
    [e setThreadgroupMemoryLength:256 * sizeof(float) atIndex:0];
    [e dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
}

static void encode_kv_store(dsm_metal *s, id<MTLComputeCommandEncoder> e, uint32_t layer,
                            uint32_t pos) {
    uint32_t kvw = s->m->kv_heads * s->m->key_dim;
    cache_params cp = {layer, pos, s->context, kvw};
    [e setComputePipelineState:s->cache];
    [e setBuffer:s->k offset:0 atIndex:0];
    [e setBuffer:s->keys offset:0 atIndex:1];
    [e setBytes:&cp length:sizeof(cp) atIndex:2];
    dsm_mtl_dispatch_1d(e, s->cache, kvw);
    [e setBuffer:s->v offset:0 atIndex:0];
    [e setBuffer:s->values offset:0 atIndex:1];
    [e setBytes:&cp length:sizeof(cp) atIndex:2];
    dsm_mtl_dispatch_1d(e, s->cache, kvw);
}
static void encode_v_store(dsm_metal *s, id<MTLComputeCommandEncoder> e, uint32_t layer,
                           uint32_t pos) {
    uint32_t kvw = s->m->kv_heads * s->m->key_dim;
    cache_params cp = {layer, pos, s->context, kvw};
    [e setComputePipelineState:s->cache];
    [e setBuffer:s->v offset:0 atIndex:0];
    [e setBuffer:s->values offset:0 atIndex:1];
    [e setBytes:&cp length:sizeof(cp) atIndex:2];
    dsm_mtl_dispatch_1d(e, s->cache, kvw);
}
void dsm_mtl_encode_attention_decode_at(dsm_metal *s, id<MTLComputeCommandEncoder> e,
                                        uint32_t layer, uint32_t pos, id<MTLBuffer> q,
                                        NSUInteger q_off, id<MTLBuffer> out, NSUInteger out_off) {
    const dsm_model *m = s->m;
    attn_params ap = {m->heads, m->kv_heads, m->key_dim, pos, s->context, layer};
    if (dsm_mtl_attention_decode_enabled()) {
        uint32_t splits = (pos + 64) / 64;
        attn_split_params split = {m->heads,   m->kv_heads, m->key_dim, pos,
                                   s->context, layer,       splits};
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
void dsm_mtl_encode_attention_decode(dsm_metal *s, id<MTLComputeCommandEncoder> e, uint32_t layer,
                                     uint32_t pos) {
    dsm_mtl_encode_attention_decode_at(s, e, layer, pos, s->q, 0, s->att, 0);
}
static void encode_partial_rope(dsm_metal *s, id<MTLComputeCommandEncoder> e, id<MTLBuffer> x,
                                uint32_t heads) {
    rope_partial_params p = {heads, s->m->key_dim, s->m->rope_dims, s->pos, s->m->rope_base};
    [e setComputePipelineState:s->rope_partial];
    [e setBuffer:x offset:0 atIndex:0];
    [e setBytes:&p length:sizeof(p) atIndex:1];
    dsm_mtl_dispatch_1d(e, s->rope_partial, heads * (s->m->rope_dims / 2));
}
static void encode_hybrid_moe(dsm_metal *s, id<MTLComputeCommandEncoder> e, const dsm_layer *w) {
    const dsm_model *m = s->m;
    encode_moe(s, e, w);
    dsm_mtl_encode_gemv(s, e, w->gate_shexp, s->norm_x, s->gate);
    dsm_mtl_encode_gemv(s, e, w->up_shexp, s->norm_x, s->up);
    dsm_mtl_encode_simple(e, s->swiglu, s->gate, s->up, m->shared_hidden);
    dsm_mtl_encode_gemv(s, e, w->down_shexp, s->gate, s->tmp);
    dsm_mtl_encode_gemv(s, e, w->gate_inp_shexp, s->norm_x, s->shared_gate);
    [e setComputePipelineState:s->shexp_add];
    [e setBuffer:s->x offset:0 atIndex:0];
    [e setBuffer:s->tmp offset:0 atIndex:1];
    [e setBuffer:s->shared_gate offset:0 atIndex:2];
    [e setBytes:&m->dim length:sizeof(m->dim) atIndex:3];
    dsm_mtl_dispatch_1d(e, s->shexp_add, m->dim);
}
static void encode_hybrid_dense_ffn(dsm_metal *s, id<MTLComputeCommandEncoder> e,
                                    const dsm_layer *w) {
    const dsm_model *m = s->m;
    dsm_mtl_encode_gemv(s, e, w->gate, s->norm_x, s->gate);
    dsm_mtl_encode_gemv(s, e, w->up, s->norm_x, s->up);
    dsm_mtl_encode_simple(e, s->swiglu, s->gate, s->up, m->hidden);
    dsm_mtl_encode_gemv(s, e, w->down, s->gate, s->tmp);
    dsm_mtl_encode_simple(e, s->add, s->x, s->tmp, m->dim);
}
static void encode_layer_hybrid(dsm_metal *s, id<MTLComputeCommandEncoder> e, uint32_t layer) {
    const dsm_model *m = s->m;
    const dsm_layer *w = &m->layer[layer];
    dsm_mtl_encode_norm(s, e, s->x, w->attn_norm, s->norm_x, m->dim, 1);
    if (m->is_recurrent[layer]) {
        uint32_t qk_width = m->ssm_groups * m->ssm_state;
        uint32_t mixed_width = m->ssm_inner + 2 * qk_width;
        dsm_mtl_encode_gemv(s, e, w->qkv, s->norm_x, s->qkv_mixed);
        dsm_mtl_encode_gemv(s, e, w->z, s->norm_x, s->z);
        dsm_mtl_encode_gemv(s, e, w->alpha, s->norm_x, s->alpha);
        dsm_mtl_encode_gemv(s, e, w->beta, s->norm_x, s->beta);
        ssm_conv_params conv = {mixed_width, m->ssm_conv - 1};
        [e setComputePipelineState:s->ssm_conv];
        dsm_mtl_set_checked(e, s->qkv_mixed, 0, (NSUInteger)mixed_width * sizeof(float), 0);
        dsm_mtl_set_checked(e, s->conv_state,
                            (NSUInteger)layer * mixed_width * (m->ssm_conv - 1) * sizeof(float),
                            (NSUInteger)mixed_width * (m->ssm_conv - 1) * sizeof(float), 1);
        id<MTLBuffer> wb;
        uint64_t off;
        if (!dsm_mtl_tensor_view(s, w->conv1d, &wb, &off))
            return;
        dsm_mtl_set_checked(e, wb, (NSUInteger)off, (NSUInteger)w->conv1d->bytes, 2);
        dsm_mtl_set_checked(e, s->conv_out, 0, (NSUInteger)mixed_width * sizeof(float), 3);
        [e setBytes:&conv length:sizeof(conv) atIndex:4];
        dsm_mtl_dispatch_1d(e, s->ssm_conv, mixed_width);
        [e setComputePipelineState:s->split_gdn_qk];
        dsm_mtl_set_checked(e, s->conv_out, 0, (NSUInteger)mixed_width * sizeof(float), 0);
        dsm_mtl_set_checked(e, s->q, 0, (NSUInteger)qk_width * sizeof(float), 1);
        dsm_mtl_set_checked(e, s->k, 0, (NSUInteger)qk_width * sizeof(float), 2);
        [e setBytes:&qk_width length:sizeof(qk_width) atIndex:3];
        dsm_mtl_dispatch_1d(e, s->split_gdn_qk, qk_width);
        norm_params l2 = {m->ssm_state, m->ssm_groups * 2, m->rms_eps};
        [e setComputePipelineState:s->l2norm];
        dsm_mtl_set_checked(e, s->q, 0, (NSUInteger)qk_width * sizeof(float), 0);
        [e setBytes:&l2 length:sizeof(l2) atIndex:1];
        [e setThreadgroupMemoryLength:256 * sizeof(float) atIndex:0];
        [e dispatchThreadgroups:MTLSizeMake(m->ssm_groups, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [e setComputePipelineState:s->l2norm];
        dsm_mtl_set_checked(e, s->k, 0, (NSUInteger)qk_width * sizeof(float), 0);
        [e setBytes:&l2 length:sizeof(l2) atIndex:1];
        [e setThreadgroupMemoryLength:256 * sizeof(float) atIndex:0];
        [e dispatchThreadgroups:MTLSizeMake(m->ssm_groups, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        if (!dsm_mtl_tensor_view(s, w->dt_bias, &wb, &off))
            return;
        [e setComputePipelineState:s->gdn_gates];
        [e setBuffer:s->alpha offset:0 atIndex:0];
        [e setBuffer:s->beta offset:0 atIndex:1];
        [e setBuffer:wb offset:(NSUInteger)off atIndex:2];
        if (!dsm_mtl_tensor_view(s, w->ssm_a, &wb, &off))
            return;
        [e setBuffer:wb offset:(NSUInteger)off atIndex:3];
        dsm_mtl_dispatch_1d(e, s->gdn_gates, m->ssm_vheads);
        gdn_params gdn = {m->ssm_groups, m->ssm_vheads, m->ssm_state, layer, m->rms_eps};
        [e setComputePipelineState:s->gdn];
        [e setBuffer:s->q offset:0 atIndex:0];
        [e setBuffer:s->k offset:0 atIndex:1];
        [e setBuffer:s->conv_out offset:(NSUInteger)(2 * qk_width) * sizeof(float) atIndex:2];
        [e setBuffer:s->alpha offset:0 atIndex:3];
        [e setBuffer:s->beta offset:0 atIndex:4];
        [e setBuffer:s->ssm_state offset:0 atIndex:5];
        [e setBuffer:s->att offset:0 atIndex:6];
        [e setBytes:&gdn length:sizeof(gdn) atIndex:7];
        [e dispatchThreadgroups:MTLSizeMake((m->ssm_state + 3) / 4, m->ssm_vheads, 1)
            threadsPerThreadgroup:MTLSizeMake(32, 4, 1)];
        norm_params gated = {m->ssm_state, m->ssm_vheads, m->rms_eps};
        if (!dsm_mtl_tensor_view(s, w->ssm_norm, &wb, &off))
            return;
        [e setComputePipelineState:s->gated_rmsnorm];
        [e setBuffer:s->att offset:0 atIndex:0];
        [e setBuffer:s->z offset:0 atIndex:1];
        [e setBuffer:wb offset:(NSUInteger)off atIndex:2];
        [e setBytes:&gated length:sizeof(gated) atIndex:3];
        [e setThreadgroupMemoryLength:256 * sizeof(float) atIndex:0];
        [e dispatchThreadgroups:MTLSizeMake(m->ssm_vheads, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        dsm_mtl_encode_gemv(s, e, w->ssm_out, s->att, s->tmp);
    } else {
        dsm_mtl_encode_gemv(s, e, w->q, s->norm_x, s->qg);
        dsm_mtl_encode_gemv(s, e, w->k, s->norm_x, s->k);
        dsm_mtl_encode_gemv(s, e, w->v, s->norm_x, s->v);
        [e setComputePipelineState:s->split_q_gate];
        [e setBuffer:s->qg offset:0 atIndex:0];
        [e setBuffer:s->q offset:0 atIndex:1];
        [e setBuffer:s->gate offset:0 atIndex:2];
        [e setBytes:&m->key_dim length:sizeof(m->key_dim) atIndex:3];
        dsm_mtl_dispatch_1d(e, s->split_q_gate, m->q_width);
        dsm_mtl_encode_norm(s, e, s->q, w->q_norm, s->q, m->key_dim, m->heads);
        dsm_mtl_encode_norm(s, e, s->k, w->k_norm, s->k, m->key_dim, m->kv_heads);
        encode_partial_rope(s, e, s->q, m->heads);
        encode_partial_rope(s, e, s->k, m->kv_heads);
        encode_kv_store(s, e, m->attn_index[layer], s->pos);
        dsm_mtl_encode_attention_decode(s, e, m->attn_index[layer], s->pos);
        [e setComputePipelineState:s->sigmoid_mul];
        [e setBuffer:s->att offset:0 atIndex:0];
        [e setBuffer:s->gate offset:0 atIndex:1];
        [e setBytes:&m->attn_width length:sizeof(m->attn_width) atIndex:2];
        dsm_mtl_dispatch_1d(e, s->sigmoid_mul, m->attn_width);
        dsm_mtl_encode_gemv(s, e, w->o, s->att, s->tmp);
    }
    dsm_mtl_encode_simple(e, s->add, s->x, s->tmp, m->dim);
    dsm_mtl_encode_norm(s, e, s->x, w->ffn_norm, s->norm_x, m->dim, 1);
    if (m->is_moe)
        encode_hybrid_moe(s, e, w);
    else
        encode_hybrid_dense_ffn(s, e, w);
}
static void encode_layer(dsm_metal *s, id<MTLComputeCommandEncoder> e, uint32_t layer) {
    const dsm_model *m = s->m;
    const dsm_layer *w = &m->layer[layer];
    dsm_mtl_encode_norm(s, e, s->x, w->attn_norm, s->norm_x, m->dim, 1);
    dsm_mtl_encode_gemv(s, e, w->q, s->norm_x, s->q);
    dsm_mtl_encode_gemv(s, e, w->k, s->norm_x, s->k);
    dsm_mtl_encode_gemv(s, e, w->v, s->norm_x, s->v);
    if (s->decode_fuse) {
        encode_qk_norm_rope(s, e, w, layer);
        encode_v_store(s, e, layer, s->pos);
    } else {
        dsm_mtl_encode_norm(s, e, s->q, w->q_norm, s->q, m->key_dim, m->heads);
        dsm_mtl_encode_norm(s, e, s->k, w->k_norm, s->k, m->key_dim, m->kv_heads);
        dsm_mtl_encode_rope(s, e, s->q, m->heads);
        dsm_mtl_encode_rope(s, e, s->k, m->kv_heads);
        encode_kv_store(s, e, layer, s->pos);
    }
    dsm_mtl_encode_attention_decode(s, e, layer, s->pos);
    dsm_mtl_encode_gemv(s, e, w->o, s->att, s->tmp);
    if (s->decode_fuse)
        encode_add_rmsnorm(s, e, w->ffn_norm);
    else {
        dsm_mtl_encode_simple(e, s->add, s->x, s->tmp, m->dim);
        dsm_mtl_encode_norm(s, e, s->x, w->ffn_norm, s->norm_x, m->dim, 1);
    }
    if (m->is_moe) {
        encode_moe(s, e, w);
    } else {
        dsm_mtl_encode_gemv(s, e, w->gate, s->norm_x, s->gate);
        dsm_mtl_encode_gemv(s, e, w->up, s->norm_x, s->up);
        dsm_mtl_encode_simple(e, s->swiglu, s->gate, s->up, m->hidden);
        dsm_mtl_encode_gemv(s, e, w->down, s->gate, s->tmp);
        if (s->decode_fuse) {
            const gguf_tensor *weight =
                layer + 1 < m->layers ? m->layer[layer + 1].attn_norm : m->output_norm;
            encode_add_rmsnorm(s, e, weight);
        } else {
            dsm_mtl_encode_simple(e, s->add, s->x, s->tmp, m->dim);
        }
    }
}
int dsm_metal_token(dsm_metal *s, int32_t token, float *logits) {
    const dsm_model *m = s->m;
    if (token < 0 || (uint32_t)token >= m->vocab || s->pos >= s->context)
        return 0;
    const ggml_type_info *info = gguf_type_info(m->token_embd->type);
    const uint8_t *row = (const uint8_t *)m->token_embd->data +
                         (uint64_t)token * (m->dim / info->block_elems) * info->block_bytes;
    float *x = s->x.contents;
    dsm_dequant_row(m->token_embd->type, row, x, m->dim);
    if (s->profile.enabled && !dsm_mtl_qwen35_arch(m))
        return dsm_mtl_profile_token(s, logits);
    if (dsm_mtl_qwen35_arch(m)) {
        if (getenv("DSM_DUMP_DIR")) {
            for (uint32_t l = 0; l < m->layers; ++l) {
                id<MTLCommandBuffer> cb = [s->queue commandBuffer];
                id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
                encode_layer_hybrid(s, e, l);
                [e endEncoding];
                [cb commit];
                [cb waitUntilCompleted];
                if (cb.status != MTLCommandBufferStatusCompleted)
                    return 0;
                dsm_mtl_dump_hybrid_layer(s, l);
            }
            id<MTLCommandBuffer> cb = [s->queue commandBuffer];
            id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
            dsm_mtl_encode_norm(s, e, s->x, m->output_norm, s->norm_x, m->dim, 1);
            dsm_mtl_encode_gemv(s, e, m->output, s->norm_x, s->logits);
            [e endEncoding];
            [cb commit];
            [cb waitUntilCompleted];
            if (cb.status != MTLCommandBufferStatusCompleted)
                return 0;
        } else {
            id<MTLCommandBuffer> cb = [s->queue commandBuffer];
            id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
            for (uint32_t l = 0; l < m->layers; ++l)
                encode_layer_hybrid(s, e, l);
            dsm_mtl_encode_norm(s, e, s->x, m->output_norm, s->norm_x, m->dim, 1);
            dsm_mtl_encode_gemv(s, e, m->output, s->norm_x, s->logits);
            [e endEncoding];
            [cb commit];
            [cb waitUntilCompleted];
            if (cb.status != MTLCommandBufferStatusCompleted)
                return 0;
        }
        memcpy(logits, s->logits.contents, m->vocab * 4);
        ++s->pos;
        return 1;
    }
    id<MTLCommandBuffer> cb = [s->queue commandBuffer];
    id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
    for (uint32_t l = 0; l < m->layers; ++l)
        encode_layer(s, e, l);
    if (!s->decode_fuse || m->is_moe)
        dsm_mtl_encode_norm(s, e, s->x, m->output_norm, s->norm_x, m->dim, 1);
    dsm_mtl_encode_gemv(s, e, m->output, s->norm_x, s->logits);
    [e endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    if (cb.status != MTLCommandBufferStatusCompleted)
        return 0;
    memcpy(logits, s->logits.contents, m->vocab * 4);
    ++s->pos;
    return 1;
}
