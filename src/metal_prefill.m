#include "metal_internal.h"

static int mm_type_index(enum ggml_type type) {
    switch (type) {
    case GGML_TYPE_Q8_0:
        return 0;
    case GGML_TYPE_Q2_K:
        return 1;
    case GGML_TYPE_Q3_K:
        return 2;
    case GGML_TYPE_Q4_K:
        return 3;
    case GGML_TYPE_Q5_K:
        return 4;
    case GGML_TYPE_Q6_K:
        return 5;
    default:
        return -1;
    }
}
/* FP32 staging needs 2048 + 1024 floats of threadgroup memory; half needs 8 KiB
   (also the size of the partial-tile output staging). */
static NSUInteger matmul_tg_bytes(const dsm_metal *s) {
    return s->mm_f32 || s->mm_f32w_enabled ? 12288 : 8192;
}
id<MTLComputePipelineState> dsm_mtl_matmul_pipeline(const dsm_metal *s, enum ggml_type type) {
    if (type == GGML_TYPE_F32)
        return s->mm_f32w_enabled ? s->mm_f32w : nil;
    if (s->mm_f32) {
        int i = mm_type_index(type);
        return i < 0 ? nil : s->mmf[i];
    }
    switch (type) {
    case GGML_TYPE_Q8_0:
        return s->mm_q8;
    case GGML_TYPE_Q2_K:
        return s->mm_q2k;
    case GGML_TYPE_Q3_K:
        return s->mm_q3k;
    case GGML_TYPE_Q4_K:
        return s->mm_q4k;
    case GGML_TYPE_Q5_K:
        return s->mm_q5k;
    case GGML_TYPE_Q6_K:
        return s->mm_q6k;
    default:
        return nil;
    }
}
id<MTLComputePipelineState> dsm_mtl_matmul_id_pipeline(const dsm_metal *s, enum ggml_type type) {
    if (s->mm_f32) {
        int i = mm_type_index(type);
        return i < 0 ? nil : s->mmf_id[i];
    }
    switch (type) {
    case GGML_TYPE_Q8_0:
        return s->mm_id_q8;
    case GGML_TYPE_Q2_K:
        return s->mm_id_q2k;
    case GGML_TYPE_Q3_K:
        return s->mm_id_q3k;
    case GGML_TYPE_Q4_K:
        return s->mm_id_q4k;
    case GGML_TYPE_Q5_K:
        return s->mm_id_q5k;
    case GGML_TYPE_Q6_K:
        return s->mm_id_q6k;
    default:
        return nil;
    }
}
uint32_t dsm_mtl_matmul_id_max_tiles(uint32_t experts, uint32_t slots, uint32_t rows) {
    uint32_t entries = slots * rows;
    uint32_t tiles = (entries + 31) / 32 + (experts < entries ? experts : entries) - 1;
    return entries < tiles ? entries : tiles;
}
int dsm_mtl_encode_expert_matmul(dsm_metal *s, id<MTLComputeCommandEncoder> e, const gguf_tensor *w,
                                 id<MTLBuffer> x, id<MTLBuffer> y, uint32_t rows,
                                 uint32_t input_rows, uint32_t input_stride, uint32_t output_stride,
                                 int build_map) {
    const dsm_model *m = s->m;
    (void)output_stride;
    id<MTLComputePipelineState> p =
        rows > 8 && w->dims[0] % 32 == 0 ? dsm_mtl_matmul_id_pipeline(s, w->type) : nil;
    if (!p)
        return 0;
    id<MTLBuffer> wb;
    uint64_t off;
    if (!dsm_mtl_tensor_view(s, w, &wb, &off))
        return 0;
    mul_mm_id_map0_params map = {
        .ne10 = (int32_t)m->experts,
        .ne11 = (int32_t)input_rows,
        .nb11 = input_stride * sizeof(float),
        .nb12 = (uint64_t)input_rows * input_stride * sizeof(float),
        .ne21 = (int32_t)rows,
        .ne20 = (int32_t)m->experts_used,
        .nb21 = m->experts_used * sizeof(uint32_t),
    };
    /* The token->expert tile map depends only on the routing ids, so callers
       build it once per layer and reuse it for gate, up and down. */
    if (!build_map)
        goto skip_map0;
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
        .ne00 = (int32_t)w->dims[0],
        .ne02 = (int32_t)m->experts,
        .nb01 = w->bytes / w->dims[1] / m->experts,
        .nb02 = w->bytes / m->experts,
        .nb03 = 0,
        .ne11 = (int32_t)input_rows,
        .nb10 = sizeof(float),
        .nb11 = input_stride * sizeof(float),
        .nb12 = (uint64_t)input_rows * input_stride * sizeof(float),
        .nb13 = 0,
        .ne20 = (int32_t)m->experts_used,
        .ne0 = (int32_t)w->dims[1],
        .ne1 = (int32_t)m->experts_used,
        .r2 = 1,
        .r3 = 1,
    };
    [e setComputePipelineState:p];
    [e setBytes:&args length:sizeof(args) atIndex:0];
    [e setBuffer:wb offset:(NSUInteger)off atIndex:1];
    [e setBuffer:x offset:0 atIndex:2];
    [e setBuffer:s->mm_id_tpe offset:0 atIndex:3];
    [e setBuffer:s->mm_id_hids offset:0 atIndex:4];
    [e setBuffer:y offset:0 atIndex:5];
    [e setThreadgroupMemoryLength:matmul_tg_bytes(s) atIndex:0];
    [e dispatchThreadgroups:MTLSizeMake(
                                dsm_mtl_matmul_id_max_tiles(m->experts, m->experts_used, rows),
                                ((uint32_t)w->dims[1] + 63) / 64, 1)
        threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    return 1;
}
void dsm_mtl_encode_matmul(dsm_metal *s, id<MTLComputeCommandEncoder> e, const gguf_tensor *w,
                           id<MTLBuffer> x, id<MTLBuffer> y, uint32_t rows) {
    id<MTLComputePipelineState> p = rows > 8 && w->ndim == 2 && w->dims[0] % 32 == 0
                                        ? dsm_mtl_matmul_pipeline(s, w->type)
                                        : nil;
    if (p) {
        assert(w->dims[0] % 32 == 0);
        id<MTLBuffer> wb;
        uint64_t off;
        if (!dsm_mtl_tensor_view(s, w, &wb, &off))
            return;
        mul_mm_params args = {
            .ne00 = (int32_t)w->dims[0],
            .ne02 = 1,
            .nb01 = w->bytes / w->dims[1],
            .nb02 = 0,
            .nb03 = 0,
            .ne12 = 1,
            .nb10 = sizeof(float),
            .nb11 = w->dims[0] * sizeof(float),
            .nb12 = 0,
            .nb13 = 0,
            .ne0 = (int32_t)w->dims[1],
            .ne1 = (int32_t)rows,
            .r2 = 1,
            .r3 = 1,
        };
        [e setComputePipelineState:p];
        [e setBytes:&args length:sizeof(args) atIndex:0];
        [e setBuffer:wb offset:(NSUInteger)off atIndex:1];
        [e setBuffer:x offset:0 atIndex:2];
        [e setBuffer:y offset:0 atIndex:3];
        [e setThreadgroupMemoryLength:matmul_tg_bytes(s) atIndex:0];
        [e dispatchThreadgroups:MTLSizeMake((rows + 31) / 32, ((uint32_t)w->dims[1] + 63) / 64, 1)
            threadsPerThreadgroup:MTLSizeMake(32, 4, 1)];
        return;
    }
    for (uint32_t row = 0; row < rows; ++row) {
        dsm_mtl_encode_gemv_at(s, e, w, x, (NSUInteger)row * w->dims[0] * sizeof(float), y,
                               (NSUInteger)row * w->dims[1] * sizeof(float));
    }
}
static void encode_layer_batch(dsm_metal *s, id<MTLComputeCommandEncoder> e, uint32_t layer,
                               uint32_t pos0, uint32_t rows) {
    const dsm_model *m = s->m;
    const dsm_layer *w = &m->layer[layer];
    uint32_t kvw = m->kv_heads * m->key_dim;
    dsm_mtl_encode_norm_at(s, e, s->bx, 0, w->attn_norm, s->bnorm, 0, m->dim, rows);
    dsm_mtl_encode_matmul(s, e, w->q, s->bnorm, s->bq, rows);
    dsm_mtl_encode_matmul(s, e, w->k, s->bnorm, s->bk, rows);
    dsm_mtl_encode_matmul(s, e, w->v, s->bnorm, s->bv, rows);
    dsm_mtl_encode_norm_at(s, e, s->bq, 0, w->q_norm, s->bq, 0, m->key_dim, rows * m->heads);
    dsm_mtl_encode_norm_at(s, e, s->bk, 0, w->k_norm, s->bk, 0, m->key_dim, rows * m->kv_heads);
    rope_batch_params rope = {m->heads, m->key_dim, pos0, rows, m->rope_base};
    [e setComputePipelineState:s->rope_batch];
    [e setBuffer:s->bq offset:0 atIndex:0];
    [e setBytes:&rope length:sizeof(rope) atIndex:1];
    dsm_mtl_dispatch_1d(e, s->rope_batch, rows * m->heads * (m->key_dim / 2));
    rope.heads = m->kv_heads;
    [e setComputePipelineState:s->rope_batch];
    [e setBuffer:s->bk offset:0 atIndex:0];
    [e setBytes:&rope length:sizeof(rope) atIndex:1];
    dsm_mtl_dispatch_1d(e, s->rope_batch, rows * m->kv_heads * (m->key_dim / 2));
    cache_batch_params cache = {layer, pos0, s->context, kvw, rows};
    [e setComputePipelineState:s->cache_batch];
    [e setBuffer:s->bk offset:0 atIndex:0];
    [e setBuffer:s->keys offset:0 atIndex:1];
    [e setBytes:&cache length:sizeof(cache) atIndex:2];
    dsm_mtl_dispatch_1d(e, s->cache_batch, rows * kvw);
    [e setComputePipelineState:s->cache_batch];
    [e setBuffer:s->bv offset:0 atIndex:0];
    [e setBuffer:s->values offset:0 atIndex:1];
    [e setBytes:&cache length:sizeof(cache) atIndex:2];
    dsm_mtl_dispatch_1d(e, s->cache_batch, rows * kvw);
    attn_prefill_params attention = {m->heads,   m->kv_heads, m->key_dim, pos0,
                                     s->context, layer,       rows};
    uint32_t q_per_kv = m->heads / m->kv_heads;
    id<MTLComputePipelineState> attention_pipeline =
        rows > 8 && dsm_mtl_attention_prefill_enabled() && m->key_dim == 128 && q_per_kv <= 8
            ? (dsm_mtl_attention_flash_enabled() ? s->attention_prefill_fa128
                                                 : s->attention_prefill_d128_tiled)
            : s->attention_prefill;
    [e setComputePipelineState:attention_pipeline];
    [e setBuffer:s->bq offset:0 atIndex:0];
    [e setBuffer:s->keys offset:0 atIndex:1];
    [e setBuffer:s->values offset:0 atIndex:2];
    [e setBuffer:s->batt offset:0 atIndex:3];
    [e setBytes:&attention length:sizeof(attention) atIndex:4];
    NSUInteger attention_tgmem =
        attention_pipeline == s->attention_prefill_d128_tiled
            ? 32768
            : (attention_pipeline == s->attention_prefill_fa128
                   ? 12288
                   : (NSUInteger)(2 * 16 * 128 + 4 * q_per_kv * 128) * sizeof(float));
    [e setThreadgroupMemoryLength:attention_tgmem atIndex:0];
    if (attention_pipeline == s->attention_prefill_fa128) {
        /* 8 query slots = (8 / G) rows x G heads sharing one KV head */
        uint32_t group = 8 % q_per_kv == 0 ? q_per_kv : 1, group_rows = 8 / group;
        [e dispatchThreadgroups:MTLSizeMake((rows + group_rows - 1) / group_rows, m->heads / group,
                                            1)
            threadsPerThreadgroup:MTLSizeMake(32, 4, 1)];
    } else {
        [e dispatchThreadgroups:MTLSizeMake((rows + 3) / 4, m->kv_heads, 1)
            threadsPerThreadgroup:MTLSizeMake(32, q_per_kv, 1)];
    }
    dsm_mtl_encode_matmul(s, e, w->o, s->batt, s->btmp, rows);
    [e setComputePipelineState:s->add];
    [e setBuffer:s->bx offset:0 atIndex:0];
    [e setBuffer:s->btmp offset:0 atIndex:1];
    {
        uint32_t n = rows * m->dim;
        [e setBytes:&n length:sizeof(n) atIndex:2];
        dsm_mtl_dispatch_1d(e, s->add, n);
    }
    dsm_mtl_encode_norm_at(s, e, s->bx, 0, w->ffn_norm, s->bnorm, 0, m->dim, rows);
    if (!m->is_moe) {
        dsm_mtl_encode_matmul(s, e, w->gate, s->bnorm, s->bgate, rows);
        dsm_mtl_encode_matmul(s, e, w->up, s->bnorm, s->bup, rows);
        [e setComputePipelineState:s->swiglu];
        [e setBuffer:s->bgate offset:0 atIndex:0];
        [e setBuffer:s->bup offset:0 atIndex:1];
        {
            uint32_t n = rows * m->hidden;
            [e setBytes:&n length:sizeof(n) atIndex:2];
            dsm_mtl_dispatch_1d(e, s->swiglu, n);
        }
        dsm_mtl_encode_matmul(s, e, w->down, s->bgate, s->btmp, rows);
        [e setComputePipelineState:s->add];
        [e setBuffer:s->bx offset:0 atIndex:0];
        [e setBuffer:s->btmp offset:0 atIndex:1];
        {
            uint32_t n = rows * m->dim;
            [e setBytes:&n length:sizeof(n) atIndex:2];
            dsm_mtl_dispatch_1d(e, s->add, n);
        }
    } else {
        dsm_mtl_encode_matmul(s, e, w->gate_inp, s->bnorm, s->broute, rows);
        dsm_mtl_encode_moe_topk_batch(s, e, rows);
        int use_mm_id = dsm_mtl_mul_mm_id_enabled() && rows > 8 &&
                        dsm_mtl_encode_expert_matmul(s, e, w->gate_exps, s->bnorm, s->bgate, rows,
                                                     1, m->dim, m->expert_hidden, 1) &&
                        dsm_mtl_encode_expert_matmul(s, e, w->up_exps, s->bnorm, s->bup, rows, 1,
                                                     m->dim, m->expert_hidden, 0);
        if (use_mm_id) {
            [e setComputePipelineState:s->expert_swiglu];
            [e setBuffer:s->bgate offset:0 atIndex:0];
            [e setBuffer:s->bup offset:0 atIndex:1];
            [e setBytes:&m->expert_hidden length:sizeof(m->expert_hidden) atIndex:2];
            [e setBytes:&m->experts_used length:sizeof(m->experts_used) atIndex:3];
            [e setBytes:&rows length:sizeof(rows) atIndex:4];
            dsm_mtl_dispatch_1d(e, s->expert_swiglu, rows * m->expert_hidden * m->experts_used);
            use_mm_id = dsm_mtl_encode_expert_matmul(s, e, w->down_exps, s->bgate, s->btmp, rows,
                                                     m->experts_used, m->expert_hidden, m->dim, 0);
        }
        if (!use_mm_id) {
            for (uint32_t row = 0; row < rows; ++row) {
                NSUInteger x_off = (NSUInteger)row * m->dim * sizeof(float);
                NSUInteger gate_off =
                    (NSUInteger)row * m->experts_used * m->expert_hidden * sizeof(float);
                NSUInteger tmp_off = (NSUInteger)row * m->experts_used * m->dim * sizeof(float);
                dsm_mtl_encode_expert_gemv_batch_row(s, e, w->gate_exps, s->bnorm, x_off, s->bgate,
                                                     gate_off, row, 0, m->expert_hidden);
                dsm_mtl_encode_expert_gemv_batch_row(s, e, w->up_exps, s->bnorm, x_off, s->bup,
                                                     gate_off, row, 0, m->expert_hidden);
                uint32_t single_row = 1;
                [e setComputePipelineState:s->expert_swiglu];
                [e setBuffer:s->bgate offset:gate_off atIndex:0];
                [e setBuffer:s->bup offset:gate_off atIndex:1];
                [e setBytes:&m->expert_hidden length:sizeof(m->expert_hidden) atIndex:2];
                [e setBytes:&m->experts_used length:sizeof(m->experts_used) atIndex:3];
                [e setBytes:&single_row length:sizeof(single_row) atIndex:4];
                dsm_mtl_dispatch_1d(e, s->expert_swiglu, m->expert_hidden * m->experts_used);
                dsm_mtl_encode_expert_gemv_batch_row(s, e, w->down_exps, s->bgate, gate_off,
                                                     s->btmp, tmp_off, row, m->expert_hidden,
                                                     m->dim);
            }
        }
        [e setComputePipelineState:s->moe_add_batch];
        [e setBuffer:s->bx offset:0 atIndex:0];
        [e setBuffer:s->btmp offset:0 atIndex:1];
        [e setBuffer:s->bw offset:0 atIndex:2];
        [e setBytes:&m->experts_used length:sizeof(m->experts_used) atIndex:3];
        [e setBytes:&m->dim length:sizeof(m->dim) atIndex:4];
        [e setBytes:&rows length:sizeof(rows) atIndex:5];
        dsm_mtl_dispatch_1d(e, s->moe_add_batch, rows * m->dim);
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
        if (s->mm_f32w_enabled)
            dsm_mtl_encode_matmul(s, e, w->gate_inp, s->bnorm, s->broute, rows);
        else
            dsm_mtl_encode_f32_rows(s, e, w->gate_inp, s->bnorm, s->broute, rows, m->dim,
                                    m->experts);
        dsm_mtl_encode_moe_topk_batch(s, e, rows);
    }
    if (ST(5)) {
        int use_mm_id = dsm_mtl_mul_mm_id_enabled() && rows > 8;
        /* The first expert matmul builds the token-to-expert map; the others reuse it. */
        if (use_mm_id)
            use_mm_id = dsm_mtl_encode_expert_matmul(s, e, w->gate_exps, s->bnorm, s->bgate, rows,
                                                     1, m->dim, m->expert_hidden, 1);
        if (use_mm_id)
            use_mm_id = dsm_mtl_encode_expert_matmul(s, e, w->up_exps, s->bnorm, s->bup, rows, 1,
                                                     m->dim, m->expert_hidden, 0);
        if (use_mm_id) {
            [e setComputePipelineState:s->expert_swiglu];
            [e setBuffer:s->bgate offset:0 atIndex:0];
            [e setBuffer:s->bup offset:0 atIndex:1];
            [e setBytes:&m->expert_hidden length:sizeof(m->expert_hidden) atIndex:2];
            [e setBytes:&m->experts_used length:sizeof(m->experts_used) atIndex:3];
            [e setBytes:&rows length:sizeof(rows) atIndex:4];
            dsm_mtl_dispatch_1d(e, s->expert_swiglu, rows * m->expert_hidden * m->experts_used);
        }
        if (use_mm_id)
            use_mm_id = dsm_mtl_encode_expert_matmul(s, e, w->down_exps, s->bgate, s->btmp, rows,
                                                     m->experts_used, m->expert_hidden, m->dim, 0);
        if (!use_mm_id)
            for (uint32_t row = 0; row < rows; ++row) {
                NSUInteger x_off = (NSUInteger)row * m->dim * sizeof(float);
                NSUInteger gate_off =
                    (NSUInteger)row * m->experts_used * m->expert_hidden * sizeof(float);
                NSUInteger tmp_off = (NSUInteger)row * m->experts_used * m->dim * sizeof(float);
                dsm_mtl_encode_expert_gemv_batch_row(s, e, w->gate_exps, s->bnorm, x_off, s->bgate,
                                                     gate_off, row, 0, m->expert_hidden);
                dsm_mtl_encode_expert_gemv_batch_row(s, e, w->up_exps, s->bnorm, x_off, s->bup,
                                                     gate_off, row, 0, m->expert_hidden);
                uint32_t one = 1;
                [e setComputePipelineState:s->expert_swiglu];
                [e setBuffer:s->bgate offset:gate_off atIndex:0];
                [e setBuffer:s->bup offset:gate_off atIndex:1];
                [e setBytes:&m->expert_hidden length:sizeof(m->expert_hidden) atIndex:2];
                [e setBytes:&m->experts_used length:sizeof(m->experts_used) atIndex:3];
                [e setBytes:&one length:sizeof(one) atIndex:4];
                dsm_mtl_dispatch_1d(e, s->expert_swiglu, m->expert_hidden * m->experts_used);
                dsm_mtl_encode_expert_gemv_batch_row(s, e, w->down_exps, s->bgate, gate_off,
                                                     s->btmp, tmp_off, row, m->expert_hidden,
                                                     m->dim);
            }
        [e setComputePipelineState:s->moe_add_batch];
        [e setBuffer:s->bx offset:0 atIndex:0];
        [e setBuffer:s->btmp offset:0 atIndex:1];
        [e setBuffer:s->bw offset:0 atIndex:2];
        [e setBytes:&m->experts_used length:sizeof(m->experts_used) atIndex:3];
        [e setBytes:&m->dim length:sizeof(m->dim) atIndex:4];
        [e setBytes:&rows length:sizeof(rows) atIndex:5];
        dsm_mtl_dispatch_1d(e, s->moe_add_batch, rows * m->dim);
    }
    if (!ST(6))
        return;
    dsm_mtl_encode_f32_rows(s, e, w->gate_inp_shexp, s->bnorm, s->hb_shg, rows, m->dim, 1);
    dsm_mtl_encode_matmul(s, e, w->gate_shexp, s->bnorm, s->hb_sgate, rows);
    dsm_mtl_encode_matmul(s, e, w->up_shexp, s->bnorm, s->hb_sup, rows);
    {
        uint32_t n = rows * m->shared_hidden;
        [e setComputePipelineState:s->swiglu];
        [e setBuffer:s->hb_sgate offset:0 atIndex:0];
        [e setBuffer:s->hb_sup offset:0 atIndex:1];
        [e setBytes:&n length:sizeof(n) atIndex:2];
        dsm_mtl_dispatch_1d(e, s->swiglu, n);
    }
    dsm_mtl_encode_matmul(s, e, w->down_shexp, s->hb_sgate, s->hb_sdown, rows);
    {
        uint32_t n = rows * m->dim;
        [e setComputePipelineState:s->shexp_add_batch];
        [e setBuffer:s->bx offset:0 atIndex:0];
        [e setBuffer:s->hb_sdown offset:0 atIndex:1];
        [e setBuffer:s->hb_shg offset:0 atIndex:2];
        [e setBytes:&m->dim length:sizeof(m->dim) atIndex:3];
        [e setBytes:&n length:sizeof(n) atIndex:4];
        dsm_mtl_dispatch_1d(e, s->shexp_add_batch, n);
    }
}
static void encode_layer_hybrid_batch(dsm_metal *s, id<MTLComputeCommandEncoder> e, uint32_t layer,
                                      uint32_t pos0, uint32_t rows) {
    const dsm_model *m = s->m;
    const dsm_layer *w = &m->layer[layer];
    uint32_t qk = m->ssm_groups * m->ssm_state;
    uint32_t mixed = m->ssm_inner + 2 * qk;
    if (ST(1))
        dsm_mtl_encode_norm_at(s, e, s->bx, 0, w->attn_norm, s->bnorm, 0, m->dim, rows);
    if (m->is_recurrent[layer]) {
        if (ST(1)) {
            dsm_mtl_encode_matmul(s, e, w->qkv, s->bnorm, s->hb_qkv, rows);
            dsm_mtl_encode_matmul(s, e, w->z, s->bnorm, s->hb_z, rows);
            /* qwen35moe ships F32 alpha/beta, dense qwen35 ships Q8_0. */
            if (w->alpha->type == GGML_TYPE_F32 && !s->mm_f32w_enabled)
                dsm_mtl_encode_f32_rows(s, e, w->alpha, s->bnorm, s->hb_g, rows, m->dim,
                                        m->ssm_vheads);
            else
                dsm_mtl_encode_matmul(s, e, w->alpha, s->bnorm, s->hb_g, rows);
            if (w->beta->type == GGML_TYPE_F32 && !s->mm_f32w_enabled)
                dsm_mtl_encode_f32_rows(s, e, w->beta, s->bnorm, s->hb_beta, rows, m->dim,
                                        m->ssm_vheads);
            else
                dsm_mtl_encode_matmul(s, e, w->beta, s->bnorm, s->hb_beta, rows);
        }
        id<MTLBuffer> wb;
        uint64_t off;
        if (ST(2)) {
            ssm_conv_batch_params conv = {mixed, m->ssm_conv - 1, rows};
            if (!dsm_mtl_tensor_view(s, w->conv1d, &wb, &off))
                return;
            [e setComputePipelineState:s->ssm_conv_batch];
            dsm_mtl_set_checked(e, s->hb_qkv, 0, (NSUInteger)rows * mixed * sizeof(float), 0);
            dsm_mtl_set_checked(e, s->conv_state,
                                (NSUInteger)layer * mixed * (m->ssm_conv - 1) * sizeof(float),
                                (NSUInteger)mixed * (m->ssm_conv - 1) * sizeof(float), 1);
            dsm_mtl_set_checked(e, wb, (NSUInteger)off, (NSUInteger)w->conv1d->bytes, 2);
            dsm_mtl_set_checked(e, s->hb_conv, 0, (NSUInteger)rows * mixed * sizeof(float), 3);
            [e setBytes:&conv length:sizeof(conv) atIndex:4];
            dsm_mtl_dispatch_1d(e, s->ssm_conv_batch, mixed);
            uint32_t n = rows * qk;
            [e setComputePipelineState:s->split_gdn_qk_batch];
            dsm_mtl_set_checked(e, s->hb_conv, 0, (NSUInteger)rows * mixed * sizeof(float), 0);
            dsm_mtl_set_checked(e, s->hb_dq, 0, (NSUInteger)n * sizeof(float), 1);
            dsm_mtl_set_checked(e, s->hb_dk, 0, (NSUInteger)n * sizeof(float), 2);
            [e setBytes:&qk length:sizeof(qk) atIndex:3];
            [e setBytes:&mixed length:sizeof(mixed) atIndex:4];
            [e setBytes:&n length:sizeof(n) atIndex:5];
            dsm_mtl_dispatch_1d(e, s->split_gdn_qk_batch, n);
            norm_params l2 = {m->ssm_state, rows * m->ssm_groups, m->rms_eps};
            [e setComputePipelineState:s->l2norm];
            dsm_mtl_set_checked(e, s->hb_dq, 0, (NSUInteger)n * sizeof(float), 0);
            [e setBytes:&l2 length:sizeof(l2) atIndex:1];
            [e setThreadgroupMemoryLength:256 * sizeof(float) atIndex:0];
            [e dispatchThreadgroups:MTLSizeMake(l2.groups, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            dsm_mtl_set_checked(e, s->hb_dk, 0, (NSUInteger)n * sizeof(float), 0);
            [e dispatchThreadgroups:MTLSizeMake(l2.groups, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            n = rows * m->ssm_vheads;
            if (!dsm_mtl_tensor_view(s, w->dt_bias, &wb, &off))
                return;
            [e setComputePipelineState:s->gdn_gates_batch];
            dsm_mtl_set_checked(e, s->hb_g, 0, (NSUInteger)n * sizeof(float), 0);
            dsm_mtl_set_checked(e, s->hb_beta, 0, (NSUInteger)n * sizeof(float), 1);
            dsm_mtl_set_checked(e, wb, (NSUInteger)off, (NSUInteger)w->dt_bias->bytes, 2);
            if (!dsm_mtl_tensor_view(s, w->ssm_a, &wb, &off))
                return;
            dsm_mtl_set_checked(e, wb, (NSUInteger)off, (NSUInteger)w->ssm_a->bytes, 3);
            [e setBytes:&m->ssm_vheads length:sizeof(m->ssm_vheads) atIndex:4];
            [e setBytes:&n length:sizeof(n) atIndex:5];
            dsm_mtl_dispatch_1d(e, s->gdn_gates_batch, n);
        }
        if (ST(7)) {
            gdn_batch_params gdn = {m->ssm_groups, m->ssm_vheads, m->ssm_state, layer, rows, qk,
                                    mixed,         m->ssm_inner};
            [e setComputePipelineState:s->gdn_batch];
            dsm_mtl_set_checked(e, s->hb_dq, 0, (NSUInteger)rows * qk * sizeof(float), 0);
            dsm_mtl_set_checked(e, s->hb_dk, 0, (NSUInteger)rows * qk * sizeof(float), 1);
            dsm_mtl_set_checked(
                e, s->hb_conv, (NSUInteger)2 * qk * sizeof(float),
                (NSUInteger)(rows - 1) * mixed * sizeof(float) + m->ssm_inner * sizeof(float), 2);
            dsm_mtl_set_checked(e, s->hb_g, 0, (NSUInteger)rows * m->ssm_vheads * sizeof(float), 3);
            dsm_mtl_set_checked(e, s->hb_beta, 0, (NSUInteger)rows * m->ssm_vheads * sizeof(float),
                                4);
            dsm_mtl_set_checked(e, s->ssm_state, 0, s->ssm_state.length, 5);
            dsm_mtl_set_checked(e, s->hb_mix, 0, (NSUInteger)rows * m->ssm_inner * sizeof(float),
                                6);
            [e setBytes:&gdn length:sizeof(gdn) atIndex:7];
            [e dispatchThreadgroups:MTLSizeMake((m->ssm_state + 3) / 4, m->ssm_vheads, 1)
                threadsPerThreadgroup:MTLSizeMake(32, 4, 1)];
        }
        if (ST(2)) {
            norm_params gated = {m->ssm_state, rows * m->ssm_vheads, m->rms_eps};
            if (!dsm_mtl_tensor_view(s, w->ssm_norm, &wb, &off))
                return;
            [e setComputePipelineState:s->gated_rmsnorm];
            dsm_mtl_set_checked(e, s->hb_mix, 0, (NSUInteger)rows * m->ssm_inner * sizeof(float),
                                0);
            dsm_mtl_set_checked(e, s->hb_z, 0, (NSUInteger)rows * m->ssm_inner * sizeof(float), 1);
            dsm_mtl_set_checked(e, wb, (NSUInteger)off, (NSUInteger)w->ssm_norm->bytes, 2);
            [e setBytes:&gated length:sizeof(gated) atIndex:3];
            [e setThreadgroupMemoryLength:256 * sizeof(float) atIndex:0];
            [e dispatchThreadgroups:MTLSizeMake(gated.groups, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        }
        if (ST(3))
            dsm_mtl_encode_matmul(s, e, w->ssm_out, s->hb_mix, s->hb_out, rows);
    } else {
        if (ST(1)) {
            dsm_mtl_encode_matmul(s, e, w->q, s->bnorm, s->hb_qg, rows);
            dsm_mtl_encode_matmul(s, e, w->k, s->bnorm, s->hb_ak, rows);
            dsm_mtl_encode_matmul(s, e, w->v, s->bnorm, s->hb_av, rows);
        }
        if (ST(2)) {
            {
                uint32_t n = rows * m->q_width;
                [e setComputePipelineState:s->split_q_gate];
                dsm_mtl_set_checked(e, s->hb_qg, 0,
                                    (NSUInteger)rows * 2 * m->q_width * sizeof(float), 0);
                dsm_mtl_set_checked(e, s->hb_aq, 0, (NSUInteger)n * sizeof(float), 1);
                dsm_mtl_set_checked(e, s->hb_agate, 0, (NSUInteger)n * sizeof(float), 2);
                [e setBytes:&m->key_dim length:sizeof(m->key_dim) atIndex:3];
                dsm_mtl_dispatch_1d(e, s->split_q_gate, n);
            }
            dsm_mtl_encode_norm_at(s, e, s->hb_aq, 0, w->q_norm, s->hb_aq, 0, m->key_dim,
                                   rows * m->heads);
            dsm_mtl_encode_norm_at(s, e, s->hb_ak, 0, w->k_norm, s->hb_ak, 0, m->key_dim,
                                   rows * m->kv_heads);
            rope_partial_batch_params rope = {m->heads, m->key_dim, m->rope_dims,
                                              pos0,     rows,       m->rope_base};
            [e setComputePipelineState:s->rope_partial_batch];
            dsm_mtl_set_checked(e, s->hb_aq, 0, (NSUInteger)rows * m->q_width * sizeof(float), 0);
            [e setBytes:&rope length:sizeof(rope) atIndex:1];
            dsm_mtl_dispatch_1d(e, s->rope_partial_batch, rows * m->heads * (m->rope_dims / 2));
            rope.heads = m->kv_heads;
            [e setComputePipelineState:s->rope_partial_batch];
            dsm_mtl_set_checked(e, s->hb_ak, 0,
                                (NSUInteger)rows * m->kv_heads * m->key_dim * sizeof(float), 0);
            [e setBytes:&rope length:sizeof(rope) atIndex:1];
            dsm_mtl_dispatch_1d(e, s->rope_partial_batch, rows * m->kv_heads * (m->rope_dims / 2));
            cache_batch_params cache = {m->attn_index[layer], pos0, s->context,
                                        m->kv_heads * m->key_dim, rows};
            [e setComputePipelineState:s->cache_batch];
            dsm_mtl_set_checked(e, s->hb_ak, 0, (NSUInteger)rows * cache.width * sizeof(float), 0);
            dsm_mtl_set_checked(e, s->keys, 0, s->keys.length, 1);
            [e setBytes:&cache length:sizeof(cache) atIndex:2];
            dsm_mtl_dispatch_1d(e, s->cache_batch, rows * cache.width);
            [e setComputePipelineState:s->cache_batch];
            dsm_mtl_set_checked(e, s->hb_av, 0, (NSUInteger)rows * cache.width * sizeof(float), 0);
            dsm_mtl_set_checked(e, s->values, 0, s->values.length, 1);
            [e setBytes:&cache length:sizeof(cache) atIndex:2];
            dsm_mtl_dispatch_1d(e, s->cache_batch, rows * cache.width);
        }
        if (ST(8) && rows > 8 && dsm_mtl_attention_prefill_enabled() && m->key_dim == 256 &&
            m->heads / m->kv_heads <= 8) {
            attn_prefill_params attention = {m->heads,   m->kv_heads,          m->key_dim, pos0,
                                             s->context, m->attn_index[layer], rows};
            [e setComputePipelineState:s->attention_prefill_d256];
            dsm_mtl_set_checked(e, s->hb_aq, 0, (NSUInteger)rows * m->q_width * sizeof(float), 0);
            dsm_mtl_set_checked(e, s->keys, 0, s->keys.length, 1);
            dsm_mtl_set_checked(e, s->values, 0, s->values.length, 2);
            dsm_mtl_set_checked(e, s->hb_mix, 0, (NSUInteger)rows * m->attn_width * sizeof(float),
                                3);
            [e setBytes:&attention length:sizeof(attention) atIndex:4];
            [e setThreadgroupMemoryLength:32768 atIndex:0];
            [e dispatchThreadgroups:MTLSizeMake((rows + 3) / 4, m->kv_heads, 1)
                threadsPerThreadgroup:MTLSizeMake(32, m->heads / m->kv_heads, 1)];
        } else if (ST(8))
            for (uint32_t row = 0; row < rows; ++row) {
                dsm_mtl_encode_attention_decode_at(s, e, m->attn_index[layer], pos0 + row, s->hb_aq,
                                                   (NSUInteger)row * m->q_width * sizeof(float),
                                                   s->hb_mix,
                                                   (NSUInteger)row * m->attn_width * sizeof(float));
            }
        if (ST(2)) {
            uint32_t n = rows * m->attn_width;
            [e setComputePipelineState:s->sigmoid_mul];
            dsm_mtl_set_checked(e, s->hb_mix, 0, (NSUInteger)n * sizeof(float), 0);
            dsm_mtl_set_checked(e, s->hb_agate, 0, (NSUInteger)rows * m->q_width * sizeof(float),
                                1);
            [e setBytes:&n length:sizeof(n) atIndex:2];
            dsm_mtl_dispatch_1d(e, s->sigmoid_mul, n);
        }
        if (ST(3))
            dsm_mtl_encode_matmul(s, e, w->o, s->hb_mix, s->hb_out, rows);
    }
    if (ST(3)) {
        uint32_t n = rows * m->dim;
        [e setComputePipelineState:s->add];
        [e setBuffer:s->bx offset:0 atIndex:0];
        [e setBuffer:s->hb_out offset:0 atIndex:1];
        [e setBytes:&n length:sizeof(n) atIndex:2];
        dsm_mtl_dispatch_1d(e, s->add, n);
        dsm_mtl_encode_norm_at(s, e, s->bx, 0, w->ffn_norm, s->bnorm, 0, m->dim, rows);
    }
    if (m->is_moe) {
        encode_hybrid_moe_batch(s, e, w, rows);
    } else if (ST(6)) {
        dsm_mtl_encode_matmul(s, e, w->gate, s->bnorm, s->hb_sgate, rows);
        dsm_mtl_encode_matmul(s, e, w->up, s->bnorm, s->hb_sup, rows);
        {
            uint32_t n = rows * m->hidden;
            [e setComputePipelineState:s->swiglu];
            dsm_mtl_set_checked(e, s->hb_sgate, 0, (NSUInteger)n * sizeof(float), 0);
            dsm_mtl_set_checked(e, s->hb_sup, 0, (NSUInteger)n * sizeof(float), 1);
            [e setBytes:&n length:sizeof(n) atIndex:2];
            dsm_mtl_dispatch_1d(e, s->swiglu, n);
        }
        dsm_mtl_encode_matmul(s, e, w->down, s->hb_sgate, s->hb_sdown, rows);
        {
            uint32_t n = rows * m->dim;
            [e setComputePipelineState:s->add];
            dsm_mtl_set_checked(e, s->bx, 0, (NSUInteger)n * sizeof(float), 0);
            dsm_mtl_set_checked(e, s->hb_sdown, 0, (NSUInteger)n * sizeof(float), 1);
            [e setBytes:&n length:sizeof(n) atIndex:2];
            dsm_mtl_dispatch_1d(e, s->add, n);
        }
    }
}
static int prefill_chunk(dsm_metal *s, const int32_t *tokens, uint32_t rows, float *logits) {
    const dsm_model *m = s->m;
    const ggml_type_info *info = gguf_type_info(m->token_embd->type);
    float *x = s->bx.contents;
    uint32_t pos0 = s->pos;
    for (uint32_t row = 0; row < rows; ++row) {
        int32_t token = tokens[row];
        if (token < 0 || (uint32_t)token >= m->vocab || pos0 + row >= s->context)
            return 0;
        const uint8_t *embedding =
            (const uint8_t *)m->token_embd->data +
            (uint64_t)token * (m->dim / info->block_elems) * info->block_bytes;
        dsm_dequant_row(m->token_embd->type, embedding, x + (uint64_t)row * m->dim, m->dim);
    }
    id<MTLCommandBuffer> cb = [s->queue commandBuffer];
    id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
    static int prof = -1;
    if (prof < 0)
        prof = getenv("DSM_STAGE_PROF") != NULL;
    if (prof && dsm_mtl_qwen35_arch(m)) {
        [e endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        for (uint32_t layer = 0; layer < m->layers; ++layer)
            for (int st = 1; st <= 8; ++st) {
                cb = [s->queue commandBuffer];
                e = [cb computeCommandEncoder];
                dbg_stage = st;
                encode_layer_hybrid_batch(s, e, layer, pos0, rows);
                dbg_stage = 0;
                [e endEncoding];
                [cb commit];
                [cb waitUntilCompleted];
                dsm_stage_gpu[st][m->is_recurrent[layer] ? 0 : 1] +=
                    cb.GPUEndTime - cb.GPUStartTime;
            }
        cb = [s->queue commandBuffer];
        e = [cb computeCommandEncoder];
    } else
        for (uint32_t layer = 0; layer < m->layers; ++layer) {
            if (dsm_mtl_qwen35_arch(m))
                encode_layer_hybrid_batch(s, e, layer, pos0, rows);
            else
                encode_layer_batch(s, e, layer, pos0, rows);
        }
    if (logits) {
        NSUInteger off = (NSUInteger)(rows - 1) * m->dim * sizeof(float);
        dsm_mtl_encode_norm_at(s, e, s->bx, off, m->output_norm, s->norm_x, 0, m->dim, 1);
        dsm_mtl_encode_gemv(s, e, m->output, s->norm_x, s->logits);
    }
    [e endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    if (cb.status != MTLCommandBufferStatusCompleted)
        return 0;
    s->pos += rows;
    if (logits)
        memcpy(logits, s->logits.contents, m->vocab * sizeof(*logits));
    return 1;
}
int dsm_metal_prefill(dsm_metal *s, const int32_t *tokens, uint32_t n, float *logits) {
    if (!s || !tokens || !n || !logits || n > s->context - s->pos)
        return 0;
    if (s->kv_f16 || (dsm_mtl_qwen35_arch(s->m) &&
         (!dsm_mtl_hybrid_prefill_enabled() || getenv("DSM_DUMP_DIR"))) ||
        s->profile.enabled || !dsm_mtl_prefill_enabled()) {
        for (uint32_t i = 0; i < n; ++i)
            if (!dsm_metal_token(s, tokens[i], logits))
                return 0;
        return 1;
    }
    for (uint32_t start = 0; start < n;) {
        uint32_t rows = n - start;
        if (rows > s->batch_chunk)
            rows = s->batch_chunk;
        if (!prefill_chunk(s, tokens + start, rows, start + rows == n ? logits : NULL))
            return 0;
        start += rows;
    }
    return 1;
}
