#include "metal_internal.h"

void dsm_mtl_dump_hybrid_layer(const dsm_metal *s, uint32_t layer) {
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
void dsm_mtl_profile_print(const dsm_metal *s) {
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
int dsm_metal_matvec_test(dsm_metal *s,const gguf_tensor *w,const float *input,float *output){if(!s||!w||!dsm_quant_supported(w->type))return 0;id<MTLBuffer> x=dsm_mtl_buffer(s->device,w->dims[0]*sizeof(float)),y=dsm_mtl_buffer(s->device,w->dims[1]*sizeof(float));if(!x||!y)return 0;memcpy(x.contents,input,w->dims[0]*sizeof(float));id<MTLCommandBuffer> cb=[s->queue commandBuffer];id<MTLComputeCommandEncoder> e=[cb computeCommandEncoder];dsm_mtl_encode_gemv(s,e,w,x,y);[e endEncoding];[cb commit];[cb waitUntilCompleted];if(cb.status!=MTLCommandBufferStatusCompleted)return 0;memcpy(output,y.contents,w->dims[1]*sizeof(float));return 1;}
int dsm_metal_matmul_test(dsm_metal *s, const gguf_tensor *w, const float *input,
                          uint32_t rows, float *output) {
    if (!s || !w || !input || !output || !rows || w->ndim != 2 ||
        !dsm_mtl_matmul_pipeline(s, w->type) || w->dims[0] % 32) return 0;
    id<MTLBuffer> x = dsm_mtl_buffer(s->device, (uint64_t)rows * w->dims[0] * sizeof(float));
    id<MTLBuffer> y = dsm_mtl_buffer(s->device, (uint64_t)rows * w->dims[1] * sizeof(float));
    if (!x || !y) return 0;
    memcpy(x.contents, input, (uint64_t)rows * w->dims[0] * sizeof(float));
    id<MTLCommandBuffer> cb = [s->queue commandBuffer];
    id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
    dsm_mtl_encode_matmul(s, e, w, x, y, rows);
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
        w->dims[2] != s->m->experts || !dsm_mtl_matmul_id_pipeline(s, w->type)) return 0;
    id<MTLBuffer> x = dsm_mtl_buffer(s->device, (uint64_t)rows * input_rows * w->dims[0] * sizeof(float));
    id<MTLBuffer> y = dsm_mtl_buffer(s->device, (uint64_t)rows * s->m->experts_used * w->dims[1] * sizeof(float));
    if (!x || !y) return 0;
    memcpy(x.contents, input, (uint64_t)rows * input_rows * w->dims[0] * sizeof(float));
    memcpy(s->bids.contents, ids, (uint64_t)rows * s->m->experts_used * sizeof(*ids));
    id<MTLCommandBuffer> cb = [s->queue commandBuffer];
    id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
    if (!dsm_mtl_encode_expert_matmul(s, e, w, x, y, rows, input_rows, (uint32_t)w->dims[0],
                              (uint32_t)w->dims[1], 1)) return 0;
    [e endEncoding]; [cb commit]; [cb waitUntilCompleted];
    if (cb.status != MTLCommandBufferStatusCompleted) return 0;
    memcpy(output, y.contents,
           (uint64_t)rows * s->m->experts_used * w->dims[1] * sizeof(*output));
    return 1;
}
int dsm_metal_expert_matvec_test(dsm_metal *s,const gguf_tensor *w,uint32_t expert,const float *input,float *output){
    if(!s||!w||!dsm_mtl_expert_gemv_supported(w->type)||w->ndim!=3||expert>=w->dims[2])return 0;
    id<MTLBuffer> wb,x,y,ids;uint64_t off;if(!dsm_mtl_tensor_view(s,w,&wb,&off))return 0;
    x=dsm_mtl_buffer(s->device,w->dims[0]*sizeof(float));y=dsm_mtl_buffer(s->device,w->dims[1]*sizeof(float));ids=dsm_mtl_buffer(s->device,sizeof(expert));if(!x||!y||!ids)return 0;
    memcpy(x.contents,input,w->dims[0]*sizeof(float));memcpy(ids.contents,&expert,sizeof(expert));
    expert_gemv_params p={(uint32_t)w->dims[0],(uint32_t)w->dims[1],(uint32_t)(w->bytes/w->dims[2]),(uint32_t)w->dims[1],0};
    id<MTLCommandBuffer> cb=[s->queue commandBuffer];id<MTLComputeCommandEncoder> e=[cb computeCommandEncoder];
    [e setComputePipelineState:dsm_mtl_expert_gemv_pipeline(s,w->type)];[e setBuffer:wb offset:(NSUInteger)off atIndex:0];[e setBuffer:x offset:0 atIndex:1];[e setBuffer:y offset:0 atIndex:2];[e setBytes:&p length:sizeof(p) atIndex:3];[e setBuffer:ids offset:0 atIndex:4];if(w->type==GGML_TYPE_Q8_0){[e setThreadgroupMemoryLength:2*32*sizeof(float) atIndex:0];[e dispatchThreadgroups:MTLSizeMake((p.out+1)/2,1,1) threadsPerThreadgroup:MTLSizeMake(32,4,1)];}else if(w->type==GGML_TYPE_Q3_K||w->type==GGML_TYPE_Q4_K){[e dispatchThreadgroups:MTLSizeMake((p.out+3)/4,1,1) threadsPerThreadgroup:MTLSizeMake(32,2,1)];}else{[e setThreadgroupMemoryLength:256*sizeof(float) atIndex:0];[e dispatchThreadgroups:MTLSizeMake(p.out,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];}
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
    if (!s || !max_abs || !max_rel || !rows || !dsm_mtl_qwen35_arch(s->m) ||
        layer >= s->m->layers || !s->m->is_recurrent[layer]) return 1;
    m = s->m;
    w = &m->layer[layer];
    if (!w->conv1d || !w->dt_bias || !w->ssm_a || !m->ssm_conv ||
        m->ssm_conv < 2 || !m->ssm_state || m->ssm_state > 128 ||
        m->ssm_state % 32 || !m->ssm_groups || !m->ssm_vheads) return 1;
    if (!dsm_mtl_tensor_view(s, w->conv1d, &conv_weight, &conv_weight_off) ||
        !dsm_mtl_tensor_view(s, w->dt_bias, &dt_bias, &dt_bias_off) ||
        !dsm_mtl_tensor_view(s, w->ssm_a, &ssm_a, &ssm_a_off)) return 1;

    qk_width = m->ssm_groups * m->ssm_state;
    mixed_width = m->ssm_inner + 2 * qk_width;
    history = m->ssm_conv - 1;
    v_stride = m->ssm_vheads * m->ssm_state;
    qk_count = rows * qk_width;
    gate_count = rows * m->ssm_vheads;
    conv_state_bytes = (uint64_t)mixed_width * history * sizeof(float);
    ssm_state_bytes = (uint64_t)m->ssm_vheads * m->ssm_state * m->ssm_state * sizeof(float);
    input = dsm_mtl_buffer(s->device, (uint64_t)rows * mixed_width * sizeof(float));
    conv_decode = dsm_mtl_buffer(s->device, (uint64_t)rows * mixed_width * sizeof(float));
    conv_batch = dsm_mtl_buffer(s->device, (uint64_t)rows * mixed_width * sizeof(float));
    q_decode = dsm_mtl_buffer(s->device, qk_width * sizeof(float));
    k_decode = dsm_mtl_buffer(s->device, qk_width * sizeof(float));
    q_batch = dsm_mtl_buffer(s->device, (uint64_t)qk_count * sizeof(float));
    k_batch = dsm_mtl_buffer(s->device, (uint64_t)qk_count * sizeof(float));
    alpha_decode = dsm_mtl_buffer(s->device, (uint64_t)gate_count * sizeof(float));
    beta_decode = dsm_mtl_buffer(s->device, (uint64_t)gate_count * sizeof(float));
    alpha_batch = dsm_mtl_buffer(s->device, (uint64_t)gate_count * sizeof(float));
    beta_batch = dsm_mtl_buffer(s->device, (uint64_t)gate_count * sizeof(float));
    out_decode = dsm_mtl_buffer(s->device, (uint64_t)rows * v_stride * sizeof(float));
    out_batch = dsm_mtl_buffer(s->device, (uint64_t)rows * v_stride * sizeof(float));
    history_decode = dsm_mtl_buffer(s->device, conv_state_bytes);
    history_batch = dsm_mtl_buffer(s->device, conv_state_bytes);
    state_decode = dsm_mtl_buffer(s->device, ssm_state_bytes);
    state_batch = dsm_mtl_buffer(s->device, ssm_state_bytes);
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
        [e setBytes:&conv length:sizeof(conv) atIndex:4]; dsm_mtl_dispatch_1d(e, s->ssm_conv, mixed_width);
        [e setComputePipelineState:s->split_gdn_qk]; [e setBuffer:conv_decode offset:mixed_off atIndex:0];
        [e setBuffer:q_decode offset:0 atIndex:1]; [e setBuffer:k_decode offset:0 atIndex:2];
        [e setBytes:&qk_width length:sizeof(qk_width) atIndex:3]; dsm_mtl_dispatch_1d(e, s->split_gdn_qk, qk_width);
        [e setComputePipelineState:s->l2norm]; [e setBuffer:q_decode offset:0 atIndex:0]; [e setBytes:&l2 length:sizeof(l2) atIndex:1];
        [e setThreadgroupMemoryLength:256 * sizeof(float) atIndex:0]; [e dispatchThreadgroups:MTLSizeMake(m->ssm_groups, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [e setBuffer:k_decode offset:0 atIndex:0]; [e dispatchThreadgroups:MTLSizeMake(m->ssm_groups, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [e setComputePipelineState:s->gdn_gates]; [e setBuffer:alpha_decode offset:gate_off atIndex:0]; [e setBuffer:beta_decode offset:gate_off atIndex:1];
        [e setBuffer:dt_bias offset:(NSUInteger)dt_bias_off atIndex:2]; [e setBuffer:ssm_a offset:(NSUInteger)ssm_a_off atIndex:3]; dsm_mtl_dispatch_1d(e, s->gdn_gates, m->ssm_vheads);
        [e setComputePipelineState:s->gdn]; [e setBuffer:q_decode offset:0 atIndex:0]; [e setBuffer:k_decode offset:0 atIndex:1];
        [e setBuffer:conv_decode offset:mixed_off + (NSUInteger)2 * qk_width * sizeof(float) atIndex:2]; [e setBuffer:alpha_decode offset:gate_off atIndex:3]; [e setBuffer:beta_decode offset:gate_off atIndex:4];
        [e setBuffer:state_decode offset:0 atIndex:5]; [e setBuffer:out_decode offset:out_off atIndex:6]; [e setBytes:&gdn length:sizeof(gdn) atIndex:7];
        [e dispatchThreadgroups:MTLSizeMake((m->ssm_state + 3) / 4, m->ssm_vheads, 1) threadsPerThreadgroup:MTLSizeMake(32, 4, 1)];
    }
    ssm_conv_batch_params conv_batch_p = {mixed_width, history, rows};
    gdn_batch_params gdn_batch_p = {m->ssm_groups, m->ssm_vheads, m->ssm_state, 0, rows, qk_width, mixed_width, v_stride};
    [e setComputePipelineState:s->ssm_conv_batch]; [e setBuffer:input offset:0 atIndex:0]; [e setBuffer:history_batch offset:0 atIndex:1];
    [e setBuffer:conv_weight offset:(NSUInteger)conv_weight_off atIndex:2]; [e setBuffer:conv_batch offset:0 atIndex:3]; [e setBytes:&conv_batch_p length:sizeof(conv_batch_p) atIndex:4]; dsm_mtl_dispatch_1d(e, s->ssm_conv_batch, mixed_width);
    [e setComputePipelineState:s->split_gdn_qk_batch]; [e setBuffer:conv_batch offset:0 atIndex:0]; [e setBuffer:q_batch offset:0 atIndex:1]; [e setBuffer:k_batch offset:0 atIndex:2];
    [e setBytes:&qk_width length:sizeof(qk_width) atIndex:3]; [e setBytes:&mixed_width length:sizeof(mixed_width) atIndex:4]; [e setBytes:&qk_count length:sizeof(qk_count) atIndex:5]; dsm_mtl_dispatch_1d(e, s->split_gdn_qk_batch, qk_count);
    l2.groups = rows * m->ssm_groups;
    [e setComputePipelineState:s->l2norm]; [e setBuffer:q_batch offset:0 atIndex:0]; [e setBytes:&l2 length:sizeof(l2) atIndex:1]; [e setThreadgroupMemoryLength:256 * sizeof(float) atIndex:0]; [e dispatchThreadgroups:MTLSizeMake(l2.groups, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [e setBuffer:k_batch offset:0 atIndex:0]; [e dispatchThreadgroups:MTLSizeMake(l2.groups, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [e setComputePipelineState:s->gdn_gates_batch]; [e setBuffer:alpha_batch offset:0 atIndex:0]; [e setBuffer:beta_batch offset:0 atIndex:1]; [e setBuffer:dt_bias offset:(NSUInteger)dt_bias_off atIndex:2]; [e setBuffer:ssm_a offset:(NSUInteger)ssm_a_off atIndex:3]; [e setBytes:&m->ssm_vheads length:sizeof(m->ssm_vheads) atIndex:4]; [e setBytes:&gate_count length:sizeof(gate_count) atIndex:5]; dsm_mtl_dispatch_1d(e, s->gdn_gates_batch, gate_count);
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
    PROFILE_ENCODE(PROFILE_RMSNORM,1,0,dsm_mtl_encode_norm(s,e,s->x,w->attn_norm,s->norm_x,m->dim,1));
    PROFILE_ENCODE(PROFILE_QKV_GEMV,3,w->q->bytes+w->k->bytes+w->v->bytes,dsm_mtl_encode_gemv(s,e,w->q,s->norm_x,s->q);dsm_mtl_encode_gemv(s,e,w->k,s->norm_x,s->k);dsm_mtl_encode_gemv(s,e,w->v,s->norm_x,s->v));
    PROFILE_ENCODE(PROFILE_QK_NORM_ROPE,4,0,dsm_mtl_encode_norm(s,e,s->q,w->q_norm,s->q,m->key_dim,m->heads);dsm_mtl_encode_norm(s,e,s->k,w->k_norm,s->k,m->key_dim,m->kv_heads);dsm_mtl_encode_rope(s,e,s->q,m->heads);dsm_mtl_encode_rope(s,e,s->k,m->kv_heads));
    cache_params cp={layer,s->pos,s->context,kvw};
    PROFILE_ENCODE(PROFILE_KV_STORE,2,0,[e setComputePipelineState:s->cache];[e setBuffer:s->k offset:0 atIndex:0];[e setBuffer:s->keys offset:0 atIndex:1];[e setBytes:&cp length:sizeof(cp) atIndex:2];dsm_mtl_dispatch_1d(e,s->cache,kvw);[e setBuffer:s->v offset:0 atIndex:0];[e setBuffer:s->values offset:0 atIndex:1];[e setBytes:&cp length:sizeof(cp) atIndex:2];dsm_mtl_dispatch_1d(e,s->cache,kvw));
    PROFILE_ENCODE(PROFILE_ATTENTION,dsm_mtl_attention_decode_enabled()?2:1,0,dsm_mtl_encode_attention_decode(s,e,layer,s->pos));
    PROFILE_ENCODE(PROFILE_O_GEMV,1,w->o->bytes,dsm_mtl_encode_gemv(s,e,w->o,s->att,s->tmp));
    PROFILE_ENCODE(PROFILE_RESIDUAL,1,0,dsm_mtl_encode_simple(e,s->add,s->x,s->tmp,m->dim));
    PROFILE_ENCODE(PROFILE_RMSNORM,1,0,dsm_mtl_encode_norm(s,e,s->x,w->ffn_norm,s->norm_x,m->dim,1));
    if (m->is_moe) {
        PROFILE_ENCODE(PROFILE_MOE_ROUTER, 1, w->gate_inp->bytes, dsm_mtl_encode_moe_router(s, e, w));
        PROFILE_ENCODE(PROFILE_MOE_TOPK, 1, 0, dsm_mtl_encode_moe_topk(s, e));
        PROFILE_ENCODE(PROFILE_MOE_EXPERT_GATE_UP, 3, m->experts_used * (w->gate_exps->bytes + w->up_exps->bytes) / m->experts, dsm_mtl_encode_moe_gate_up(s, e, w));
        PROFILE_ENCODE(PROFILE_MOE_EXPERT_DOWN, 1, m->experts_used * w->down_exps->bytes / m->experts, dsm_mtl_encode_moe_down(s, e, w));
        PROFILE_ENCODE(PROFILE_MOE_ACCUMULATE, 1, 0, dsm_mtl_encode_moe_accumulate(s, e));
    } else {
        PROFILE_ENCODE(PROFILE_FFN_GATE_UP_GEMV,2,w->gate->bytes+w->up->bytes,dsm_mtl_encode_gemv(s,e,w->gate,s->norm_x,s->gate);dsm_mtl_encode_gemv(s,e,w->up,s->norm_x,s->up));
        PROFILE_ENCODE(PROFILE_SWIGLU,1,0,dsm_mtl_encode_simple(e,s->swiglu,s->gate,s->up,m->hidden));
        PROFILE_ENCODE(PROFILE_FFN_DOWN_GEMV,1,w->down->bytes,dsm_mtl_encode_gemv(s,e,w->down,s->gate,s->tmp));
        PROFILE_ENCODE(PROFILE_RESIDUAL,1,0,dsm_mtl_encode_simple(e,s->add,s->x,s->tmp,m->dim));
    }
#undef PROFILE_ENCODE
    return 1;
}
int dsm_mtl_profile_token(dsm_metal *s, float *logits) {
    const dsm_model *m=s->m; uint32_t kvw=m->kv_heads*m->key_dim;
    for(uint32_t l=0;l<m->layers;++l)if(!profile_layer(s,&m->layer[l],l,kvw))return 0;
    id<MTLCommandBuffer> cb; double start; id<MTLComputeCommandEncoder> e=profile_encoder(s,&cb,&start);
    dsm_mtl_encode_norm(s,e,s->x,m->output_norm,s->norm_x,m->dim,1);
    if(!profile_complete(s,PROFILE_RMSNORM,cb,e,1,0,start))return 0;
    e=profile_encoder(s,&cb,&start);dsm_mtl_encode_gemv(s,e,m->output,s->norm_x,s->logits);
    if(!profile_complete(s,PROFILE_OUTPUT_GEMV,cb,e,1,m->output->bytes,start))return 0;
    memcpy(logits,s->logits.contents,m->vocab*4);++s->pos;++s->profile.tokens;return 1;
}
double dsm_metal_gemv_bench(dsm_metal *s, const gguf_tensor *w, int reps) {
    uint32_t rows = w->ndim > 1 ? (uint32_t)w->dims[1] : 1;
    id<MTLBuffer> x = dsm_mtl_buffer(s->device, w->dims[0] * sizeof(float));
    id<MTLBuffer> y = dsm_mtl_buffer(s->device, (uint64_t)rows * sizeof(float));
    if (!x || !y || reps < 1) return -1.0;
    float *input = x.contents;
    for (uint64_t i = 0; i < w->dims[0]; ++i) input[i] = 0.001f * (float)(i % 97);
    id<MTLCommandBuffer> cb = [s->queue commandBuffer];
    id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
    for (int rep = 0; rep < reps; ++rep) dsm_mtl_encode_gemv(s, e, w, x, y);
    [e endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    return cb.status == MTLCommandBufferStatusCompleted
                   ? cb.GPUEndTime - cb.GPUStartTime : -1.0;
}
