#include "metal_internal.h"

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

id<MTLBuffer> dsm_mtl_buffer(id<MTLDevice> device, uint64_t bytes) {
    return [device newBufferWithLength:(NSUInteger)bytes options:MTLResourceStorageModeShared];
}
void dsm_mtl_set_checked(id<MTLComputeCommandEncoder> e, id<MTLBuffer> b,
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
int dsm_mtl_prefill_enabled(void) {
    const char *value = getenv("DSM_PREFILL");
    return !value || strcmp(value, "0");
}
int dsm_mtl_hybrid_prefill_enabled(void) {
    const char *value = getenv("DSM_HYBRID_PREFILL");
    return !value || strcmp(value, "0");
}
static int f32_weight_matmul_enabled(void) {
    const char *value = getenv("DSM_MM_F32W");
    return !value || strcmp(value, "0");
}
int dsm_mtl_mul_mm_id_enabled(void) {
    const char *value = getenv("DSM_MUL_MM_ID");
    return !value || strcmp(value, "0");
}
int dsm_mtl_attention_decode_enabled(void) {
    const char *value = getenv("DSM_ATTN_DECODE");
    return !value || strcmp(value, "0");
}
int dsm_mtl_attention_prefill_enabled(void) {
    const char *value = getenv("DSM_ATTN_PREFILL");
    return !value || strcmp(value, "0");
}
int dsm_mtl_attention_flash_enabled(void) {
    const char *value = getenv("DSM_ATTN_FA");
    return !value || strcmp(value, "0");
}
int dsm_mtl_qwen35_arch(const dsm_model *m) {
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
    if (!dsm_mtl_qwen35_arch(s->m) || s->checkpoints) return 1;
    s->checkpoint_count = checkpoint_ring_size();
    s->checkpoints = calloc(s->checkpoint_count, sizeof(*s->checkpoints));
    if (!s->checkpoints) return 0;
    for (uint32_t i = 0; i < s->checkpoint_count; ++i) {
        s->checkpoints[i].conv_state = dsm_mtl_buffer(s->device, s->conv_state.length);
        s->checkpoints[i].ssm_state = dsm_mtl_buffer(s->device, s->ssm_state.length);
        if (!s->checkpoints[i].conv_state || !s->checkpoints[i].ssm_state) return 0;
    }
    return 1;
}
static void invalidate_checkpoints_after(dsm_metal *s, uint32_t pos) {
    for (uint32_t i = 0; i < s->checkpoint_count; ++i)
        if (s->checkpoints[i].valid && s->checkpoints[i].pos > pos)
            s->checkpoints[i].valid = 0;
}
int dsm_mtl_tensor_view(const dsm_metal *s, const gguf_tensor *t, id<MTLBuffer> *buffer_out, uint64_t *offset_out) {
    uint64_t offset = (const uint8_t *)t->data - s->m->gguf.map;
    for (uint32_t i = 0; i < s->n_views; ++i) if (offset >= s->views[i].file_offset &&
        offset - s->views[i].file_offset <= s->views[i].bytes &&
        t->bytes <= s->views[i].bytes - (offset - s->views[i].file_offset)) {
        *buffer_out = s->views[i].buffer; *offset_out = offset - s->views[i].file_offset; return 1;
    }
    return 0;
}
void dsm_mtl_dispatch_1d(id<MTLComputeCommandEncoder> e, id<MTLComputePipelineState> p, uint32_t n) {
    NSUInteger w = MIN((NSUInteger)256, p.maxTotalThreadsPerThreadgroup);
    [e dispatchThreads:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(w, 1, 1)];
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
        if (!dsm_mtl_tensor_view(s, &m->gguf.tensors[i], &buffer, &offset)) {
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
        if (!dsm_mtl_expert_gemv_supported(w->gate_exps->type) ||
            !dsm_mtl_expert_gemv_supported(w->up_exps->type) ||
            !dsm_mtl_expert_gemv_supported(w->down_exps->type)) {
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
    s->mm_f32 = mmf_env ? strcmp(mmf_env, "0") != 0 : dsm_mtl_qwen35_arch(m);
    s->mm_f32w_enabled = f32_weight_matmul_enabled();
    const char *fuse_env = getenv("DSM_DECODE_FUSE");
    s->decode_fuse = !fuse_env || strcmp(fuse_env, "0") != 0;
    if (!create_weight_views(s, error, error_size)) goto bad;
    uint32_t ffn=m->is_moe?m->expert_hidden:m->hidden, slots=m->is_moe?m->experts_used:1, context_alloc=(context+63)&~UINT32_C(63);uint64_t kv=(uint64_t)(dsm_mtl_qwen35_arch(m) ? m->n_attn_layers : m->layers)*context_alloc*m->kv_heads*m->key_dim*4;s->x=dsm_mtl_buffer(s->device,m->dim*4);s->norm_x=dsm_mtl_buffer(s->device,m->dim*4);s->q=dsm_mtl_buffer(s->device,m->q_width*4);s->k=dsm_mtl_buffer(s->device,(dsm_mtl_qwen35_arch(m) ? (m->ssm_groups*m->ssm_state > m->kv_heads*m->key_dim ? m->ssm_groups*m->ssm_state : m->kv_heads*m->key_dim) : m->kv_heads*m->key_dim)*4);s->v=dsm_mtl_buffer(s->device,m->kv_heads*m->value_dim*4);s->att=dsm_mtl_buffer(s->device,m->attn_width*4);s->tmp=dsm_mtl_buffer(s->device,m->dim*slots*4);s->gate=dsm_mtl_buffer(s->device,ffn*slots*4);s->up=dsm_mtl_buffer(s->device,ffn*slots*4);s->route=m->is_moe?dsm_mtl_buffer(s->device,m->experts*4):NULL;s->expert_ids=m->is_moe?dsm_mtl_buffer(s->device,m->experts_used*4):NULL;s->expert_weights=m->is_moe?dsm_mtl_buffer(s->device,m->experts_used*4):NULL;s->scores=dsm_mtl_buffer(s->device,(uint64_t)context*m->heads*4);s->keys=dsm_mtl_buffer(s->device,kv);s->values=dsm_mtl_buffer(s->device,kv);s->logits=dsm_mtl_buffer(s->device,m->vocab*4);s->attention_partial=dsm_mtl_buffer(s->device,(uint64_t)m->heads*((context+63)/64)*(m->key_dim+2)*sizeof(float));s->qg=dsm_mtl_qwen35_arch(m)?dsm_mtl_buffer(s->device,(uint64_t)m->q_width*2*4):NULL;s->qkv_mixed=dsm_mtl_qwen35_arch(m)?dsm_mtl_buffer(s->device,(m->ssm_inner+2*m->ssm_groups*m->ssm_state)*sizeof(float)):NULL;s->conv_out=dsm_mtl_qwen35_arch(m)?dsm_mtl_buffer(s->device,(m->ssm_inner+2*m->ssm_groups*m->ssm_state)*sizeof(float)):NULL;s->z=dsm_mtl_qwen35_arch(m)?dsm_mtl_buffer(s->device,m->ssm_inner*4):NULL;s->alpha=dsm_mtl_qwen35_arch(m)?dsm_mtl_buffer(s->device,m->ssm_vheads*4):NULL;s->beta=dsm_mtl_qwen35_arch(m)?dsm_mtl_buffer(s->device,m->ssm_vheads*4):NULL;s->conv_state=dsm_mtl_qwen35_arch(m)?dsm_mtl_buffer(s->device,(uint64_t)m->layers*(m->ssm_inner+2*m->ssm_groups*m->ssm_state)*(m->ssm_conv-1)*4):NULL;s->ssm_state=dsm_mtl_qwen35_arch(m)?dsm_mtl_buffer(s->device,(uint64_t)m->layers*m->ssm_vheads*m->ssm_state*m->ssm_state*4):NULL;s->shared_gate=dsm_mtl_qwen35_arch(m)?dsm_mtl_buffer(s->device,4):NULL;s->batch_chunk=prefill_chunk_size(m);s->bx=dsm_mtl_buffer(s->device,(uint64_t)s->batch_chunk*m->dim*4);s->bnorm=dsm_mtl_buffer(s->device,(uint64_t)s->batch_chunk*m->dim*4);s->bq=dsm_mtl_buffer(s->device,(uint64_t)s->batch_chunk*m->q_width*4);s->bk=dsm_mtl_buffer(s->device,(uint64_t)s->batch_chunk*m->kv_heads*m->key_dim*4);s->bv=dsm_mtl_buffer(s->device,(uint64_t)s->batch_chunk*m->kv_heads*m->value_dim*4);s->batt=dsm_mtl_buffer(s->device,(uint64_t)s->batch_chunk*m->attn_width*4);s->btmp=dsm_mtl_buffer(s->device,(uint64_t)s->batch_chunk*m->dim*slots*4);s->bgate=dsm_mtl_buffer(s->device,(uint64_t)s->batch_chunk*ffn*slots*4);s->bup=dsm_mtl_buffer(s->device,(uint64_t)s->batch_chunk*ffn*slots*4);s->broute=m->is_moe?dsm_mtl_buffer(s->device,(uint64_t)s->batch_chunk*m->experts*4):NULL;s->bids=m->is_moe?dsm_mtl_buffer(s->device,(uint64_t)s->batch_chunk*m->experts_used*4):NULL;s->bw=m->is_moe?dsm_mtl_buffer(s->device,(uint64_t)s->batch_chunk*m->experts_used*4):NULL;uint32_t max_tiles=m->is_moe?dsm_mtl_matmul_id_max_tiles(m->experts,m->experts_used,s->batch_chunk):0;s->mm_id_tpe=m->is_moe?dsm_mtl_buffer(s->device,(uint64_t)(1+3*max_tiles)*sizeof(uint32_t)):NULL;s->mm_id_hids=m->is_moe?dsm_mtl_buffer(s->device,(uint64_t)(m->experts_used*s->batch_chunk+m->experts)*sizeof(int32_t)):NULL;if(!s->x||!s->norm_x||!s->q||!s->k||!s->v||!s->att||!s->tmp||!s->gate||!s->up||(m->is_moe&&(!s->route||!s->expert_ids||!s->expert_weights))||!s->scores||!s->keys||!s->values||!s->logits||!s->attention_partial||(dsm_mtl_qwen35_arch(m)&&(!s->qg||!s->qkv_mixed||!s->conv_out||!s->z||!s->alpha||!s->beta||!s->conv_state||!s->ssm_state||!s->shared_gate))||!s->bx||!s->bnorm||!s->bq||!s->bk||!s->bv||!s->batt||!s->btmp||!s->bgate||!s->bup||(m->is_moe&&(!s->broute||!s->bids||!s->bw||!s->mm_id_tpe||!s->mm_id_hids))){set_error(error,error_size,"cannot allocate Metal activations");goto bad;}
    if (dsm_mtl_qwen35_arch(m)) {
        uint32_t qk = m->ssm_groups * m->ssm_state;
        uint32_t mixed = m->ssm_inner + 2 * qk;
        uint32_t mix = m->ssm_inner > m->attn_width ? m->ssm_inner : m->attn_width;
        uint64_t rows = s->batch_chunk;
        s->hb_qkv = dsm_mtl_buffer(s->device, rows * mixed * sizeof(float));
        s->hb_conv = dsm_mtl_buffer(s->device, rows * mixed * sizeof(float));
        s->hb_dq = dsm_mtl_buffer(s->device, rows * qk * sizeof(float));
        s->hb_dk = dsm_mtl_buffer(s->device, rows * qk * sizeof(float));
        s->hb_z = dsm_mtl_buffer(s->device, rows * m->ssm_inner * sizeof(float));
        s->hb_g = dsm_mtl_buffer(s->device, rows * m->ssm_vheads * sizeof(float));
        s->hb_beta = dsm_mtl_buffer(s->device, rows * m->ssm_vheads * sizeof(float));
        s->hb_mix = dsm_mtl_buffer(s->device, rows * mix * sizeof(float));
        s->hb_qg = dsm_mtl_buffer(s->device, rows * 2 * m->q_width * sizeof(float));
        s->hb_aq = dsm_mtl_buffer(s->device, rows * m->q_width * sizeof(float));
        s->hb_agate = dsm_mtl_buffer(s->device, rows * m->q_width * sizeof(float));
        s->hb_ak = dsm_mtl_buffer(s->device, rows * m->kv_heads * m->key_dim * sizeof(float));
        s->hb_av = dsm_mtl_buffer(s->device, rows * m->kv_heads * m->value_dim * sizeof(float));
        s->hb_out = dsm_mtl_buffer(s->device, rows * m->dim * sizeof(float));
        uint32_t ffn_width = m->is_moe ? m->shared_hidden : m->hidden;
        s->hb_shg = m->is_moe ? dsm_mtl_buffer(s->device, rows * sizeof(float)) : NULL;
        s->hb_sgate = dsm_mtl_buffer(s->device, rows * ffn_width * sizeof(float));
        s->hb_sup = dsm_mtl_buffer(s->device, rows * ffn_width * sizeof(float));
        s->hb_sdown = dsm_mtl_buffer(s->device, rows * m->dim * sizeof(float));
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
void dsm_metal_free(dsm_metal *s){if(!s)return;if(s->profile.enabled)dsm_mtl_profile_print(s);free(s->checkpoints);free(s->views);free(s);}
uint32_t dsm_metal_pos(const dsm_metal *s){return s->pos;}
int dsm_metal_checkpoint(dsm_metal *s) {
    if (!s || !dsm_mtl_qwen35_arch(s->m)) return s != NULL;
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
    if (!dsm_mtl_qwen35_arch(s->m)) { s->pos = pos; return pos; }
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
