#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "metal.h"
#include "quants.h"

#include <libgen.h>
#include <mach-o/dyld.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

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

typedef struct { id<MTLBuffer> buffer; uint64_t file_offset, bytes; } weight_view;
typedef enum {
    PROFILE_QKV_GEMV,
    PROFILE_QK_NORM_ROPE,
    PROFILE_KV_STORE,
    PROFILE_ATTENTION,
    PROFILE_O_GEMV,
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
    id<MTLComputePipelineState> q8, q2k, q3k, q4k, q5k, q6k, f32, norm, rope, cache, attention, add, swiglu;
    weight_view *views;
    id<MTLBuffer> x, norm_x, q, k, v, att, tmp, gate, up, scores, keys, values, logits;
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
    for (uint32_t i = 0; i < s->n_views; ++i) if (offset >= s->views[i].file_offset && t->bytes <= s->views[i].bytes - (offset - s->views[i].file_offset)) {
        *buffer_out = s->views[i].buffer; *offset_out = offset - s->views[i].file_offset; return 1;
    }
    return 0;
}
static void dispatch_1d(id<MTLComputeCommandEncoder> e, id<MTLComputePipelineState> p, uint32_t n) {
    NSUInteger w = MIN((NSUInteger)256, p.maxTotalThreadsPerThreadgroup);
    [e dispatchThreads:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(w, 1, 1)];
}
static void encode_gemv(dsm_metal *s, id<MTLComputeCommandEncoder> e, const gguf_tensor *w, id<MTLBuffer> x, id<MTLBuffer> y) {
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
    [e setComputePipelineState:state]; [e setBuffer:wb offset:(NSUInteger)off atIndex:0]; [e setBuffer:x offset:0 atIndex:1]; [e setBuffer:y offset:0 atIndex:2];
    if (w->type == GGML_TYPE_Q8_0) {
        [e setBytes:&mul_mv length:sizeof(mul_mv) atIndex:3];
        [e setThreadgroupMemoryLength:32 * sizeof(float) * 2 atIndex:0];
        [e dispatchThreadgroups:MTLSizeMake((p.out + 1) / 2, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 4, 1)];
    } else if (w->type == GGML_TYPE_Q2_K) {
        [e setBytes:&mul_mv length:sizeof(mul_mv) atIndex:3];
        [e dispatchThreadgroups:MTLSizeMake((p.out + 7) / 8, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 2, 1)];
    } else if (w->type == GGML_TYPE_Q3_K || w->type == GGML_TYPE_Q5_K) {
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
static void encode_norm(dsm_metal *s, id<MTLComputeCommandEncoder> e, id<MTLBuffer> x, const gguf_tensor *w, id<MTLBuffer> out, uint32_t width, uint32_t groups) {
    id<MTLBuffer> wb; uint64_t off; if (!tensor_view(s,w,&wb,&off)) return; norm_params p={width,groups,s->m->rms_eps};
    [e setComputePipelineState:s->norm];[e setBuffer:x offset:0 atIndex:0];[e setBuffer:wb offset:(NSUInteger)off atIndex:1];[e setBuffer:out offset:0 atIndex:2];[e setBytes:&p length:sizeof(p) atIndex:3];[e setThreadgroupMemoryLength:256 * sizeof(float) atIndex:0];[e dispatchThreadgroups:MTLSizeMake(groups,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
}
static void encode_rope(dsm_metal *s,id<MTLComputeCommandEncoder> e,id<MTLBuffer> x,uint32_t heads) { rope_params p={heads,s->m->key_dim,s->pos,s->m->rope_base};[e setComputePipelineState:s->rope];[e setBuffer:x offset:0 atIndex:0];[e setBytes:&p length:sizeof(p) atIndex:1];dispatch_1d(e,s->rope,heads*(s->m->key_dim/2)); }
static void encode_simple(id<MTLComputeCommandEncoder> e,id<MTLComputePipelineState> p,id<MTLBuffer> a,id<MTLBuffer> b,uint32_t n) { [e setComputePipelineState:p];[e setBuffer:a offset:0 atIndex:0];[e setBuffer:b offset:0 atIndex:1];[e setBytes:&n length:4 atIndex:2];dispatch_1d(e,p,n); }

int dsm_metal_available(void) { return MTLCreateSystemDefaultDevice() != nil; }
int dsm_metal_init(dsm_metal **out,const dsm_model *m,uint32_t context,char *error,uint32_t error_size) {
    *out=NULL; if (!context || context > m->context) { set_error(error,error_size,"invalid Metal context"); return 0; }
    dsm_metal *s=calloc(1,sizeof(*s)); if(!s){set_error(error,error_size,"out of memory");return 0;} s->m=m;s->context=context;s->profile.enabled=getenv("DSM_PROFILE") && !strcmp(getenv("DSM_PROFILE"),"1");s->device=MTLCreateSystemDefaultDevice(); if(!s->device){set_error(error,error_size,"no Metal device");goto bad;}s->queue=[s->device newCommandQueue];
    NSError *ns_error=nil; NSString *source=[NSString stringWithContentsOfFile:shader_path() encoding:NSUTF8StringEncoding error:&ns_error];if(!source){set_error(error,error_size,ns_error.localizedDescription.UTF8String);goto bad;}MTLCompileOptions *options=[MTLCompileOptions new];if (@available(macOS 15.0,*)) options.mathMode=MTLMathModeSafe;
    id<MTLLibrary> lib=[s->device newLibraryWithSource:source options:options error:&ns_error];if(!lib){set_error(error,error_size,ns_error.localizedDescription.UTF8String);goto bad;}
    s->q8=pipeline(s->device,lib,@"q8_gemv",error,error_size);s->q2k=pipeline(s->device,lib,@"q2k_gemv",error,error_size);s->q3k=pipeline(s->device,lib,@"q3k_gemv",error,error_size);s->q4k=pipeline(s->device,lib,@"q4k_gemv",error,error_size);s->q5k=pipeline(s->device,lib,@"q5k_gemv",error,error_size);s->q6k=pipeline(s->device,lib,@"q6k_gemv",error,error_size);s->f32=pipeline(s->device,lib,@"f32_gemv",error,error_size);s->norm=pipeline(s->device,lib,@"rmsnorm",error,error_size);s->rope=pipeline(s->device,lib,@"rope_neox",error,error_size);s->cache=pipeline(s->device,lib,@"cache_write",error,error_size);s->attention=pipeline(s->device,lib,@"attention_decode",error,error_size);s->add=pipeline(s->device,lib,@"add_inplace",error,error_size);s->swiglu=pipeline(s->device,lib,@"swiglu",error,error_size);if(!s->q8||!s->q2k||!s->q3k||!s->q4k||!s->q5k||!s->q6k||!s->f32||!s->norm||!s->rope||!s->cache||!s->attention||!s->add||!s->swiglu)goto bad;
    const uint64_t page=(uint64_t)getpagesize(), start=m->gguf.tensor_data_offset&~(page-1), end=m->gguf.size, max=[s->device maxBufferLength]&~(page-1);if(!max){set_error(error,error_size,"Metal max buffer length is zero");goto bad;}s->n_views=(uint32_t)((end-start+max-1)/max);s->views=calloc(s->n_views,sizeof(*s->views));if(!s->views){set_error(error,error_size,"out of memory");goto bad;}for(uint32_t i=0;i<s->n_views;++i){uint64_t off=start+(uint64_t)i*max,bytes=end-off<max?end-off:max;bytes=(bytes+page-1)&~(page-1);s->views[i].buffer=[s->device newBufferWithBytesNoCopy:(void *)(m->gguf.map+off) length:(NSUInteger)bytes options:MTLResourceStorageModeShared deallocator:nil];s->views[i].file_offset=off;s->views[i].bytes=bytes;if(!s->views[i].buffer){set_error(error,error_size,"cannot wrap GGUF mmap in Metal buffer");goto bad;}}
    uint64_t kv=(uint64_t)m->layers*context*m->kv_heads*m->key_dim*4;s->x=buffer(s->device,m->dim*4);s->norm_x=buffer(s->device,m->dim*4);s->q=buffer(s->device,m->dim*4);s->k=buffer(s->device,m->kv_heads*m->key_dim*4);s->v=buffer(s->device,m->kv_heads*m->value_dim*4);s->att=buffer(s->device,m->dim*4);s->tmp=buffer(s->device,m->dim*4);s->gate=buffer(s->device,m->hidden*4);s->up=buffer(s->device,m->hidden*4);s->scores=buffer(s->device,(uint64_t)context*m->heads*4);s->keys=buffer(s->device,kv);s->values=buffer(s->device,kv);s->logits=buffer(s->device,m->vocab*4);if(!s->x||!s->norm_x||!s->q||!s->k||!s->v||!s->att||!s->tmp||!s->gate||!s->up||!s->scores||!s->keys||!s->values||!s->logits){set_error(error,error_size,"cannot allocate Metal activations");goto bad;}*out=s;return 1;
bad:dsm_metal_free(s);return 0;
}
void dsm_metal_free(dsm_metal *s){if(!s)return;if(s->profile.enabled)dsm_metal_profile_print(s);free(s->views);free(s);}
uint32_t dsm_metal_pos(const dsm_metal *s){return s->pos;}
int dsm_metal_truncate(dsm_metal *s,uint32_t pos){if(!s||pos>s->pos)return 0;s->pos=pos;return 1;}
int dsm_metal_matvec_test(dsm_metal *s,const gguf_tensor *w,const float *input,float *output){if(!s||!w||!dsm_quant_supported(w->type))return 0;id<MTLBuffer> x=buffer(s->device,w->dims[0]*sizeof(float)),y=buffer(s->device,w->dims[1]*sizeof(float));if(!x||!y)return 0;memcpy(x.contents,input,w->dims[0]*sizeof(float));id<MTLCommandBuffer> cb=[s->queue commandBuffer];id<MTLComputeCommandEncoder> e=[cb computeCommandEncoder];encode_gemv(s,e,w,x,y);[e endEncoding];[cb commit];[cb waitUntilCompleted];if(cb.status!=MTLCommandBufferStatusCompleted)return 0;memcpy(output,y.contents,w->dims[1]*sizeof(float));return 1;}
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
    PROFILE_ENCODE(PROFILE_FFN_GATE_UP_GEMV,2,w->gate->bytes+w->up->bytes,encode_gemv(s,e,w->gate,s->norm_x,s->gate);encode_gemv(s,e,w->up,s->norm_x,s->up));
    PROFILE_ENCODE(PROFILE_SWIGLU,1,0,encode_simple(e,s->swiglu,s->gate,s->up,m->hidden));
    PROFILE_ENCODE(PROFILE_FFN_DOWN_GEMV,1,w->down->bytes,encode_gemv(s,e,w->down,s->gate,s->tmp));
    PROFILE_ENCODE(PROFILE_RESIDUAL,1,0,encode_simple(e,s->add,s->x,s->tmp,m->dim));
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
int dsm_metal_token(dsm_metal *s,int32_t token,float *logits){const dsm_model *m=s->m;if(token<0||(uint32_t)token>=m->vocab||s->pos>=s->context)return 0;const ggml_type_info *info=gguf_type_info(m->token_embd->type);const uint8_t *row=(const uint8_t *)m->token_embd->data+(uint64_t)token*(m->dim/info->block_elems)*info->block_bytes;float *x=s->x.contents;dsm_dequant_row(m->token_embd->type,row,x,m->dim);if(s->profile.enabled)return profile_token(s,logits);
    id<MTLCommandBuffer> cb=[s->queue commandBuffer];id<MTLComputeCommandEncoder> e=[cb computeCommandEncoder];uint32_t kvw=m->kv_heads*m->key_dim;for(uint32_t l=0;l<m->layers;++l){const dsm_layer *w=&m->layer[l];encode_norm(s,e,s->x,w->attn_norm,s->norm_x,m->dim,1);encode_gemv(s,e,w->q,s->norm_x,s->q);encode_gemv(s,e,w->k,s->norm_x,s->k);encode_gemv(s,e,w->v,s->norm_x,s->v);encode_norm(s,e,s->q,w->q_norm,s->q,m->key_dim,m->heads);encode_norm(s,e,s->k,w->k_norm,s->k,m->key_dim,m->kv_heads);encode_rope(s,e,s->q,m->heads);encode_rope(s,e,s->k,m->kv_heads);cache_params cp={l,s->pos,s->context,kvw};[e setComputePipelineState:s->cache];[e setBuffer:s->k offset:0 atIndex:0];[e setBuffer:s->keys offset:0 atIndex:1];[e setBytes:&cp length:sizeof(cp) atIndex:2];dispatch_1d(e,s->cache,kvw);[e setBuffer:s->v offset:0 atIndex:0];[e setBuffer:s->values offset:0 atIndex:1];[e setBytes:&cp length:sizeof(cp) atIndex:2];dispatch_1d(e,s->cache,kvw);attn_params ap={m->heads,m->kv_heads,m->key_dim,s->pos,s->context,l};[e setComputePipelineState:s->attention];[e setBuffer:s->q offset:0 atIndex:0];[e setBuffer:s->keys offset:0 atIndex:1];[e setBuffer:s->values offset:0 atIndex:2];[e setBuffer:s->att offset:0 atIndex:3];[e setBuffer:s->scores offset:0 atIndex:4];[e setBytes:&ap length:sizeof(ap) atIndex:5];[e setThreadgroupMemoryLength:256*sizeof(float) atIndex:0];[e dispatchThreadgroups:MTLSizeMake(m->heads,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];encode_gemv(s,e,w->o,s->att,s->tmp);encode_simple(e,s->add,s->x,s->tmp,m->dim);encode_norm(s,e,s->x,w->ffn_norm,s->norm_x,m->dim,1);encode_gemv(s,e,w->gate,s->norm_x,s->gate);encode_gemv(s,e,w->up,s->norm_x,s->up);encode_simple(e,s->swiglu,s->gate,s->up,m->hidden);encode_gemv(s,e,w->down,s->gate,s->tmp);encode_simple(e,s->add,s->x,s->tmp,m->dim);}encode_norm(s,e,s->x,m->output_norm,s->norm_x,m->dim,1);encode_gemv(s,e,m->output,s->norm_x,s->logits);[e endEncoding];[cb commit];[cb waitUntilCompleted];if(cb.status!=MTLCommandBufferStatusCompleted)return 0;memcpy(logits,s->logits.contents,m->vocab*4);++s->pos;return 1;}
