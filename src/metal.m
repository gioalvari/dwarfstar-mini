#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "metal.h"
#include "quants.h"

#include <libgen.h>
#include <mach-o/dyld.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

typedef struct { uint32_t in, out; } gemv_params;
typedef struct { uint32_t width, groups; float eps; } norm_params;
typedef struct { uint32_t heads, dim, pos; float base; } rope_params;
typedef struct { uint32_t layer, pos, context, width; } cache_params;
typedef struct { uint32_t heads, kv_heads, dim, pos, context, layer; } attn_params;

typedef struct { id<MTLBuffer> buffer; uint64_t file_offset, bytes; } weight_view;
struct dsm_metal {
    const dsm_model *m; uint32_t pos, context, n_views;
    id<MTLDevice> device; id<MTLCommandQueue> queue;
    id<MTLComputePipelineState> q8, q2k, q3k, q4k, q5k, q6k, f32, norm, rope, cache, attention, add, swiglu;
    weight_view *views;
    id<MTLBuffer> x, norm_x, q, k, v, att, tmp, gate, up, scores, keys, values, logits;
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
    id<MTLComputePipelineState> state = s->f32;
    switch (w->type) {
        case GGML_TYPE_Q8_0: state = s->q8; break;
        case GGML_TYPE_Q2_K: state = s->q2k; break;
        case GGML_TYPE_Q3_K: state = s->q3k; break;
        case GGML_TYPE_Q4_K: state = s->q4k; break;
        case GGML_TYPE_Q5_K: state = s->q5k; break;
        case GGML_TYPE_Q6_K: state = s->q6k; break;
    }
    [e setComputePipelineState:state]; [e setBuffer:wb offset:(NSUInteger)off atIndex:0]; [e setBuffer:x offset:0 atIndex:1]; [e setBuffer:y offset:0 atIndex:2]; [e setBytes:&p length:sizeof(p) atIndex:3];
    [e setThreadgroupMemoryLength:256 * sizeof(float) atIndex:0]; [e dispatchThreadgroups:MTLSizeMake(p.out, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
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
    dsm_metal *s=calloc(1,sizeof(*s)); if(!s){set_error(error,error_size,"out of memory");return 0;} s->m=m;s->context=context;s->device=MTLCreateSystemDefaultDevice(); if(!s->device){set_error(error,error_size,"no Metal device");goto bad;}s->queue=[s->device newCommandQueue];
    NSError *ns_error=nil; NSString *source=[NSString stringWithContentsOfFile:shader_path() encoding:NSUTF8StringEncoding error:&ns_error];if(!source){set_error(error,error_size,ns_error.localizedDescription.UTF8String);goto bad;}MTLCompileOptions *options=[MTLCompileOptions new];if (@available(macOS 15.0,*)) options.mathMode=MTLMathModeSafe;
    id<MTLLibrary> lib=[s->device newLibraryWithSource:source options:options error:&ns_error];if(!lib){set_error(error,error_size,ns_error.localizedDescription.UTF8String);goto bad;}
    s->q8=pipeline(s->device,lib,@"q8_gemv",error,error_size);s->q2k=pipeline(s->device,lib,@"q2k_gemv",error,error_size);s->q3k=pipeline(s->device,lib,@"q3k_gemv",error,error_size);s->q4k=pipeline(s->device,lib,@"q4k_gemv",error,error_size);s->q5k=pipeline(s->device,lib,@"q5k_gemv",error,error_size);s->q6k=pipeline(s->device,lib,@"q6k_gemv",error,error_size);s->f32=pipeline(s->device,lib,@"f32_gemv",error,error_size);s->norm=pipeline(s->device,lib,@"rmsnorm",error,error_size);s->rope=pipeline(s->device,lib,@"rope_neox",error,error_size);s->cache=pipeline(s->device,lib,@"cache_write",error,error_size);s->attention=pipeline(s->device,lib,@"attention_decode",error,error_size);s->add=pipeline(s->device,lib,@"add_inplace",error,error_size);s->swiglu=pipeline(s->device,lib,@"swiglu",error,error_size);if(!s->q8||!s->q2k||!s->q3k||!s->q4k||!s->q5k||!s->q6k||!s->f32||!s->norm||!s->rope||!s->cache||!s->attention||!s->add||!s->swiglu)goto bad;
    const uint64_t page=(uint64_t)getpagesize(), start=m->gguf.tensor_data_offset&~(page-1), end=m->gguf.size, max=[s->device maxBufferLength]&~(page-1);if(!max){set_error(error,error_size,"Metal max buffer length is zero");goto bad;}s->n_views=(uint32_t)((end-start+max-1)/max);s->views=calloc(s->n_views,sizeof(*s->views));if(!s->views){set_error(error,error_size,"out of memory");goto bad;}for(uint32_t i=0;i<s->n_views;++i){uint64_t off=start+(uint64_t)i*max,bytes=end-off<max?end-off:max;bytes=(bytes+page-1)&~(page-1);s->views[i].buffer=[s->device newBufferWithBytesNoCopy:(void *)(m->gguf.map+off) length:(NSUInteger)bytes options:MTLResourceStorageModeShared deallocator:nil];s->views[i].file_offset=off;s->views[i].bytes=bytes;if(!s->views[i].buffer){set_error(error,error_size,"cannot wrap GGUF mmap in Metal buffer");goto bad;}}
    uint64_t kv=(uint64_t)m->layers*context*m->kv_heads*m->key_dim*4;s->x=buffer(s->device,m->dim*4);s->norm_x=buffer(s->device,m->dim*4);s->q=buffer(s->device,m->dim*4);s->k=buffer(s->device,m->kv_heads*m->key_dim*4);s->v=buffer(s->device,m->kv_heads*m->value_dim*4);s->att=buffer(s->device,m->dim*4);s->tmp=buffer(s->device,m->dim*4);s->gate=buffer(s->device,m->hidden*4);s->up=buffer(s->device,m->hidden*4);s->scores=buffer(s->device,(uint64_t)context*m->heads*4);s->keys=buffer(s->device,kv);s->values=buffer(s->device,kv);s->logits=buffer(s->device,m->vocab*4);if(!s->x||!s->norm_x||!s->q||!s->k||!s->v||!s->att||!s->tmp||!s->gate||!s->up||!s->scores||!s->keys||!s->values||!s->logits){set_error(error,error_size,"cannot allocate Metal activations");goto bad;}*out=s;return 1;
bad:dsm_metal_free(s);return 0;
}
void dsm_metal_free(dsm_metal *s){if(!s)return;free(s->views);free(s);}
uint32_t dsm_metal_pos(const dsm_metal *s){return s->pos;}
int dsm_metal_matvec_test(dsm_metal *s,const gguf_tensor *w,const float *input,float *output){if(!s||!w||!dsm_quant_supported(w->type))return 0;id<MTLBuffer> x=buffer(s->device,w->dims[0]*sizeof(float)),y=buffer(s->device,w->dims[1]*sizeof(float));if(!x||!y)return 0;memcpy(x.contents,input,w->dims[0]*sizeof(float));id<MTLCommandBuffer> cb=[s->queue commandBuffer];id<MTLComputeCommandEncoder> e=[cb computeCommandEncoder];encode_gemv(s,e,w,x,y);[e endEncoding];[cb commit];[cb waitUntilCompleted];if(cb.status!=MTLCommandBufferStatusCompleted)return 0;memcpy(output,y.contents,w->dims[1]*sizeof(float));return 1;}
int dsm_metal_token(dsm_metal *s,int32_t token,float *logits){const dsm_model *m=s->m;if(token<0||(uint32_t)token>=m->vocab||s->pos>=s->context)return 0;const ggml_type_info *info=gguf_type_info(m->token_embd->type);const uint8_t *row=(const uint8_t *)m->token_embd->data+(uint64_t)token*(m->dim/info->block_elems)*info->block_bytes;float *x=s->x.contents;dsm_dequant_row(m->token_embd->type,row,x,m->dim);
    id<MTLCommandBuffer> cb=[s->queue commandBuffer];id<MTLComputeCommandEncoder> e=[cb computeCommandEncoder];uint32_t kvw=m->kv_heads*m->key_dim;for(uint32_t l=0;l<m->layers;++l){const dsm_layer *w=&m->layer[l];encode_norm(s,e,s->x,w->attn_norm,s->norm_x,m->dim,1);encode_gemv(s,e,w->q,s->norm_x,s->q);encode_gemv(s,e,w->k,s->norm_x,s->k);encode_gemv(s,e,w->v,s->norm_x,s->v);encode_norm(s,e,s->q,w->q_norm,s->q,m->key_dim,m->heads);encode_norm(s,e,s->k,w->k_norm,s->k,m->key_dim,m->kv_heads);encode_rope(s,e,s->q,m->heads);encode_rope(s,e,s->k,m->kv_heads);cache_params cp={l,s->pos,s->context,kvw};[e setComputePipelineState:s->cache];[e setBuffer:s->k offset:0 atIndex:0];[e setBuffer:s->keys offset:0 atIndex:1];[e setBytes:&cp length:sizeof(cp) atIndex:2];dispatch_1d(e,s->cache,kvw);[e setBuffer:s->v offset:0 atIndex:0];[e setBuffer:s->values offset:0 atIndex:1];[e setBytes:&cp length:sizeof(cp) atIndex:2];dispatch_1d(e,s->cache,kvw);attn_params ap={m->heads,m->kv_heads,m->key_dim,s->pos,s->context,l};[e setComputePipelineState:s->attention];[e setBuffer:s->q offset:0 atIndex:0];[e setBuffer:s->keys offset:0 atIndex:1];[e setBuffer:s->values offset:0 atIndex:2];[e setBuffer:s->att offset:0 atIndex:3];[e setBuffer:s->scores offset:0 atIndex:4];[e setBytes:&ap length:sizeof(ap) atIndex:5];[e dispatchThreadgroups:MTLSizeMake(m->heads,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];encode_gemv(s,e,w->o,s->att,s->tmp);encode_simple(e,s->add,s->x,s->tmp,m->dim);encode_norm(s,e,s->x,w->ffn_norm,s->norm_x,m->dim,1);encode_gemv(s,e,w->gate,s->norm_x,s->gate);encode_gemv(s,e,w->up,s->norm_x,s->up);encode_simple(e,s->swiglu,s->gate,s->up,m->hidden);encode_gemv(s,e,w->down,s->gate,s->tmp);encode_simple(e,s->add,s->x,s->tmp,m->dim);}encode_norm(s,e,s->x,m->output_norm,s->norm_x,m->dim,1);encode_gemv(s,e,m->output,s->norm_x,s->logits);[e endEncoding];[cb commit];[cb waitUntilCompleted];if(cb.status!=MTLCommandBufferStatusCompleted)return 0;memcpy(logits,s->logits.contents,m->vocab*4);++s->pos;return 1;}
