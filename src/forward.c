#include "forward.h"
#include "quants.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

struct dsm_forward {
    const dsm_model *m; uint32_t pos, context;
    float *k_cache, *v_cache;
    float *x, *norm, *q, *k, *v, *att, *tmp, *gate, *up, *scores;
};
static float f16(uint16_t h) { union { uint32_t u; float f; } u; unsigned sign=h>>15, exp=(h>>10)&31, frac=h&1023; if(!exp) u.u=sign<<31; else if(exp==31) u.u=(sign<<31)|0x7f800000|(frac<<13); else u.u=(sign<<31)|((exp+112)<<23)|(frac<<13); return u.f; }
static float weight_at(const gguf_tensor *w, uint64_t row, uint64_t col) {
    const ggml_type_info *info=gguf_type_info(w->type); float out[256]; uint64_t block=col/info->block_elems;
    dsm_dequant_row(w->type,(const uint8_t *)w->data+(row*(w->dims[0]/info->block_elems)+block)*info->block_bytes,out,info->block_elems); return out[col%info->block_elems];
}
static void matvec(const gguf_tensor *w, const float *x, float *y) {
    uint64_t in=w->dims[0], out=w->dims[1]; const ggml_type_info *info=gguf_type_info(w->type); uint64_t bytes=in/info->block_elems*info->block_bytes; for(uint64_t r=0;r<out;++r) y[r]=dsm_dot_row(w->type,(const uint8_t *)w->data+r*bytes,x,in);
}
static void rmsnorm(float *out,const float *in,const gguf_tensor *weight,uint32_t n,float eps) { float ss=0;for(uint32_t i=0;i<n;++i)ss+=in[i]*in[i]; float scale=1.0f/sqrtf(ss/n+eps); const float *w=weight?(const float *)weight->data:NULL;for(uint32_t i=0;i<n;++i)out[i]=in[i]*scale*(w?w[i]:1); }
static void rope(float *x,uint32_t heads,uint32_t dim,uint32_t pos,float base) { for(uint32_t h=0;h<heads;++h)for(uint32_t i=0;i<dim/2;++i){float a=x[h*dim+i],b=x[h*dim+i+dim/2],theta=(float)pos/powf(base,(float)i/(dim/2));float c=cosf(theta),s=sinf(theta);x[h*dim+i]=a*c-b*s;x[h*dim+i+dim/2]=a*s+b*c;} }
int dsm_forward_init(dsm_forward **out,const dsm_model *m,uint32_t context) { *out=NULL; if(!context||context>m->context)return 0; dsm_forward *s=calloc(1,sizeof(*s)); if(!s)return 0;s->m=m;s->context=context;uint64_t kv=(uint64_t)m->layers*context*m->kv_heads*m->key_dim;s->k_cache=malloc(kv*4);s->v_cache=malloc(kv*4);s->x=malloc(m->dim*4);s->norm=malloc(m->dim*4);s->q=malloc(m->dim*4);s->k=malloc(m->kv_heads*m->key_dim*4);s->v=malloc(m->kv_heads*m->value_dim*4);s->att=malloc(m->dim*4);s->tmp=malloc(m->dim*4);s->gate=malloc(m->hidden*4);s->up=malloc(m->hidden*4);s->scores=malloc(context*4);if(!s->k_cache||!s->v_cache||!s->x||!s->norm||!s->q||!s->k||!s->v||!s->att||!s->tmp||!s->gate||!s->up||!s->scores){dsm_forward_free(s);return 0;}*out=s;return 1; }
void dsm_forward_free(dsm_forward *s){if(!s)return;free(s->k_cache);free(s->v_cache);free(s->x);free(s->norm);free(s->q);free(s->k);free(s->v);free(s->att);free(s->tmp);free(s->gate);free(s->up);free(s->scores);free(s);}
uint32_t dsm_forward_pos(const dsm_forward *s){return s->pos;}
int dsm_forward_token(dsm_forward *s,int32_t token,float *logits) { const dsm_model *m=s->m;if(token<0||(uint32_t)token>=m->vocab||s->pos>=s->context)return 0;for(uint32_t i=0;i<m->dim;++i)s->x[i]=weight_at(m->token_embd,(uint32_t)token,i);uint32_t hd=m->key_dim,kvw=m->kv_heads*hd;for(uint32_t l=0;l<m->layers;++l){const dsm_layer *w=&m->layer[l];rmsnorm(s->norm,s->x,w->attn_norm,m->dim,m->rms_eps);matvec(w->q,s->norm,s->q);matvec(w->k,s->norm,s->k);matvec(w->v,s->norm,s->v);for(uint32_t h=0;h<m->heads;++h)rmsnorm(s->q+h*hd,s->q+h*hd,w->q_norm,hd,m->rms_eps);for(uint32_t h=0;h<m->kv_heads;++h)rmsnorm(s->k+h*hd,s->k+h*hd,w->k_norm,hd,m->rms_eps);rope(s->q,m->heads,hd,s->pos,m->rope_base);rope(s->k,m->kv_heads,hd,s->pos,m->rope_base);uint64_t off=((uint64_t)l*s->context+s->pos)*kvw;memcpy(s->k_cache+off,s->k,kvw*4);memcpy(s->v_cache+off,s->v,kvw*4);for(uint32_t h=0;h<m->heads;++h){uint32_t kh=h/(m->heads/m->kv_heads);float max=-INFINITY;for(uint32_t p=0;p<=s->pos;++p){const float *kc=s->k_cache+((uint64_t)l*s->context+p)*kvw+kh*hd;float d=0;for(uint32_t j=0;j<hd;++j)d+=s->q[h*hd+j]*kc[j];s->scores[p]=d/sqrtf((float)hd);if(s->scores[p]>max)max=s->scores[p];}float z=0;for(uint32_t p=0;p<=s->pos;++p){s->scores[p]=expf(s->scores[p]-max);z+=s->scores[p];}for(uint32_t j=0;j<hd;++j){float a=0;for(uint32_t p=0;p<=s->pos;++p)a+=s->scores[p]/z*s->v_cache[((uint64_t)l*s->context+p)*kvw+kh*hd+j];s->att[h*hd+j]=a;}}matvec(w->o,s->att,s->tmp);for(uint32_t i=0;i<m->dim;++i)s->x[i]+=s->tmp[i];rmsnorm(s->norm,s->x,w->ffn_norm,m->dim,m->rms_eps);matvec(w->gate,s->norm,s->gate);matvec(w->up,s->norm,s->up);for(uint32_t i=0;i<m->hidden;++i){float a=s->gate[i];s->gate[i]=(a/(1+expf(-a)))*s->up[i];}matvec(w->down,s->gate,s->tmp);for(uint32_t i=0;i<m->dim;++i)s->x[i]+=s->tmp[i];}rmsnorm(s->norm,s->x,m->output_norm,m->dim,m->rms_eps);matvec(m->output,s->norm,logits);++s->pos;return 1; }
