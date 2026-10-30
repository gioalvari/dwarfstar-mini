#include "sampler.h"

#include <math.h>
#include <stdlib.h>

typedef struct { float p; uint32_t id; } candidate;
static int compare(const void *a,const void *b) { const candidate *x=a,*y=b; return x->p<y->p ? 1 : x->p>y->p ? -1 : 0; }
static uint64_t next(dsm_sampler *s){s->state^=s->state<<13;s->state^=s->state>>7;s->state^=s->state<<17;return s->state;}
void dsm_sampler_init(dsm_sampler *s,float temp,float top_p,uint64_t seed){s->temperature=temp;s->top_p=top_p;s->state=seed?seed:1;}
int32_t dsm_sample(dsm_sampler *s,const float *logits,uint32_t n){uint32_t best=0;for(uint32_t i=1;i<n;++i)if(logits[i]>logits[best])best=i;if(s->temperature<=0)return (int32_t)best;candidate *c=malloc(n*sizeof(*c));if(!c)return (int32_t)best;float max=logits[best],z=0;for(uint32_t i=0;i<n;++i){c[i]=(candidate){expf((logits[i]-max)/s->temperature),i};z+=c[i].p;}for(uint32_t i=0;i<n;++i)c[i].p/=z;qsort(c,n,sizeof(*c),compare);float cutoff=s->top_p>0&&s->top_p<1?s->top_p:1,sum=0;uint32_t keep=0;while(keep<n&&sum<cutoff)sum+=c[keep++].p;float r=(float)(next(s)>>11)*(1.0f/9007199254740992.0f)*sum,acc=0;for(uint32_t i=0;i<keep;++i){acc+=c[i].p;if(r<=acc){int32_t id=(int32_t)c[i].id;free(c);return id;}}int32_t id=(int32_t)c[keep-1].id;free(c);return id;}
