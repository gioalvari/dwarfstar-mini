#ifndef DSM_SAMPLER_H
#define DSM_SAMPLER_H

#include <stdint.h>

typedef struct { float temperature, top_p; uint64_t state; } dsm_sampler;
void dsm_sampler_init(dsm_sampler *sampler, float temperature, float top_p, uint64_t seed);
int32_t dsm_sample(dsm_sampler *sampler, const float *logits, uint32_t n);

#endif
