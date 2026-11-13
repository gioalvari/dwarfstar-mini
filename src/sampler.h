#ifndef DSM_SAMPLER_H
#define DSM_SAMPLER_H

#include <stdint.h>

typedef struct {
    float temperature;
    float top_p;
    float min_p;
    float presence_penalty;
    float frequency_penalty;
    uint32_t top_k;
    uint64_t state;
    uint32_t *counts;
    uint32_t count_size;
} dsm_sampler;

void dsm_sampler_init(dsm_sampler *sampler, float temperature, float top_p,
                      uint32_t top_k, float min_p, uint64_t seed,
                      float presence_penalty, float frequency_penalty);
void dsm_sampler_free(dsm_sampler *sampler);
int32_t dsm_sample(dsm_sampler *sampler, const float *logits, uint32_t n);

#endif
