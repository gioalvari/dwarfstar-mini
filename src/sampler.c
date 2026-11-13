#include "sampler.h"

#include <math.h>
#include <stdlib.h>

typedef struct {
    float p;
    uint32_t id;
} candidate;

static int compare(const void *a, const void *b) {
    const candidate *x = a;
    const candidate *y = b;
    return x->p < y->p ? 1 : x->p > y->p ? -1 : 0;
}

static uint64_t next(dsm_sampler *sampler) {
    sampler->state ^= sampler->state << 13;
    sampler->state ^= sampler->state >> 7;
    sampler->state ^= sampler->state << 17;
    return sampler->state;
}

void dsm_sampler_init(dsm_sampler *sampler, float temperature, float top_p,
                      uint32_t top_k, float min_p, uint64_t seed,
                      float presence_penalty, float frequency_penalty) {
    *sampler = (dsm_sampler){
        .temperature = temperature,
        .top_p = top_p,
        .min_p = min_p,
        .presence_penalty = presence_penalty,
        .frequency_penalty = frequency_penalty,
        .top_k = top_k,
        .state = seed ? seed : 1,
    };
}

void dsm_sampler_free(dsm_sampler *sampler) {
    free(sampler->counts);
    sampler->counts = NULL;
    sampler->count_size = 0;
}

static int observe(dsm_sampler *sampler, uint32_t id, uint32_t n) {
    if (!sampler->presence_penalty && !sampler->frequency_penalty) return 1;
    if (sampler->count_size != n) {
        uint32_t *counts = calloc(n, sizeof(*counts));
        if (!counts) return 0;
        free(sampler->counts);
        sampler->counts = counts;
        sampler->count_size = n;
    }
    ++sampler->counts[id];
    return 1;
}

static float score(const dsm_sampler *sampler, const float *logits,
                   uint32_t id) {
    float value = logits[id];
    if (sampler->counts && id < sampler->count_size) {
        value -= sampler->presence_penalty * (sampler->counts[id] > 0);
        value -= sampler->frequency_penalty * sampler->counts[id];
    }
    return value;
}

int32_t dsm_sample(dsm_sampler *sampler, const float *logits, uint32_t n) {
    uint32_t best = 0;
    for (uint32_t i = 1; i < n; ++i) {
        if (score(sampler, logits, i) > score(sampler, logits, best)) best = i;
    }
    if (sampler->temperature <= 0) {
        observe(sampler, best, n);
        return (int32_t)best;
    }

    candidate *candidates = malloc(n * sizeof(*candidates));
    if (!candidates) return (int32_t)best;
    float max = score(sampler, logits, best);
    float normalizer = 0;
    for (uint32_t i = 0; i < n; ++i) {
        float logit = score(sampler, logits, i);
        candidates[i] = (candidate){.p = expf((logit - max) / sampler->temperature), .id = i};
        normalizer += candidates[i].p;
    }
    for (uint32_t i = 0; i < n; ++i) candidates[i].p /= normalizer;
    qsort(candidates, n, sizeof(*candidates), compare);

    uint32_t limit = sampler->top_k && sampler->top_k < n ? sampler->top_k : n;
    if (sampler->min_p > 0) {
        float minimum = candidates[0].p * sampler->min_p;
        while (limit > 1 && candidates[limit - 1].p < minimum) --limit;
    }
    float cutoff = sampler->top_p > 0 && sampler->top_p < 1 ? sampler->top_p : 1;
    float sum = 0;
    uint32_t keep = 0;
    while (keep < limit && sum < cutoff) sum += candidates[keep++].p;

    float random = (float)(next(sampler) >> 11) * (1.0f / 9007199254740992.0f) * sum;
    float accumulated = 0;
    uint32_t id = candidates[keep - 1].id;
    for (uint32_t i = 0; i < keep; ++i) {
        accumulated += candidates[i].p;
        if (random <= accumulated) {
            id = candidates[i].id;
            break;
        }
    }
    free(candidates);
    observe(sampler, id, n);
    return (int32_t)id;
}
