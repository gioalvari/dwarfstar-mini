#include "metal.h"
#include "model.h"
#include "sampler.h"
#include "tokenizer.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void usage(const char *argv0) {
    fprintf(stderr, "usage: %s MODEL.gguf PROMPT [--dump OUT.bin] [--n N] [--split K]\n", argv0);
}

enum { TOP_K = 5, GREEDY_TOKENS = 64 };

static uint32_t argmax(const float *values, uint32_t n) {
    uint32_t best = 0;
    for (uint32_t i = 1; i < n; ++i) if (values[i] > values[best]) best = i;
    return best;
}

static void top_k(const float *values, uint32_t n, uint32_t *out) {
    for (uint32_t k = 0; k < TOP_K; ++k) out[k] = UINT32_MAX;
    for (uint32_t i = 0; i < n; ++i) {
        uint32_t k = TOP_K;
        while (k > 0 && (out[k - 1] == UINT32_MAX || values[i] > values[out[k - 1]])) --k;
        if (k == TOP_K) continue;
        memmove(out + k + 1, out + k, (TOP_K - k - 1) * sizeof(*out));
        out[k] = i;
    }
}

static int same_top_k(const float *a, const float *b, uint32_t n) {
    uint32_t ta[TOP_K], tb[TOP_K];
    top_k(a, n, ta);
    top_k(b, n, tb);
    for (uint32_t i = 0; i < TOP_K; ++i) {
        int found = 0;
        for (uint32_t j = 0; j < TOP_K; ++j) found |= ta[i] == tb[j];
        if (!found) return 0;
    }
    return 1;
}

int main(int argc, char **argv) {
    const char *model_path;
    const char *prompt;
    const char *dump = NULL;
    uint32_t limit = 0, split = 0;
    dsm_model model;
    dsm_tokenizer *tokenizer = NULL;
    dsm_tokens tokens = {0};
    dsm_metal *metal = NULL;
    dsm_metal *prefill_metal = NULL;
    float *reference = NULL;
    float *prefill = NULL;
    char error[256];
    int status = 1;

    if (argc < 3) {
        usage(argv[0]);
        return 2;
    }
    model_path = argv[1];
    prompt = argv[2];
    for (int arg = 3; arg < argc; ++arg) {
        if (!strcmp(argv[arg], "--dump") && arg + 1 < argc) dump = argv[++arg];
        else if (!strcmp(argv[arg], "--n") && arg + 1 < argc)
            limit = (uint32_t)strtoul(argv[++arg], NULL, 10);
        else if (!strcmp(argv[arg], "--split") && arg + 1 < argc)
            split = (uint32_t)strtoul(argv[++arg], NULL, 10);
        else { usage(argv[0]); return 2; }
    }
    if (!dsm_model_open(&model, model_path)) {
        fprintf(stderr, "%s\n", model.error);
        goto done;
    }
    if (!dsm_tokenizer_init(&tokenizer, &model.gguf, error, sizeof(error))) {
        fprintf(stderr, "%s\n", error);
        goto done;
    }
    if (!dsm_tokenize(tokenizer, prompt, 0, &tokens) || !tokens.len) {
        fprintf(stderr, "tokenization failed\n");
        goto done;
    }
    if (limit) {
        if (limit > tokens.len) { fprintf(stderr, "--n exceeds tokenized prompt length\n"); goto done; }
        tokens.len = limit;
    }
    if (split >= tokens.len) { fprintf(stderr, "--split must be within the prompt\n"); goto done; }
    if (!dsm_metal_init(&metal, &model, (uint32_t)tokens.len + GREEDY_TOKENS, error,
                        sizeof(error))) {
        fprintf(stderr, "%s\n", error);
        goto done;
    }
    reference = malloc((size_t)model.vocab * sizeof(*reference));
    prefill = malloc((size_t)model.vocab * sizeof(*prefill));
    if (!reference || !prefill) {
        fprintf(stderr, "out of memory\n");
        goto done;
    }
    for (size_t i = 0; i < tokens.len; ++i) {
        if (!dsm_metal_token(metal, tokens.ids[i], reference)) {
            fprintf(stderr, "Metal token inference failed at %zu\n", i);
            goto done;
        }
    }
    if (!dsm_metal_init(&prefill_metal, &model, (uint32_t)tokens.len + GREEDY_TOKENS,
                        error, sizeof(error)) ||
        !(split ? (dsm_metal_prefill(prefill_metal, tokens.ids, split, prefill) &&
                   dsm_metal_prefill(prefill_metal, tokens.ids + split,
                                     (uint32_t)tokens.len - split, prefill)) :
          dsm_metal_prefill(prefill_metal, tokens.ids, (uint32_t)tokens.len, prefill))) {
        fprintf(stderr, "Metal prefill failed\n");
        goto done;
    }
    float max_abs = 0;
    uint32_t non_finite = 0;
    double max_ref = -INFINITY, max_pre = -INFINITY, z_ref = 0, z_pre = 0;
    for (uint32_t i = 0; i < model.vocab; ++i) {
        float diff = fabsf(reference[i] - prefill[i]);
        if (diff > max_abs) max_abs = diff;
        non_finite += !isfinite(reference[i]) || !isfinite(prefill[i]);
        if (reference[i] > max_ref) max_ref = reference[i];
        if (prefill[i] > max_pre) max_pre = prefill[i];
    }
    for (uint32_t i = 0; i < model.vocab; ++i) {
        z_ref += exp(reference[i] - max_ref);
        z_pre += exp(prefill[i] - max_pre);
    }
    double kl = 0, max_dprob = 0;
    for (uint32_t i = 0; i < model.vocab; ++i) {
        double p = exp(reference[i] - max_ref) / z_ref;
        double q = exp(prefill[i] - max_pre) / z_pre;
        if (p > 0 && q > 0) kl += p * log(p / q);
        if (fabs(p - q) > max_dprob) max_dprob = fabs(p - q);
    }
    if (non_finite) {
        fprintf(stderr, "non-finite logits: %u\n", non_finite);
        goto done;
    }
    uint32_t reference_argmax = argmax(reference, model.vocab);
    uint32_t prefill_argmax = argmax(prefill, model.vocab);
    printf("tokens=%zu chunk=%s split=%u max_abs=%.8g kl=%.3g max_dprob=%.3g argmax=%u/%u\n",
           tokens.len, getenv("DSM_PREFILL_CHUNK") ? getenv("DSM_PREFILL_CHUNK") : "default",
           split, max_abs, kl, max_dprob, reference_argmax, prefill_argmax);
    int top_k_ok = same_top_k(reference, prefill, model.vocab);
    printf("top%d=%s\n", TOP_K, top_k_ok ? "same" : "different");
    /* MoE top-k routing is discontinuous at near-ties: ~1e-6 summation-order noise
       can swap the 8th/9th expert and move tail logits by ~1 while the output
       distribution is unchanged, so MoE models are gated on KL and max |dprob|. */
    int logits_ok = model.is_moe ? kl < 1e-4 && max_dprob < 1e-3 : max_abs < 5e-2f;
    if (!logits_ok || reference_argmax != prefill_argmax || !top_k_ok) {
        fprintf(stderr, "prefill parity failed\n");
        goto done;
    }
    dsm_sampler sampler;
    dsm_sampler_init(&sampler, 0.0f, 0.95f, 1);
    for (uint32_t i = 0; i < GREEDY_TOKENS; ++i) {
        int32_t reference_token = dsm_sample(&sampler, reference, model.vocab);
        int32_t prefill_token = dsm_sample(&sampler, prefill, model.vocab);
        if (reference_token != prefill_token ||
            !dsm_metal_token(metal, reference_token, reference) ||
            !dsm_metal_token(prefill_metal, prefill_token, prefill)) {
            fprintf(stderr, "greedy continuation mismatch at token %u\n", i);
            goto done;
        }
    }
    if (dump) {
        FILE *file = fopen(dump, "wb");
        if (!file || fwrite(prefill, sizeof(*prefill), model.vocab, file) != model.vocab ||
            fclose(file)) {
            fprintf(stderr, "failed to write %s\n", dump);
            if (file) fclose(file);
            goto done;
        }
    }
    status = 0;
done:
    free(reference);
    free(prefill);
    dsm_metal_free(metal);
    dsm_metal_free(prefill_metal);
    dsm_tokens_free(&tokens);
    dsm_tokenizer_free(tokenizer);
    dsm_model_close(&model);
    return status;
}
