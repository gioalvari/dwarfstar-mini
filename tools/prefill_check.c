#include "metal.h"
#include "model.h"
#include "sampler.h"
#include "tokenizer.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void usage(const char *argv0) {
    fprintf(stderr, "usage: %s MODEL.gguf PROMPT [--dump OUT.bin]\n", argv0);
}

static uint32_t argmax(const float *values, uint32_t n) {
    uint32_t best = 0;
    for (uint32_t i = 1; i < n; ++i) if (values[i] > values[best]) best = i;
    return best;
}

int main(int argc, char **argv) {
    const char *model_path;
    const char *prompt;
    const char *dump = NULL;
    dsm_model model;
    dsm_tokenizer *tokenizer = NULL;
    dsm_tokens tokens = {0};
    dsm_metal *metal = NULL;
    dsm_metal *prefill_metal = NULL;
    float *reference = NULL;
    float *prefill = NULL;
    char error[256];
    int status = 1;

    if (argc != 3 && argc != 5) {
        usage(argv[0]);
        return 2;
    }
    model_path = argv[1];
    prompt = argv[2];
    if (argc == 5) {
        if (strcmp(argv[3], "--dump")) {
            usage(argv[0]);
            return 2;
        }
        dump = argv[4];
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
    if (!dsm_metal_init(&metal, &model, (uint32_t)tokens.len + 32, error,
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
    if (!dsm_metal_init(&prefill_metal, &model, (uint32_t)tokens.len + 32,
                        error, sizeof(error)) ||
        !dsm_metal_prefill(prefill_metal, tokens.ids, (uint32_t)tokens.len,
                           prefill)) {
        fprintf(stderr, "Metal prefill failed\n");
        goto done;
    }
    float max_abs = 0;
    for (uint32_t i = 0; i < model.vocab; ++i) {
        float diff = fabsf(reference[i] - prefill[i]);
        if (diff > max_abs) max_abs = diff;
    }
    uint32_t reference_argmax = argmax(reference, model.vocab);
    uint32_t prefill_argmax = argmax(prefill, model.vocab);
    printf("tokens=%zu max_abs=%.8g argmax=%u/%u\n", tokens.len, max_abs,
           reference_argmax, prefill_argmax);
    if (!(max_abs < 1e-2f) || reference_argmax != prefill_argmax) {
        fprintf(stderr, "prefill parity failed\n");
        goto done;
    }
    dsm_sampler sampler;
    dsm_sampler_init(&sampler, 0.0f, 0.95f, 1);
    for (uint32_t i = 0; i < 32; ++i) {
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
