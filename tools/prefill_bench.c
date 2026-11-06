#include "metal.h"
#include "model.h"
#include "tokenizer.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static double now(void) {
    struct timespec time;
    clock_gettime(CLOCK_MONOTONIC, &time);
    return time.tv_sec + time.tv_nsec * 1e-9;
}

int main(int argc, char **argv) {
    if (argc < 3 || argc > 4) {
        fprintf(stderr, "usage: %s MODEL PROMPT [REPETITIONS]\n", argv[0]);
        return 2;
    }
    dsm_model model;
    dsm_tokenizer *tokenizer = NULL;
    dsm_tokens tokens = {0};
    char error[256];
    int repetitions = argc == 4 ? atoi(argv[3]) : 3;
    if (repetitions < 1 || !dsm_model_open(&model, argv[1]) ||
        !dsm_tokenizer_init(&tokenizer, &model.gguf, error, sizeof(error)) ||
        !dsm_tokenize(tokenizer, argv[2], 0, &tokens)) return 1;
    float *logits = malloc((size_t)model.vocab * sizeof(*logits));
    if (!logits) return 1;
    for (int rep = 0; rep < repetitions; ++rep) {
        dsm_metal *metal = NULL;
        if (!dsm_metal_init(&metal, &model, (uint32_t)tokens.len, error,
                            sizeof(error))) return 1;
        double start = now();
        if (!dsm_metal_prefill(metal, tokens.ids, (uint32_t)tokens.len, logits)) return 1;
        printf("rep%d prefill %.2f tok/s\n", rep,
               tokens.len / (now() - start));
        dsm_metal_free(metal);
    }
    free(logits);
    dsm_tokens_free(&tokens);
    dsm_tokenizer_free(tokenizer);
    dsm_model_close(&model);
    return 0;
}
