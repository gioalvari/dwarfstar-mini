#include "metal.h"
#include "model.h"
#include "tokenizer.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void usage(const char *argv0) {
    fprintf(stderr, "usage: %s MODEL.gguf PROMPT [--dump OUT.bin]\n", argv0);
}

int main(int argc, char **argv) {
    const char *model_path;
    const char *prompt;
    const char *dump = NULL;
    dsm_model model;
    dsm_tokenizer *tokenizer = NULL;
    dsm_tokens tokens = {0};
    dsm_metal *metal = NULL;
    float *logits = NULL;
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
    if (!dsm_metal_init(&metal, &model, (uint32_t)tokens.len, error,
                        sizeof(error))) {
        fprintf(stderr, "%s\n", error);
        goto done;
    }
    logits = malloc((size_t)model.vocab * sizeof(*logits));
    if (!logits) {
        fprintf(stderr, "out of memory\n");
        goto done;
    }
    for (size_t i = 0; i < tokens.len; ++i) {
        if (!dsm_metal_token(metal, tokens.ids[i], logits)) {
            fprintf(stderr, "Metal token inference failed at %zu\n", i);
            goto done;
        }
    }
    if (dump) {
        FILE *file = fopen(dump, "wb");
        if (!file || fwrite(logits, sizeof(*logits), model.vocab, file) != model.vocab ||
            fclose(file)) {
            fprintf(stderr, "failed to write %s\n", dump);
            if (file) fclose(file);
            goto done;
        }
    }
    status = 0;
done:
    free(logits);
    dsm_metal_free(metal);
    dsm_tokens_free(&tokens);
    dsm_tokenizer_free(tokenizer);
    dsm_model_close(&model);
    return status;
}
