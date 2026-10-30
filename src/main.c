#include "forward.h"
#include "model.h"
#include "sampler.h"
#include "tokenizer.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static void usage(const char *argv0) { fprintf(stderr,"usage: %s -m MODEL -p PROMPT [-n N] [--temp T] [--chat] [--dump-logits FILE] [--tokens-only]\n",argv0); }
static double now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec/1e9;}
int main(int argc, char **argv) {
    const char *path = NULL, *prompt = NULL, *dump = NULL;
    int n = 16, chat = 0, tokens_only = 0;
    float temp = 0;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "-m") && i + 1 < argc) path = argv[++i];
        else if (!strcmp(argv[i], "-p") && i + 1 < argc) prompt = argv[++i];
        else if (!strcmp(argv[i], "-n") && i + 1 < argc) n = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--temp") && i + 1 < argc) temp = strtof(argv[++i], NULL);
        else if (!strcmp(argv[i], "--chat")) chat = 1;
        else if (!strcmp(argv[i], "--dump-logits") && i + 1 < argc) dump = argv[++i];
        else if (!strcmp(argv[i], "--tokens-only")) tokens_only = 1;
        else { usage(argv[0]); return 2; }
    }
    if (!path || !prompt || n < 0) { usage(argv[0]); return 2; }
    char *rendered = NULL;
    if (chat) { size_t len = strlen(prompt) + 64; rendered = malloc(len); if (!rendered) return 1; snprintf(rendered, len, "<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n", prompt); prompt = rendered; }
    dsm_model model;
    if (!dsm_model_open(&model, path)) { fprintf(stderr, "dsm: %s\n", model.error); free(rendered); return 1; }
    dsm_tokenizer *tok; char err[256];
    if (!dsm_tokenizer_init(&tok, &model.gguf, err, sizeof(err))) { fprintf(stderr, "dsm: %s\n", err); dsm_model_close(&model); free(rendered); return 1; }
    dsm_tokens ids;
    if (!dsm_tokenize(tok, prompt, 0, &ids)) { fprintf(stderr, "dsm: tokenization failed\n"); dsm_tokenizer_free(tok); dsm_model_close(&model); free(rendered); return 1; }
    if (tokens_only) { for (size_t i = 0; i < ids.len; ++i) printf("%s%d", i ? " " : "", ids.ids[i]); puts(""); dsm_tokens_free(&ids); dsm_tokenizer_free(tok); dsm_model_close(&model); free(rendered); return 0; }
    dsm_forward *f; uint32_t context = (uint32_t)(ids.len + (size_t)n + 1);
    if (!dsm_forward_init(&f, &model, context)) { fprintf(stderr, "dsm: cannot allocate forward state\n"); return 1; }
    float *logits = malloc((size_t)model.vocab * sizeof(*logits)); if (!logits) return 1;
    double started = now();
    for (size_t i = 0; i < ids.len; ++i) if (!dsm_forward_token(f, ids.ids[i], logits)) { fprintf(stderr, "dsm: prompt inference failed\n"); return 1; }
    if (dump) { FILE *fp = fopen(dump, "wb"); if (!fp || fwrite(logits, sizeof(*logits), model.vocab, fp) != model.vocab) { fprintf(stderr, "dsm: failed to write logits\n"); return 1; } fclose(fp); }
    dsm_sampler sampler; dsm_sampler_init(&sampler, temp, 0.95f, 1);
    for (int i = 0; i < n; ++i) { int32_t id = dsm_sample(&sampler, logits, model.vocab); if (!dsm_token_decode(tok, id, stdout) || !dsm_forward_token(f, id, logits)) break; }
    double elapsed = now() - started; fprintf(stderr, "\ndsm: %.2f tokens/s (%u tokens, %.3f s)\n", (ids.len + n) / elapsed, (unsigned)(ids.len + n), elapsed);
    free(logits); dsm_forward_free(f); dsm_tokens_free(&ids); dsm_tokenizer_free(tok); dsm_model_close(&model); free(rendered); return 0;
}
