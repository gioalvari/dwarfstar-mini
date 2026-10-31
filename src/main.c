#include "forward.h"
#include "metal.h"
#include "model.h"
#include "sampler.h"
#include "tokenizer.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static void usage(const char *argv0) { fprintf(stderr,"usage: %s -m MODEL -p PROMPT [-n N] [--backend metal|cpu] [--temp T] [--chat] [--dump-logits FILE] [--tokens-only]\n",argv0); }
static double now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec/1e9;}
int main(int argc, char **argv) {
    const char *path = NULL, *prompt = NULL, *dump = NULL, *backend = NULL;
    int n = 16, chat = 0, tokens_only = 0;
    float temp = 0;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "-m") && i + 1 < argc) path = argv[++i];
        else if (!strcmp(argv[i], "-p") && i + 1 < argc) prompt = argv[++i];
        else if (!strcmp(argv[i], "-n") && i + 1 < argc) n = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--temp") && i + 1 < argc) temp = strtof(argv[++i], NULL);
        else if (!strcmp(argv[i], "--backend") && i + 1 < argc) backend = argv[++i];
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
    int use_metal = backend ? !strcmp(backend, "metal") : dsm_metal_available();
    if (backend && strcmp(backend, "cpu") && strcmp(backend, "metal")) { fprintf(stderr, "dsm: unknown backend '%s'\n", backend); return 2; }
    dsm_forward *f = NULL; dsm_metal *metal = NULL; uint32_t context = (uint32_t)(ids.len + (size_t)n + 1); char metal_error[256];
    if (use_metal) {
        if (!dsm_metal_init(&metal, &model, context, metal_error, sizeof(metal_error))) {
            if (backend) { fprintf(stderr, "dsm: Metal initialization failed: %s\n", metal_error); return 1; }
            fprintf(stderr, "dsm: Metal unavailable (%s); falling back to CPU\n", metal_error); use_metal = 0;
        }
    }
    if (!use_metal && !dsm_forward_init(&f, &model, context)) { fprintf(stderr, "dsm: cannot allocate CPU forward state\n"); return 1; }
    float *logits = malloc((size_t)model.vocab * sizeof(*logits)); if (!logits) return 1;
    double prefill_started = now();
    for (size_t i = 0; i < ids.len; ++i) if (!(use_metal ? dsm_metal_token(metal, ids.ids[i], logits) : dsm_forward_token(f, ids.ids[i], logits))) { fprintf(stderr, "dsm: prompt inference failed\n"); return 1; }
    double prefill_elapsed = now() - prefill_started;
    if (dump) { FILE *fp = fopen(dump, "wb"); if (!fp || fwrite(logits, sizeof(*logits), model.vocab, fp) != model.vocab) { fprintf(stderr, "dsm: failed to write logits\n"); return 1; } fclose(fp); }
    dsm_sampler sampler; dsm_sampler_init(&sampler, temp, 0.95f, 1); double decode_started = now(); int generated = 0;
    for (int i = 0; i < n; ++i) { int32_t id = dsm_sample(&sampler, logits, model.vocab); if (!dsm_token_decode(tok, id, stdout) || !(use_metal ? dsm_metal_token(metal, id, logits) : dsm_forward_token(f, id, logits))) break; ++generated; }
    double decode_elapsed = now() - decode_started;
    fprintf(stderr, "\ndsm: %s prefill %.2f tokens/s (%zu tokens, %.3f s); decode %.2f tokens/s (%d tokens, %.3f s)\n", use_metal ? "metal" : "cpu", ids.len / prefill_elapsed, ids.len, prefill_elapsed, generated / decode_elapsed, generated, decode_elapsed);
    free(logits); dsm_forward_free(f); dsm_metal_free(metal); dsm_tokens_free(&ids); dsm_tokenizer_free(tok); dsm_model_close(&model); free(rendered); return 0;
}
