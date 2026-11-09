#include "metal.h"
#include "model.h"
#include "sampler.h"
#include "tokenizer.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { GREEDY_TOKENS = 32, TOP_K = 5 };

static uint32_t argmax(const float *values, uint32_t n) {
    uint32_t best = 0;
    for (uint32_t i = 1; i < n; ++i) if (values[i] > values[best]) best = i;
    return best;
}
static void top_k(const float *values, uint32_t n, uint32_t *out) {
    for (uint32_t k = 0; k < TOP_K; ++k) out[k] = UINT32_MAX;
    for (uint32_t i = 0; i < n; ++i) {
        uint32_t k = TOP_K;
        while (k && (out[k - 1] == UINT32_MAX || values[i] > values[out[k - 1]])) --k;
        if (k == TOP_K) continue;
        memmove(out + k + 1, out + k, (TOP_K - k - 1) * sizeof(*out));
        out[k] = i;
    }
}
static int same_top_k(const float *a, const float *b, uint32_t n) {
    uint32_t ta[TOP_K], tb[TOP_K];
    top_k(a, n, ta); top_k(b, n, tb);
    return !memcmp(ta, tb, sizeof(ta));
}
static double kl(const float *a, const float *b, uint32_t n) {
    double ma = -INFINITY, mb = -INFINITY, za = 0, zb = 0, out = 0;
    for (uint32_t i = 0; i < n; ++i) { if (a[i] > ma) ma = a[i]; if (b[i] > mb) mb = b[i]; }
    for (uint32_t i = 0; i < n; ++i) { za += exp(a[i] - ma); zb += exp(b[i] - mb); }
    for (uint32_t i = 0; i < n; ++i) { double p = exp(a[i] - ma) / za, q = exp(b[i] - mb) / zb; if (p && q) out += p * log(p / q); }
    return out;
}
static int prefill(dsm_metal *metal, const int32_t *tokens, uint32_t n, float *logits) {
    return n ? dsm_metal_prefill(metal, tokens, n, logits) : 1;
}
static int greedy(dsm_metal *metal, float *logits, uint32_t vocab, int32_t *out) {
    dsm_sampler sampler;
    dsm_sampler_init(&sampler, 0, 0.95f, 1);
    for (uint32_t i = 0; i < GREEDY_TOKENS; ++i) {
        out[i] = dsm_sample(&sampler, logits, vocab);
        if (!dsm_metal_token(metal, out[i], logits)) return 0;
    }
    return 1;
}
static int fresh(const dsm_model *model, const dsm_tokens *tokens, float *logits,
                 int32_t *continuation, char *error, size_t error_size) {
    dsm_metal *metal = NULL;
    int ok = dsm_metal_init(&metal, model, (uint32_t)tokens->len + GREEDY_TOKENS,
                            error, (uint32_t)error_size) &&
             prefill(metal, tokens->ids, (uint32_t)tokens->len, logits);
    float *greedy_logits = ok ? malloc((size_t)model->vocab * sizeof(*greedy_logits)) : NULL;
    if (!greedy_logits) ok = 0;
    if (ok) {
        memcpy(greedy_logits, logits, (size_t)model->vocab * sizeof(*logits));
        ok = greedy(metal, greedy_logits, model->vocab, continuation);
    }
    free(greedy_logits);
    dsm_metal_free(metal);
    return ok;
}
static int verify(const dsm_model *model, dsm_metal *metal, const dsm_tokens *tokens,
                  uint32_t request, const float *reference, const int32_t *ref_greedy,
                  float *logits) {
    int32_t continuation[GREEDY_TOKENS];
    uint32_t restored = dsm_metal_rewind(metal, request);
    if (restored == UINT32_MAX || !prefill(metal, tokens->ids + restored,
                                           (uint32_t)tokens->len - restored, logits)) return 0;
    double value = kl(reference, logits, model->vocab);
    int logits_ok = value < 1e-6 && argmax(reference, model->vocab) == argmax(logits, model->vocab) &&
                    same_top_k(reference, logits, model->vocab);
    float *greedy_logits = logits_ok ? malloc((size_t)model->vocab * sizeof(*greedy_logits)) : NULL;
    if (greedy_logits) memcpy(greedy_logits, logits, (size_t)model->vocab * sizeof(*logits));
    int greedy_ok = greedy_logits && greedy(metal, greedy_logits, model->vocab, continuation) &&
                    !memcmp(ref_greedy, continuation, sizeof(continuation));
    free(greedy_logits);
    printf("request=%u restored=%u kl=%.3g argmax/top5=%s greedy32=%s\n", request, restored,
           value, logits_ok ? "same" : "different", greedy_ok ? "same" : "different");
    return greedy_ok;
}
int main(int argc, char **argv) {
    if (argc != 3) { fprintf(stderr, "usage: %s MODEL.gguf PROMPT\n", argv[0]); return 2; }
    dsm_model model; dsm_tokenizer *tokenizer = NULL; dsm_tokens tokens = {0};
    dsm_metal *metal = NULL; float *reference = NULL, *logits = NULL;
    int32_t reference_greedy[GREEDY_TOKENS]; char error[256] = {0}; int status = 1;
    if (!dsm_model_open(&model, argv[1]) || !dsm_tokenizer_init(&tokenizer, &model.gguf, error, sizeof(error)) ||
        !dsm_tokenize(tokenizer, argv[2], 0, &tokens) || tokens.len < 12) goto done;
    reference = malloc((size_t)model.vocab * sizeof(*reference));
    logits = malloc((size_t)model.vocab * sizeof(*logits));
    if (!reference || !logits || !fresh(&model, &tokens, reference, reference_greedy, error, sizeof(error)) ||
        !dsm_metal_init(&metal, &model, (uint32_t)tokens.len + GREEDY_TOKENS, error, sizeof(error))) goto done;
    uint32_t checkpoint = (uint32_t)(tokens.len / 2);
    uint32_t context = (uint32_t)tokens.len + GREEDY_TOKENS;
    if (!prefill(metal, tokens.ids, checkpoint, logits) || !dsm_metal_checkpoint(metal) ||
        !prefill(metal, tokens.ids + checkpoint, (uint32_t)tokens.len - checkpoint, logits)) goto done;
    uint64_t bytes = (uint64_t)model.layers *
        ((uint64_t)model.ssm_vheads * model.ssm_state * model.ssm_state +
         (uint64_t)(model.ssm_inner + 2 * model.ssm_groups * model.ssm_state) * (model.ssm_conv - 1)) * sizeof(float);
    printf("tokens=%zu checkpoint=%u state=%.2f MiB each\n", tokens.len, checkpoint, bytes / (1024.0 * 1024.0));
    uint32_t requests[] = {0, checkpoint, checkpoint + 5};
    for (size_t i = 0; i < sizeof(requests) / sizeof(*requests); ++i) {
        if (!verify(&model, metal, &tokens, requests[i], reference, reference_greedy, logits)) goto done;
        if (i + 1 < sizeof(requests) / sizeof(*requests) &&
            (dsm_metal_rewind(metal, 0) == UINT32_MAX || !prefill(metal, tokens.ids, checkpoint, logits) ||
             !dsm_metal_checkpoint(metal) || !prefill(metal, tokens.ids + checkpoint,
             (uint32_t)tokens.len - checkpoint, logits))) goto done;
    }
    /* A checkpoint from the original suffix must vanish after branching low. */
    dsm_metal_free(metal); metal = NULL;
    if (!dsm_metal_init(&metal, &model, context, error, sizeof(error)) ||
        dsm_metal_rewind(metal, 0) == UINT32_MAX ||
        !prefill(metal, tokens.ids, checkpoint, logits) || !dsm_metal_checkpoint(metal) ||
        !prefill(metal, tokens.ids + checkpoint, (uint32_t)tokens.len - checkpoint, logits) ||
        !dsm_metal_checkpoint(metal) || dsm_metal_rewind(metal, checkpoint) != checkpoint) {
        goto done;
    }
    int32_t changed = tokens.ids[checkpoint] == 0 ? 1 : 0;
    if (!dsm_metal_token(metal, changed, logits) ||
        !prefill(metal, tokens.ids + checkpoint + 1, (uint32_t)tokens.len - checkpoint - 1, logits) ||
        !dsm_metal_checkpoint(metal)) goto done;
    uint32_t stale_restored = dsm_metal_rewind(metal, (uint32_t)tokens.len);
    printf("stale request=%zu restored=%u\n", tokens.len, stale_restored);
    if (stale_restored != tokens.len) {
        fprintf(stderr, "stale checkpoint was restored\n"); goto done;
    }
    puts("stale=invalidated");
    status = 0;
done:
    if (status) fprintf(stderr, "rewind check failed: %s\n", error[0] ? error : "inference failure");
    dsm_metal_free(metal); free(reference); free(logits); dsm_tokens_free(&tokens);
    dsm_tokenizer_free(tokenizer); dsm_model_close(&model); return status;
}
