#include "metal.h"
#include "model.h"
#include "quants.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static float random_value(uint32_t *state) { *state = *state * 1664525u + 1013904223u; return ((float)(*state >> 8) / 8388608.0f) - 1.0f; }
static void check(dsm_metal *metal, const gguf_tensor *weights) {
    uint64_t in = weights->dims[0], out = weights->dims[1]; float *x = malloc(in * sizeof(*x)), *cpu = malloc(out * sizeof(*cpu)), *gpu = malloc(out * sizeof(*gpu)); uint32_t state = 1;
    for (uint64_t i = 0; i < in; ++i) x[i] = random_value(&state);
    const ggml_type_info *info = gguf_type_info(weights->type); uint64_t stride = in / info->block_elems * info->block_bytes;
    for (uint64_t row = 0; row < out; ++row) cpu[row] = dsm_dot_row(weights->type, (const uint8_t *)weights->data + row * stride, x, in);
    if (!dsm_metal_matvec_test(metal, weights, x, gpu)) { fprintf(stderr, "Metal GEMV failed\n"); exit(1); }
    float max_abs = 0, max_rel = 0; uint64_t bad = out;
    for (uint64_t row = 0; row < out; ++row) { float abs = fabsf(cpu[row] - gpu[row]), rel = abs / fmaxf(fabsf(cpu[row]), 1e-2f); if (abs > max_abs) max_abs = abs; if (rel > max_rel) max_rel = rel; if (bad == out && rel >= 1e-3f) bad = row; }
    printf("%-28.*s type=%-5s rows=%llu max_abs=%.8g max_rel=%.8g first_bad=%s\n", (int)weights->name.len, weights->name.ptr, info->name, (unsigned long long)out, max_abs, max_rel, bad == out ? "none" : "set");
    if (bad != out) printf("  first_bad_row=%llu cpu=%.8g gpu=%.8g\n", (unsigned long long)bad, cpu[bad], gpu[bad]);
    free(x); free(cpu); free(gpu);
}
int main(int argc, char **argv) {
    if (argc != 2) { fprintf(stderr, "usage: %s MODEL.gguf\n", argv[0]); return 2; }
    dsm_model model; char error[256]; if (!dsm_model_open(&model, argv[1])) { fprintf(stderr, "%s\n", model.error); return 1; }
    dsm_metal *metal; if (!dsm_metal_init(&metal, &model, 1, error, sizeof(error))) { fprintf(stderr, "%s\n", error); return 1; }
    uint32_t seen[64] = {0}; const gguf_tensor *must[] = {model.token_embd, model.output, model.layer[0].down};
    for (unsigned i = 0; i < sizeof(must) / sizeof(*must); ++i) if (must[i] && !seen[must[i]->type]) { check(metal, must[i]); seen[must[i]->type] = 1; }
    for (uint64_t i = 0; i < model.gguf.n_tensors; ++i) { const gguf_tensor *tensor = &model.gguf.tensors[i]; if (tensor->ndim == 2 && dsm_quant_supported(tensor->type) && !seen[tensor->type]) { check(metal, tensor); seen[tensor->type] = 1; } }
    dsm_metal_free(metal); dsm_model_close(&model); return 0;
}
