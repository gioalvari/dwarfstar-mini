#include "metal.h"
#include "model.h"
#include "quants.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static float random_value(uint32_t *state) { *state = *state * 1664525u + 1013904223u; return ((float)(*state >> 8) / 8388608.0f) - 1.0f; }
static int matmul_supported(uint32_t type) {
    return type == GGML_TYPE_Q8_0 || type == GGML_TYPE_Q2_K ||
           type == GGML_TYPE_Q3_K || type == GGML_TYPE_Q4_K ||
           type == GGML_TYPE_Q6_K;
}
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
static void check_matmul(dsm_metal *metal, const gguf_tensor *weights) {
    enum { rows = 37 };
    uint64_t in = weights->dims[0], out = weights->dims[1];
    float *x = malloc((size_t)rows * in * sizeof(*x));
    float *gemv = malloc((size_t)rows * out * sizeof(*gemv));
    float *mm = malloc((size_t)rows * out * sizeof(*mm));
    uint32_t state = 1;
    if (!x || !gemv || !mm) { fprintf(stderr, "out of memory\n"); exit(1); }
    for (uint64_t i = 0; i < (uint64_t)rows * in; ++i) x[i] = random_value(&state);
    for (uint32_t row = 0; row < rows; ++row) {
        if (!dsm_metal_matvec_test(metal, weights, x + (uint64_t)row * in,
                                   gemv + (uint64_t)row * out)) {
            fprintf(stderr, "Metal GEMV failed\n"); exit(1);
        }
    }
    if (!dsm_metal_matmul_test(metal, weights, x, rows, mm)) {
        fprintf(stderr, "Metal mul_mm failed\n"); exit(1);
    }
    float max_abs = 0, max_rel = 0;
    double error_sq = 0, reference_sq = 0;
    for (uint64_t i = 0; i < (uint64_t)rows * out; ++i) {
        float abs = fabsf(gemv[i] - mm[i]);
        float rel = abs / fmaxf(fabsf(gemv[i]), 1e-2f);
        if (abs > max_abs) max_abs = abs;
        if (rel > max_rel) max_rel = rel;
        error_sq += (double)abs * abs;
        reference_sq += (double)gemv[i] * gemv[i];
    }
    float rel_err = sqrtf((float)(error_sq / reference_sq));
    const ggml_type_info *info = gguf_type_info(weights->type);
    printf("mul_mm %-21.*s type=%-5s rows=%u max_abs=%.8g max_rel=%.8g rel_err=%.8g\n",
           (int)weights->name.len, weights->name.ptr, info->name, rows, max_abs,
           max_rel, rel_err);
    if (!(rel_err < 2e-3f)) {
        fprintf(stderr, "mul_mm relative error exceeded 2e-3\n");
        exit(1);
    }
    free(x); free(gemv); free(mm);
}
static void check_expert(dsm_metal *metal, const gguf_tensor *weights, uint32_t expert) {
    if (weights->ndim != 3 || expert >= weights->dims[2]) { fprintf(stderr, "invalid expert tensor or id\n"); exit(2); }
    uint64_t in = weights->dims[0], out = weights->dims[1]; float *x = malloc(in * sizeof(*x)), *cpu = malloc(out * sizeof(*cpu)), *gpu = malloc(out * sizeof(*gpu)); uint32_t state = 1;
    for (uint64_t i = 0; i < in; ++i) x[i] = random_value(&state);
    const ggml_type_info *info = gguf_type_info(weights->type); uint64_t expert_stride = weights->bytes / weights->dims[2], row_stride = in / info->block_elems * info->block_bytes;
    const uint8_t *base = (const uint8_t *)weights->data + expert * expert_stride;
    for (uint64_t row = 0; row < out; ++row) cpu[row] = dsm_dot_row(weights->type, base + row * row_stride, x, in);
    if (!dsm_metal_expert_matvec_test(metal, weights, expert, x, gpu)) { fprintf(stderr, "Metal expert GEMV failed\n"); exit(1); }
    float max_abs = 0, max_rel = 0; uint64_t bad = out;
    for (uint64_t row = 0; row < out; ++row) { float abs = fabsf(cpu[row] - gpu[row]), rel = abs / fmaxf(fabsf(cpu[row]), 1e-2f); if (abs > max_abs) max_abs = abs; if (rel > max_rel) max_rel = rel; if (bad == out && rel >= 1e-3f) bad = row; }
    printf("%-28.*s expert=%u type=%-5s rows=%llu max_abs=%.8g max_rel=%.8g first_bad=%s\n", (int)weights->name.len, weights->name.ptr, expert, info->name, (unsigned long long)out, max_abs, max_rel, bad == out ? "none" : "set");
    if (bad != out) printf("  first_bad_row=%llu cpu=%.8g gpu=%.8g\n", (unsigned long long)bad, cpu[bad], gpu[bad]);
    free(x); free(cpu); free(gpu);
}
static void check_expert_matmul(dsm_metal *metal, const gguf_tensor *weights) {
    enum { rows = 37, slots = 8 };
    uint64_t in = weights->dims[0], out = weights->dims[1], experts = weights->dims[2];
    float *x = malloc((size_t)rows * in * sizeof(*x));
    float *cpu = calloc((size_t)rows * slots * out, sizeof(*cpu));
    float *gpu = calloc((size_t)rows * slots * out, sizeof(*gpu));
    uint32_t *ids = malloc((size_t)rows * slots * sizeof(*ids));
    uint32_t state = 1;
    if (!x || !cpu || !gpu || !ids) { fprintf(stderr, "out of memory\n"); exit(1); }
    for (uint64_t i = 0; i < (uint64_t)rows * in; ++i) x[i] = random_value(&state);
    for (uint32_t token = 0; token < rows; ++token) {
        for (uint32_t slot = 0; slot < slots; ++slot) {
            ids[token * slots + slot] = slot < 4 ? 0 : (token + 3 * slot) % experts;
        }
    }
    const ggml_type_info *info = gguf_type_info(weights->type);
    uint64_t expert_stride = weights->bytes / experts;
    uint64_t row_stride = in / info->block_elems * info->block_bytes;
    for (uint32_t token = 0; token < rows; ++token) {
        for (uint32_t slot = 0; slot < slots; ++slot) {
            const uint8_t *base = (const uint8_t *)weights->data +
                                  (uint64_t)ids[token * slots + slot] * expert_stride;
            float *dst = cpu + ((uint64_t)token * slots + slot) * out;
            for (uint64_t row = 0; row < out; ++row) {
                dst[row] = dsm_dot_row(weights->type, base + row * row_stride,
                                       x + (uint64_t)token * in, in);
            }
        }
    }
    if (!dsm_metal_expert_matmul_test(metal, weights, x, ids, rows, 1, gpu)) {
        fprintf(stderr, "Metal expert mul_mm_id failed\n"); exit(1);
    }
    float max_abs = 0, max_rel = 0;
    double error_sq = 0, reference_sq = 0;
    for (uint64_t i = 0; i < (uint64_t)rows * slots * out; ++i) {
        float abs = fabsf(cpu[i] - gpu[i]);
        float rel = abs / fmaxf(fabsf(cpu[i]), 1e-2f);
        if (abs > max_abs) max_abs = abs;
        if (rel > max_rel) max_rel = rel;
        error_sq += (double)abs * abs;
        reference_sq += (double)cpu[i] * cpu[i];
    }
    float rel_err = sqrtf((float)(error_sq / reference_sq));
    printf("mul_mm_id %-18.*s type=%-5s rows=%u max_abs=%.8g max_rel=%.8g rel_err=%.8g\n",
           (int)weights->name.len, weights->name.ptr, info->name, rows, max_abs,
           max_rel, rel_err);
    if (!(rel_err < 2e-3f)) {
        fprintf(stderr, "mul_mm_id relative error exceeded 2e-3\n"); exit(1);
    }
    free(x); free(cpu); free(gpu); free(ids);
}
int main(int argc, char **argv) {
    if (argc != 2 && argc != 4) { fprintf(stderr, "usage: %s MODEL.gguf [TENSOR EXPERT_ID]\n", argv[0]); return 2; }
    dsm_model model; char error[256]; if (!dsm_model_open(&model, argv[1])) { fprintf(stderr, "%s\n", model.error); return 1; }
    dsm_metal *metal; if (!dsm_metal_init(&metal, &model, 1, error, sizeof(error))) { fprintf(stderr, "%s\n", error); return 1; }
    if (argc == 4) {
        char *end = NULL; unsigned long id = strtoul(argv[3], &end, 10); const gguf_tensor *tensor = gguf_tensor_find(&model.gguf, argv[2]);
        if (!tensor || !*argv[3] || *end || id > UINT32_MAX) { fprintf(stderr, "invalid tensor or expert id\n"); dsm_metal_free(metal); dsm_model_close(&model); return 2; }
        check_expert(metal, tensor, (uint32_t)id);
    } else {
        uint32_t seen[64] = {0}; const gguf_tensor *must[] = {model.token_embd, model.output, model.layer[0].down};
        for (unsigned i = 0; i < sizeof(must) / sizeof(*must); ++i) if (must[i] && !seen[must[i]->type]) { check(metal, must[i]); seen[must[i]->type] = 1; }
        for (uint64_t i = 0; i < model.gguf.n_tensors; ++i) { const gguf_tensor *tensor = &model.gguf.tensors[i]; if (tensor->ndim == 2 && dsm_quant_supported(tensor->type) && !seen[tensor->type]) { check(metal, tensor); seen[tensor->type] = 1; } }
        memset(seen, 0, sizeof(seen));
        for (uint64_t i = 0; i < model.gguf.n_tensors; ++i) {
            const gguf_tensor *tensor = &model.gguf.tensors[i];
            if (tensor->ndim == 2 && tensor->dims[0] % 32 == 0 &&
                matmul_supported(tensor->type) && !seen[tensor->type]) {
                check_matmul(metal, tensor); seen[tensor->type] = 1;
            }
        }
        memset(seen, 0, sizeof(seen));
        for (uint64_t i = 0; i < model.gguf.n_tensors; ++i) {
            const gguf_tensor *tensor = &model.gguf.tensors[i];
            if (tensor->ndim == 3 && tensor->dims[0] % 32 == 0 &&
                matmul_supported(tensor->type) && !seen[tensor->type]) {
                check_expert_matmul(metal, tensor); seen[tensor->type] = 1;
            }
        }
    }
    dsm_metal_free(metal); dsm_model_close(&model); return 0;
}
