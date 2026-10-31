#include "gguf.h"
#include "quants.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef void (*ggml_to_float_t)(const void *input, float *output, int64_t count);
typedef struct { const char *name; int64_t block_size, block_interleave; size_t type_size; _Bool quantized; ggml_to_float_t to_float; void *from_float; } ggml_traits;
extern const ggml_traits *ggml_get_type_traits(int type);

static void print_name(gguf_str text) { printf("%.*s", (int)text.len, text.ptr); }
int main(int argc, char **argv) {
    if (argc != 2) { fprintf(stderr, "usage: %s MODEL.gguf\n", argv[0]); return 2; }
    gguf_file file;
    if (!gguf_open(&file, argv[1])) { fprintf(stderr, "%s\n", file.error); return 1; }
    uint64_t histogram[64] = {0};
    for (uint64_t i = 0; i < file.n_tensors; ++i) if (file.tensors[i].type < 64) ++histogram[file.tensors[i].type];
    puts("type histogram:");
    for (unsigned type = 0; type < 64; ++type) if (histogram[type]) { const ggml_type_info *info = gguf_type_info(type); printf("  %u %-8s %llu\n", type, info ? info->name : "unknown", (unsigned long long)histogram[type]); }
    puts("dequantization differences (first two rows):");
    for (uint64_t i = 0; i < file.n_tensors; ++i) {
        const gguf_tensor *tensor = &file.tensors[i]; const ggml_type_info *info = gguf_type_info(tensor->type);
        if (!dsm_quant_supported(tensor->type) || !info || tensor->ndim < 1) continue;
        uint64_t row_count = tensor->ndim < 2 ? 1 : tensor->dims[1]; uint64_t rows = row_count < 2 ? row_count : 2, width = tensor->dims[0];
        size_t row_bytes = (size_t)(width / info->block_elems * info->block_bytes); float *reference = malloc((size_t)width * sizeof(*reference)), *actual = malloc((size_t)width * sizeof(*actual));
        if (!reference || !actual) return 1; float max_diff = 0;
        const ggml_traits *traits = ggml_get_type_traits((int)tensor->type);
        for (uint64_t row = 0; row < rows; ++row) { if (tensor->type == GGML_TYPE_F32) memcpy(reference, (const uint8_t *)tensor->data + row * row_bytes, (size_t)width * sizeof(*reference)); else traits->to_float((const uint8_t *)tensor->data + row * row_bytes, reference, (int64_t)width); dsm_dequant_row(tensor->type, (const uint8_t *)tensor->data + row * row_bytes, actual, width); for (uint64_t col = 0; col < width; ++col) { float diff = fabsf(reference[col] - actual[col]); if (diff > max_diff) max_diff = diff; } }
        printf("  "); print_name(tensor->name); printf(" type=%s max_abs_diff=%.9g\n", info->name, max_diff); free(reference); free(actual);
    }
    gguf_close(&file); return 0;
}
