#include "metal.h"
#include "model.h"

#include <stdio.h>
#include <string.h>

static const gguf_tensor *find_tensor(const dsm_model *model, const char *name) {
    size_t length = strlen(name);
    for (uint64_t i = 0; i < model->gguf.n_tensors; ++i) {
        const gguf_tensor *tensor = &model->gguf.tensors[i];
        if (tensor->name.len == length && !memcmp(tensor->name.ptr, name, length))
            return tensor;
    }
    return NULL;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s MODEL.gguf [TENSOR ...]\n", argv[0]);
        return 2;
    }
    dsm_model model;
    char error[256];
    if (!dsm_model_open(&model, argv[1])) return 1;
    dsm_metal *metal = NULL;
    if (!dsm_metal_init(&metal, &model, 64, error, sizeof(error))) {
        fprintf(stderr, "gemv_bench: %s\n", error);
        dsm_model_close(&model);
        return 1;
    }
    const char *defaults[] = {"blk.0.attn_q.weight", "output"};
    const char *const *names = argc > 2 ? (const char *const *)(argv + 2) : defaults;
    size_t count = argc > 2 ? (size_t)(argc - 2) : sizeof(defaults) / sizeof(*defaults);
    for (size_t i = 0; i < count; ++i) {
        const gguf_tensor *tensor = !strcmp(names[i], "output") ? model.output
                                                                 : find_tensor(&model, names[i]);
        if (!tensor) continue;
        int reps = (int)(4e9 / tensor->bytes);
        if (reps < 20) reps = 20;
        if (reps > 2000) reps = 2000;
        dsm_metal_gemv_bench(metal, tensor, 10);
        double seconds = dsm_metal_gemv_bench(metal, tensor, reps);
        if (seconds <= 0.0) {
            dsm_metal_free(metal);
            dsm_model_close(&model);
            return 1;
        }
        uint64_t rows = tensor->ndim > 1 ? tensor->dims[1] : 1;
        printf("%-22s type=%-2u %5llux%-6llu %7.2f MB %7.1f us %6.1f GB/s\n",
               names[i], tensor->type, (unsigned long long)tensor->dims[0],
               (unsigned long long)rows, tensor->bytes / 1e6,
               seconds * 1e6 / reps, tensor->bytes * reps / seconds / 1e9);
    }
    dsm_metal_free(metal);
    dsm_model_close(&model);
    return 0;
}
