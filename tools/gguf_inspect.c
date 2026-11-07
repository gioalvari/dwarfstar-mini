#include "model.h"

#include <stdio.h>
#include <string.h>

int main(int argc, char **argv) {
    if (argc == 3 && !strcmp(argv[2], "--layers")) {
        dsm_model model;
        if (!dsm_model_open(&model, argv[1])) {
            fprintf(stderr, "gguf-inspect: %s\n", model.error);
            return 1;
        }
        if (model.arch != DSM_ARCH_QWEN35 && model.arch != DSM_ARCH_QWEN35MOE) {
            fprintf(stderr, "gguf-inspect: --layers is only available for qwen35 models\n");
            dsm_model_close(&model);
            return 1;
        }
        uint64_t mapped, unknown;
        dsm_model_tensor_summary(&model, &mapped, &unknown);
        printf("L ");
        for (uint32_t begin = 0; begin < model.layers;) {
            uint32_t end = begin + 1;
            while (end < model.layers &&
                   model.is_recurrent[end] == model.is_recurrent[begin]) ++end;
            if (end - begin > 1) printf("%u-%u", begin, end - 1);
            else printf("%u", begin);
            printf(" %s%s", model.is_recurrent[begin] ? "rec" : "attn",
                   end == model.layers ? "\n" : ", ");
            begin = end;
        }
        uint32_t recurrent = model.layers - model.n_attn_layers;
        printf("mapped tensors: %llu; unknown: %llu; DeltaNet: %u; attention: %u\n",
               (unsigned long long)mapped, (unsigned long long)unknown, recurrent,
               model.n_attn_layers);
        dsm_model_close(&model);
        return 0;
    }
    gguf_file file;
    if ((argc != 2 && argc != 3) || !gguf_open(&file, argv[1])) return 1;
    if (argc == 3) {
        gguf_str value;
        int ok = !strcmp(argv[2], "--chat-template") &&
                 gguf_get_string(&file, "tokenizer.chat_template", &value);
        if (ok) printf("%.*s\n", (int)value.len, value.ptr);
        gguf_close(&file);
        return ok ? 0 : 1;
    }
    uint64_t count[64] = {0}, bytes[64] = {0};
    for (uint64_t i = 0; i < file.n_kv; ++i) {
        gguf_kv *kv = &file.kv[i];
        if (kv->key.len >= 8 && !memcmp(kv->key.ptr, "qwen3moe", 8)) {
            printf("%.*s type=%u\n", (int)kv->key.len, kv->key.ptr, kv->type);
        }
    }
    for (uint64_t i = 0; i < file.n_tensors; ++i) {
        gguf_tensor *t = &file.tensors[i];
        if (t->type < 64) { ++count[t->type]; bytes[t->type] += t->bytes; }
        if (t->name.len >= 5 && !memcmp(t->name.ptr, "blk.0", 5)) {
            printf("%.*s type=%u dims", (int)t->name.len, t->name.ptr, t->type);
            for (uint32_t d = 0; d < t->ndim; ++d) printf(" %llu", (unsigned long long)t->dims[d]);
            puts("");
        }
    }
    puts("type histogram:");
    for (uint32_t type = 0; type < 64; ++type)
        if (count[type]) printf("type=%u tensors=%llu bytes=%llu\n", type,
                                (unsigned long long)count[type],
                                (unsigned long long)bytes[type]);
    gguf_close(&file);
    return 0;
}
