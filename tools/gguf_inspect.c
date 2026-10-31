#include "gguf.h"

#include <stdio.h>
#include <string.h>

int main(int argc, char **argv) {
    gguf_file file;
    if (argc != 2 || !gguf_open(&file, argv[1])) return 1;
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
