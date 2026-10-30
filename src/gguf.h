#ifndef DSM_GGUF_H
#define DSM_GGUF_H

#include <stddef.h>
#include <stdint.h>

enum gguf_value_type {
    GGUF_UINT8, GGUF_INT8, GGUF_UINT16, GGUF_INT16, GGUF_UINT32, GGUF_INT32,
    GGUF_FLOAT32, GGUF_BOOL, GGUF_STRING, GGUF_ARRAY, GGUF_UINT64, GGUF_INT64,
    GGUF_FLOAT64,
};

enum ggml_type {
    GGML_TYPE_F32 = 0, GGML_TYPE_F16 = 1, GGML_TYPE_Q8_0 = 8,
    GGML_TYPE_Q4_K = 12, GGML_TYPE_Q6_K = 14, GGML_TYPE_BF16 = 30,
};

typedef struct { const char *ptr; uint64_t len; } gguf_str;
typedef struct { uint32_t type; uint64_t len, data_offset; } gguf_array;
typedef struct {
    gguf_str key;
    uint32_t type;
    uint64_t value_offset;
} gguf_kv;
typedef struct {
    gguf_str name;
    uint32_t ndim, type;
    uint64_t dims[4], offset, bytes;
    const void *data;
} gguf_tensor;
typedef struct {
    int fd;
    const uint8_t *map;
    uint64_t size, tensor_data_offset, alignment;
    uint32_t version;
    uint64_t n_tensors, n_kv;
    gguf_kv *kv;
    gguf_tensor *tensors;
    char error[256];
} gguf_file;
typedef struct { const char *name; uint32_t block_elems, block_bytes; } ggml_type_info;

const ggml_type_info *gguf_type_info(uint32_t type);
int gguf_open(gguf_file *file, const char *path);
void gguf_close(gguf_file *file);
const gguf_tensor *gguf_tensor_find(const gguf_file *file, const char *name);
const gguf_kv *gguf_kv_find(const gguf_file *file, const char *key);
int gguf_get_u32(const gguf_file *file, const char *key, uint32_t *value);
int gguf_get_f32(const gguf_file *file, const char *key, float *value);
int gguf_get_string(const gguf_file *file, const char *key, gguf_str *value);
int gguf_get_array(const gguf_file *file, const char *key, gguf_array *value);
int gguf_array_string_at(const gguf_file *file, const gguf_array *array,
                         uint64_t index, gguf_str *value);
int gguf_array_i32_at(const gguf_file *file, const gguf_array *array,
                      uint64_t index, int32_t *value);

#endif
