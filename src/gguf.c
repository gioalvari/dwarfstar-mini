#include "gguf.h"

#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

typedef struct { const uint8_t *p, *end; } cursor;

static const ggml_type_info TYPES[] = {
    [0] = {"f32", 1, 4}, [1] = {"f16", 1, 2}, [2] = {"q4_0", 32, 18},
    [3] = {"q4_1", 32, 20}, [6] = {"q5_0", 32, 22}, [7] = {"q5_1", 32, 24},
    [8] = {"q8_0", 32, 34}, [10] = {"q2_k", 256, 84}, [11] = {"q3_k", 256, 110},
    [12] = {"q4_k", 256, 144}, [13] = {"q5_k", 256, 176}, [14] = {"q6_k", 256, 210},
    [15] = {"q8_k", 256, 292}, [24] = {"i8", 1, 1}, [25] = {"i16", 1, 2},
    [26] = {"i32", 1, 4}, [27] = {"i64", 1, 8}, [28] = {"f64", 1, 8},
    [30] = {"bf16", 1, 2},
};

const ggml_type_info *gguf_type_info(uint32_t type) {
    return type < sizeof(TYPES) / sizeof(*TYPES) && TYPES[type].name ? &TYPES[type] : NULL;
}
static int take(cursor *c, void *out, size_t n) {
    if ((uint64_t)(c->end - c->p) < n) return 0;
    if (out) memcpy(out, c->p, n);
    c->p += n; return 1;
}
static int u32(cursor *c, uint32_t *v) { return take(c, v, 4); }
static int u64(cursor *c, uint64_t *v) { return take(c, v, 8); }
static int str(cursor *c, gguf_str *s) {
    if (!u64(c, &s->len) || (uint64_t)(c->end - c->p) < s->len) return 0;
    s->ptr = (const char *)c->p; c->p += s->len; return 1;
}
static uint64_t scalar_size(uint32_t type) {
    static const uint8_t sizes[] = {1,1,2,2,4,4,4,1,0,0,8,8,8};
    return type < sizeof(sizes) ? sizes[type] : 0;
}
static int skip_value(cursor *c, uint32_t type, unsigned depth) {
    uint64_t n = scalar_size(type);
    if (n) return take(c, NULL, (size_t)n);
    if (type == GGUF_STRING) { gguf_str s; return str(c, &s); }
    if (type != GGUF_ARRAY || depth == 8) return 0;
    uint32_t elem; uint64_t count;
    if (!u32(c, &elem) || !u64(c, &count)) return 0;
    n = scalar_size(elem);
    if (n) return count <= (uint64_t)(c->end - c->p) / n && take(c, NULL, (size_t)(count * n));
    for (uint64_t i = 0; i < count; ++i) if (!skip_value(c, elem, depth + 1)) return 0;
    return 1;
}
static int eq(gguf_str s, const char *text) {
    size_t n = strlen(text); return s.len == n && !memcmp(s.ptr, text, n);
}
static void fail(gguf_file *f, const char *message) {
    snprintf(f->error, sizeof(f->error), "%s", message);
}
static uint64_t aligned(uint64_t n, uint64_t a) { return (n + a - 1) / a * a; }

int gguf_open(gguf_file *f, const char *path) {
    memset(f, 0, sizeof(*f)); f->fd = -1;
    f->fd = open(path, O_RDONLY);
    struct stat st;
    if (f->fd < 0 || fstat(f->fd, &st) || st.st_size < 24) { fail(f, "cannot open GGUF"); goto bad; }
    f->size = (uint64_t)st.st_size;
    f->map = mmap(NULL, (size_t)f->size, PROT_READ, MAP_PRIVATE, f->fd, 0);
    if (f->map == MAP_FAILED) { f->map = NULL; fail(f, "cannot mmap GGUF"); goto bad; }
    cursor c = {f->map, f->map + f->size}; uint32_t magic;
    if (!u32(&c, &magic) || magic != 0x46554747 || !u32(&c, &f->version) ||
        !u64(&c, &f->n_tensors) || !u64(&c, &f->n_kv) || f->version != 3 ||
        f->n_kv > f->size || f->n_tensors > f->size) { fail(f, "invalid GGUF v3 header"); goto bad; }
    f->alignment = 32;
    f->kv = calloc((size_t)f->n_kv, sizeof(*f->kv));
    f->tensors = calloc((size_t)f->n_tensors, sizeof(*f->tensors));
    if (!f->kv || !f->tensors) { fail(f, "out of memory"); goto bad; }
    for (uint64_t i = 0; i < f->n_kv; ++i) {
        gguf_kv *kv = &f->kv[i];
        if (!str(&c, &kv->key) || !u32(&c, &kv->type)) { fail(f, "truncated metadata"); goto bad; }
        kv->value_offset = (uint64_t)(c.p - f->map);
        if (eq(kv->key, "general.alignment") && kv->type == GGUF_UINT32) {
            uint32_t a; if (!u32(&c, &a) || !a) { fail(f, "bad alignment"); goto bad; } f->alignment = a;
        } else if (!skip_value(&c, kv->type, 0)) { fail(f, "invalid metadata"); goto bad; }
    }
    for (uint64_t i = 0; i < f->n_tensors; ++i) {
        gguf_tensor *t = &f->tensors[i]; uint64_t elems = 1;
        if (!str(&c, &t->name) || !u32(&c, &t->ndim) || !t->ndim || t->ndim > 4) { fail(f, "invalid tensor directory"); goto bad; }
        for (uint32_t d = 0; d < t->ndim; ++d) {
            if (!u64(&c, &t->dims[d]) || !t->dims[d] || elems > UINT64_MAX / t->dims[d]) { fail(f, "bad tensor dimensions"); goto bad; }
            elems *= t->dims[d];
        }
        if (!u32(&c, &t->type) || !u64(&c, &t->offset)) { fail(f, "truncated tensor directory"); goto bad; }
        const ggml_type_info *info = gguf_type_info(t->type);
        if (!info || elems % info->block_elems) { fail(f, "unsupported tensor type or shape"); goto bad; }
        t->bytes = elems / info->block_elems * info->block_bytes;
    }
    f->tensor_data_offset = aligned((uint64_t)(c.p - f->map), f->alignment);
    for (uint64_t i = 0; i < f->n_tensors; ++i) {
        gguf_tensor *t = &f->tensors[i];
        if (t->offset > f->size - f->tensor_data_offset || t->bytes > f->size - f->tensor_data_offset - t->offset) { fail(f, "tensor outside file"); goto bad; }
        t->data = f->map + f->tensor_data_offset + t->offset;
    }
    return 1;
bad: gguf_close(f); return 0;
}
void gguf_close(gguf_file *f) {
    free(f->kv); free(f->tensors);
    if (f->map) munmap((void *)f->map, (size_t)f->size);
    if (f->fd >= 0) close(f->fd);
    f->map = NULL; f->kv = NULL; f->tensors = NULL; f->fd = -1;
}
const gguf_tensor *gguf_tensor_find(const gguf_file *f, const char *name) {
    for (uint64_t i = 0; i < f->n_tensors; ++i) if (eq(f->tensors[i].name, name)) return &f->tensors[i];
    return NULL;
}
const gguf_kv *gguf_kv_find(const gguf_file *f, const char *key) {
    for (uint64_t i = 0; i < f->n_kv; ++i) if (eq(f->kv[i].key, key)) return &f->kv[i];
    return NULL;
}
static const uint8_t *value(const gguf_file *f, const char *key, uint32_t type) {
    const gguf_kv *kv = gguf_kv_find(f, key);
    return kv && kv->type == type ? f->map + kv->value_offset : NULL;
}
int gguf_get_u32(const gguf_file *f, const char *key, uint32_t *v) {
    const gguf_kv *kv = gguf_kv_find(f, key); const uint8_t *p = kv ? f->map + kv->value_offset : NULL;
    if (!p) return 0; if (kv->type == GGUF_UINT32) { memcpy(v, p, 4); return 1; }
    if (kv->type == GGUF_UINT64) { uint64_t n; memcpy(&n, p, 8); if (n <= UINT32_MAX) { *v = (uint32_t)n; return 1; } }
    return 0;
}
int gguf_get_f32(const gguf_file *f, const char *key, float *v) {
    const gguf_kv *kv = gguf_kv_find(f, key); const uint8_t *p = kv ? f->map + kv->value_offset : NULL;
    if (!p) return 0; if (kv->type == GGUF_FLOAT32) { memcpy(v, p, 4); return 1; }
    if (kv->type == GGUF_FLOAT64) { double n; memcpy(&n, p, 8); *v = (float)n; return 1; }
    return 0;
}
int gguf_get_string(const gguf_file *f, const char *key, gguf_str *v) {
    const uint8_t *p = value(f, key, GGUF_STRING); if (!p) return 0;
    memcpy(&v->len, p, 8); if (v->len > f->size - (uint64_t)(p - f->map) - 8) return 0;
    v->ptr = (const char *)(p + 8); return 1;
}
int gguf_get_array(const gguf_file *f, const char *key, gguf_array *v) {
    const uint8_t *p = value(f, key, GGUF_ARRAY); if (!p) return 0;
    memcpy(&v->type, p, 4); memcpy(&v->len, p + 4, 8); v->data_offset = (uint64_t)(p - f->map) + 12;
    return v->data_offset <= f->size;
}
int gguf_array_string_at(const gguf_file *f, const gguf_array *a, uint64_t index, gguf_str *v) {
    if (a->type != GGUF_STRING || index >= a->len) return 0;
    cursor c = {f->map + a->data_offset, f->map + f->size};
    for (uint64_t i = 0; i <= index; ++i) if (!str(&c, v)) return 0;
    return 1;
}
int gguf_array_i32_at(const gguf_file *f, const gguf_array *a, uint64_t index, int32_t *v) {
    if (a->type != GGUF_INT32 || index >= a->len || index > (f->size - a->data_offset) / 4) return 0;
    memcpy(v, f->map + a->data_offset + index * 4, 4); return 1;
}
