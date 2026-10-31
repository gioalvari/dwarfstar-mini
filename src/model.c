#include "model.h"
#include "quants.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void fail(dsm_model *m, const char *what) { snprintf(m->error, sizeof(m->error), "%s", what); }
static const gguf_tensor *need(dsm_model *m, const char *format, uint32_t layer) {
    char name[64]; snprintf(name, sizeof(name), format, layer);
    const gguf_tensor *t = gguf_tensor_find(&m->gguf, name);
    if (!t) fail(m, name); return t;
}
static int matrix(const gguf_tensor *t, uint64_t in, uint64_t out) {
    return t && t->ndim == 2 && t->dims[0] == in && t->dims[1] == out &&
           dsm_quant_supported(t->type);
}
static int vector(const gguf_tensor *t, uint64_t n) {
    return t && t->ndim == 1 && t->dims[0] == n && t->type == GGML_TYPE_F32;
}
int dsm_model_open(dsm_model *m, const char *path) {
    memset(m, 0, sizeof(*m));
    if (!gguf_open(&m->gguf, path)) { snprintf(m->error, sizeof(m->error), "%s", m->gguf.error); return 0; }
    gguf_str arch;
    if (!gguf_get_string(&m->gguf, "general.architecture", &arch) || arch.len != 5 || memcmp(arch.ptr, "qwen3", 5) ||
        !gguf_get_u32(&m->gguf, "qwen3.block_count", &m->layers) ||
        !gguf_get_u32(&m->gguf, "qwen3.context_length", &m->context) ||
        !gguf_get_u32(&m->gguf, "qwen3.embedding_length", &m->dim) ||
        !gguf_get_u32(&m->gguf, "qwen3.feed_forward_length", &m->hidden) ||
        !gguf_get_u32(&m->gguf, "qwen3.attention.head_count", &m->heads) ||
        !gguf_get_u32(&m->gguf, "qwen3.attention.head_count_kv", &m->kv_heads) ||
        !gguf_get_u32(&m->gguf, "qwen3.attention.key_length", &m->key_dim) ||
        !gguf_get_u32(&m->gguf, "qwen3.attention.value_length", &m->value_dim) ||
        !gguf_get_f32(&m->gguf, "qwen3.rope.freq_base", &m->rope_base) ||
        !gguf_get_f32(&m->gguf, "qwen3.attention.layer_norm_rms_epsilon", &m->rms_eps) ||
        !m->layers || m->heads % m->kv_heads || m->dim != m->heads * m->key_dim || m->key_dim != m->value_dim) {
        fail(m, "unsupported or invalid Qwen3 metadata"); goto bad;
    }
    m->token_embd = gguf_tensor_find(&m->gguf, "token_embd.weight");
    m->output_norm = gguf_tensor_find(&m->gguf, "output_norm.weight");
    m->output = gguf_tensor_find(&m->gguf, "output.weight");
    if (!m->output) m->output = m->token_embd;
    if (!m->token_embd || m->token_embd->ndim != 2 || m->token_embd->dims[0] != m->dim ||
        !dsm_quant_supported(m->token_embd->type) || !vector(m->output_norm, m->dim) ||
        !matrix(m->output, m->dim, m->token_embd->dims[1])) { fail(m, "invalid embedding or output tensor"); goto bad; }
    m->vocab = (uint32_t)m->token_embd->dims[1];
    m->layer = calloc(m->layers, sizeof(*m->layer)); if (!m->layer) { fail(m, "out of memory"); goto bad; }
    for (uint32_t i = 0; i < m->layers; ++i) {
        dsm_layer *l = &m->layer[i];
        l->attn_norm = need(m, "blk.%u.attn_norm.weight", i); l->q = need(m, "blk.%u.attn_q.weight", i);
        l->k = need(m, "blk.%u.attn_k.weight", i); l->v = need(m, "blk.%u.attn_v.weight", i);
        l->o = need(m, "blk.%u.attn_output.weight", i); l->q_norm = need(m, "blk.%u.attn_q_norm.weight", i);
        l->k_norm = need(m, "blk.%u.attn_k_norm.weight", i); l->ffn_norm = need(m, "blk.%u.ffn_norm.weight", i);
        l->gate = need(m, "blk.%u.ffn_gate.weight", i); l->up = need(m, "blk.%u.ffn_up.weight", i);
        l->down = need(m, "blk.%u.ffn_down.weight", i);
        if (!l->attn_norm || !l->q || !l->k || !l->v || !l->o || !l->q_norm || !l->k_norm || !l->ffn_norm || !l->gate || !l->up || !l->down ||
            !vector(l->attn_norm, m->dim) || !matrix(l->q, m->dim, m->dim) || !matrix(l->k, m->dim, m->kv_heads * m->key_dim) ||
            !matrix(l->v, m->dim, m->kv_heads * m->value_dim) || !matrix(l->o, m->dim, m->dim) || !vector(l->q_norm, m->key_dim) ||
            !vector(l->k_norm, m->key_dim) || !vector(l->ffn_norm, m->dim) || !matrix(l->gate, m->dim, m->hidden) ||
            !matrix(l->up, m->dim, m->hidden) || !matrix(l->down, m->hidden, m->dim)) { fail(m, "invalid layer tensor shape or type"); goto bad; }
    }
    return 1;
bad: dsm_model_close(m); return 0;
}
void dsm_model_close(dsm_model *m) { free(m->layer); gguf_close(&m->gguf); m->layer = NULL; }
