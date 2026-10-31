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
static int expert_matrix(const gguf_tensor *t, uint64_t in, uint64_t out,
                         uint64_t experts) {
    return t && t->ndim == 3 && t->dims[0] == in && t->dims[1] == out &&
           t->dims[2] == experts && dsm_quant_supported(t->type);
}
static int vector(const gguf_tensor *t, uint64_t n) {
    return t && t->ndim == 1 && t->dims[0] == n && t->type == GGML_TYPE_F32;
}
static int metadata_bool(const gguf_file *g, const char *key, int expected) {
    const gguf_kv *kv = gguf_kv_find(g, key);
    if (!kv) return 1;
    if (kv->type != GGUF_BOOL) return 0;
    return g->map[kv->value_offset] == expected;
}
int dsm_model_open(dsm_model *m, const char *path) {
    memset(m, 0, sizeof(*m));
    if (!gguf_open(&m->gguf, path)) { snprintf(m->error, sizeof(m->error), "%s", m->gguf.error); return 0; }
    gguf_str arch;
    if (!gguf_get_string(&m->gguf, "general.architecture", &arch) ||
        !((arch.len == 5 && !memcmp(arch.ptr, "qwen3", 5)) ||
          (arch.len == 8 && !memcmp(arch.ptr, "qwen3moe", 8)))) {
        fail(m, "unsupported architecture"); goto bad;
    }
    m->is_moe = arch.len == 8;
    const char *prefix = m->is_moe ? "qwen3moe" : "qwen3";
    char key[96];
#define META_U32(name, value) (snprintf(key, sizeof(key), "%s.%s", prefix, name), gguf_get_u32(&m->gguf, key, value))
#define META_F32(name, value) (snprintf(key, sizeof(key), "%s.%s", prefix, name), gguf_get_f32(&m->gguf, key, value))
    if (!META_U32("block_count", &m->layers) ||
        !META_U32("context_length", &m->context) ||
        !META_U32("embedding_length", &m->dim) ||
        (!m->is_moe && !META_U32("feed_forward_length", &m->hidden)) ||
        (m->is_moe && (!META_U32("expert_count", &m->experts) ||
                       !META_U32("expert_used_count", &m->experts_used) ||
                       !META_U32("expert_feed_forward_length", &m->expert_hidden))) ||
        !META_U32("attention.head_count", &m->heads) ||
        !META_U32("attention.head_count_kv", &m->kv_heads) ||
        !META_U32("attention.key_length", &m->key_dim) ||
        !META_U32("attention.value_length", &m->value_dim) ||
        !META_F32("rope.freq_base", &m->rope_base) ||
        !META_F32("attention.layer_norm_rms_epsilon", &m->rms_eps) ||
        (m->is_moe && (!metadata_bool(&m->gguf, "qwen3moe.expert_weights_norm", 1) ||
                       !metadata_bool(&m->gguf, "qwen3moe.norm_top_k_prob", 1))) ||
        (m->is_moe && (!m->experts || !m->experts_used || m->experts_used > m->experts)) ||
        !m->layers || m->heads % m->kv_heads || m->key_dim != m->value_dim) {
        fail(m, "unsupported or invalid Qwen3 metadata"); goto bad;
    }
#undef META_U32
#undef META_F32
    m->q_width = m->heads * m->key_dim;
    m->attn_width = m->heads * m->value_dim;
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
        if (m->is_moe) {
            l->gate_inp = need(m, "blk.%u.ffn_gate_inp.weight", i);
            l->gate_exps = need(m, "blk.%u.ffn_gate_exps.weight", i);
            l->up_exps = need(m, "blk.%u.ffn_up_exps.weight", i);
            l->down_exps = need(m, "blk.%u.ffn_down_exps.weight", i);
        } else {
            l->gate = need(m, "blk.%u.ffn_gate.weight", i); l->up = need(m, "blk.%u.ffn_up.weight", i);
            l->down = need(m, "blk.%u.ffn_down.weight", i);
        }
        if (!l->attn_norm || !l->q || !l->k || !l->v || !l->o || !l->q_norm || !l->k_norm || !l->ffn_norm ||
            !vector(l->attn_norm, m->dim) || !matrix(l->q, m->dim, m->q_width) || !matrix(l->k, m->dim, m->kv_heads * m->key_dim) ||
            !matrix(l->v, m->dim, m->kv_heads * m->value_dim) || !matrix(l->o, m->attn_width, m->dim) || !vector(l->q_norm, m->key_dim) ||
            !vector(l->k_norm, m->key_dim) || !vector(l->ffn_norm, m->dim) ||
            (!m->is_moe && (!matrix(l->gate, m->dim, m->hidden) || !matrix(l->up, m->dim, m->hidden) || !matrix(l->down, m->hidden, m->dim))) ||
            (m->is_moe && (!matrix(l->gate_inp, m->dim, m->experts) ||
                           !expert_matrix(l->gate_exps, m->dim, m->expert_hidden, m->experts) ||
                           !expert_matrix(l->up_exps, m->dim, m->expert_hidden, m->experts) ||
                           !expert_matrix(l->down_exps, m->expert_hidden, m->dim, m->experts)))) { fail(m, "invalid layer tensor shape or type"); goto bad; }
    }
    return 1;
bad: dsm_model_close(m); return 0;
}
void dsm_model_close(dsm_model *m) { free(m->layer); gguf_close(&m->gguf); m->layer = NULL; }
