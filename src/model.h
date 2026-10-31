#ifndef DSM_MODEL_H
#define DSM_MODEL_H

#include "gguf.h"

typedef struct {
    const gguf_tensor *attn_norm, *q, *k, *v, *o, *q_norm, *k_norm;
    const gguf_tensor *ffn_norm, *gate, *up, *down;
    const gguf_tensor *gate_inp, *gate_exps, *up_exps, *down_exps;
} dsm_layer;
typedef struct {
    gguf_file gguf;
    uint32_t layers, context, dim, hidden, heads, kv_heads, key_dim, value_dim, vocab;
    uint32_t experts, experts_used, expert_hidden;
    int is_moe;
    float rope_base, rms_eps;
    const gguf_tensor *token_embd, *output_norm, *output;
    dsm_layer *layer;
    char error[256];
} dsm_model;

int dsm_model_open(dsm_model *model, const char *path);
void dsm_model_close(dsm_model *model);

#endif
