#ifndef DSM_MODEL_H
#define DSM_MODEL_H

#include "gguf.h"

enum dsm_arch {
    DSM_ARCH_QWEN3,
    DSM_ARCH_QWEN3MOE,
    DSM_ARCH_QWEN35,
    DSM_ARCH_QWEN35MOE,
};

typedef struct {
    const gguf_tensor *attn_norm, *q, *k, *v, *o, *q_norm, *k_norm;
    const gguf_tensor *ffn_norm, *gate, *up, *down;
    const gguf_tensor *gate_inp, *gate_exps, *up_exps, *down_exps;
    /* Qwen3.5 DeltaNet mixer tensors (used only when is_recurrent is set). */
    const gguf_tensor *qkv, *z, *conv1d, *dt_bias, *ssm_a, *alpha, *beta;
    const gguf_tensor *ssm_norm, *ssm_out;
    /* Qwen3.5 MoE shared expert and its scalar gate. */
    const gguf_tensor *gate_shexp, *up_shexp, *down_shexp, *gate_inp_shexp;
} dsm_layer;
typedef struct {
    gguf_file gguf;
    enum dsm_arch arch;
    uint32_t layers, context, dim, hidden, heads, kv_heads, key_dim, value_dim, vocab;
    uint32_t q_width, attn_width;
    uint32_t experts, experts_used, expert_hidden;
    /* Qwen3.5 hybrid schedule: recurrent marks DeltaNet layers; attn_index maps
       global layer indices to compact full-attention KV-cache indices. */
    uint8_t *is_recurrent;
    uint32_t *attn_index, n_attn_layers;
    /* Qwen3.5 partial MRoPE configuration. */
    uint32_t rope_dims, rope_sections[4];
    /* Qwen3.5 DeltaNet dimensions. */
    uint32_t ssm_conv, ssm_state, ssm_groups, ssm_vheads, ssm_inner;
    /* Qwen3.5 MoE shared FFN width and fused Q/attention-gate projection flag. */
    uint32_t shared_hidden, attn_gate_fused;
    /* Qwen3.5 MoE routing defaults/overrides: renormalization, scaling, and
       gating operator; rope_type retains the selected partial-IMRoPE mode. */
    float expert_weights_scale;
    uint32_t expert_weights_norm, expert_gating_func, rope_type;
    int is_moe;
    float rope_base, rms_eps;
    const gguf_tensor *token_embd, *output_norm, *output;
    dsm_layer *layer;
    char error[256];
} dsm_model;

int dsm_model_open(dsm_model *model, const char *path);
void dsm_model_close(dsm_model *model);
int dsm_model_tensor_summary(const dsm_model *model, uint64_t *mapped,
                             uint64_t *unknown);

#endif
