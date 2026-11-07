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

static int q35_vector(const gguf_tensor *t, uint64_t n) { return vector(t, n); }
static int q35_matrix(const gguf_tensor *t, uint64_t in, uint64_t out) { return matrix(t, in, out); }
static int q35_expert_matrix(const gguf_tensor *t, uint64_t in, uint64_t out, uint64_t experts) { return expert_matrix(t, in, out, experts); }
static const gguf_tensor *q35_need(dsm_model *m, const char *format, uint32_t layer) { return need(m, format, layer); }
static int q35_arch(const dsm_model *m) { return m->arch == DSM_ARCH_QWEN35 || m->arch == DSM_ARCH_QWEN35MOE; }

static int q35_meta_u32(dsm_model *m, const char *prefix, const char *name, uint32_t *value) {
    char key[96]; snprintf(key, sizeof(key), "%s.%s", prefix, name);
    return gguf_get_u32(&m->gguf, key, value);
}
static int q35_meta_f32(dsm_model *m, const char *prefix, const char *name, float *value) {
    char key[96]; snprintf(key, sizeof(key), "%s.%s", prefix, name);
    return gguf_get_f32(&m->gguf, key, value);
}
static int q35_load_schedule(dsm_model *m, const char *prefix) {
    char key[96]; gguf_array a;
    m->is_recurrent = calloc(m->layers, sizeof(*m->is_recurrent));
    m->attn_index = calloc(m->layers, sizeof(*m->attn_index));
    if (!m->is_recurrent || !m->attn_index) { fail(m, "out of memory"); return 0; }
    snprintf(key, sizeof(key), "%s.attention.recurrent_layers", prefix);
    if (gguf_get_array(&m->gguf, key, &a)) {
        if (a.type != GGUF_BOOL || a.len != m->layers) { fail(m, "invalid qwen35 attention.recurrent_layers metadata"); return 0; }
        for (uint32_t i = 0; i < m->layers; ++i) { int value; if (!gguf_array_bool_at(&m->gguf, &a, i, &value)) { fail(m, "invalid qwen35 attention.recurrent_layers values"); return 0; } m->is_recurrent[i] = (uint8_t)value; }
    } else {
        uint32_t interval;
        if (!q35_meta_u32(m, prefix, "full_attention_interval", &interval) || !interval) { fail(m, "invalid qwen35 full_attention_interval metadata"); return 0; }
        for (uint32_t i = 0; i < m->layers; ++i) m->is_recurrent[i] = (uint8_t)((i + 1) % interval != 0);
    }
    for (uint32_t i = 0; i < m->layers; ++i) if (!m->is_recurrent[i]) m->attn_index[i] = m->n_attn_layers++;
    if (!m->n_attn_layers) { fail(m, "qwen35 requires a full-attention layer"); return 0; }
    return 1;
}
static int q35_load_metadata(dsm_model *m, const char *prefix) {
    gguf_array sections; char key[96];
    if (!q35_meta_u32(m,prefix,"block_count",&m->layers) || !q35_meta_u32(m,prefix,"context_length",&m->context) || !q35_meta_u32(m,prefix,"embedding_length",&m->dim) || !q35_meta_u32(m,prefix,"attention.head_count",&m->heads) || !q35_meta_u32(m,prefix,"attention.head_count_kv",&m->kv_heads) || !q35_meta_u32(m,prefix,"attention.key_length",&m->key_dim) || !q35_meta_u32(m,prefix,"attention.value_length",&m->value_dim) || !q35_meta_f32(m,prefix,"rope.freq_base",&m->rope_base) || !q35_meta_f32(m,prefix,"attention.layer_norm_rms_epsilon",&m->rms_eps) || !q35_meta_u32(m,prefix,"rope.dimension_count",&m->rope_dims) || !q35_meta_u32(m,prefix,"ssm.conv_kernel",&m->ssm_conv) || !q35_meta_u32(m,prefix,"ssm.state_size",&m->ssm_state) || !q35_meta_u32(m,prefix,"ssm.group_count",&m->ssm_groups) || !q35_meta_u32(m,prefix,"ssm.time_step_rank",&m->ssm_vheads) || !q35_meta_u32(m,prefix,"ssm.inner_size",&m->ssm_inner)) goto invalid;
    if (m->is_moe) {
        if (!q35_meta_u32(m,prefix,"expert_count",&m->experts) || !q35_meta_u32(m,prefix,"expert_used_count",&m->experts_used) || !q35_meta_u32(m,prefix,"expert_feed_forward_length",&m->expert_hidden) || !q35_meta_u32(m,prefix,"expert_shared_feed_forward_length",&m->shared_hidden)) goto invalid;
    } else if (!q35_meta_u32(m,prefix,"feed_forward_length",&m->hidden)) goto invalid;
    snprintf(key,sizeof(key),"%s.rope.dimension_sections",prefix);
    if (!gguf_get_array(&m->gguf,key,&sections) || sections.type != GGUF_INT32 || sections.len != 4) goto invalid;
    uint32_t sum=0;
    for(uint32_t i=0;i<4;++i){int32_t part;if(!gguf_array_i32_at(&m->gguf,&sections,i,&part)||part<0)goto invalid;m->rope_sections[i]=(uint32_t)part;sum+=(uint32_t)part;}
    if (!m->layers || !m->kv_heads || m->heads % m->kv_heads || m->key_dim != m->value_dim || !m->rope_dims || m->rope_dims > m->key_dim || m->rope_dims % 2 || sum != m->rope_dims/2 || m->rope_sections[3] || !m->ssm_conv || !m->ssm_state || !m->ssm_groups || !m->ssm_vheads || !m->ssm_inner || m->ssm_vheads % m->ssm_groups || m->ssm_inner != m->ssm_vheads*m->ssm_state || (m->is_moe && (!m->experts || !m->experts_used || m->experts_used > m->experts))) goto invalid;
    m->q_width=m->heads*m->key_dim; m->attn_width=m->heads*m->value_dim; m->attn_gate_fused=1; m->rope_type=2; m->expert_weights_scale=1.0f; m->expert_gating_func=1;
    if(m->is_moe){snprintf(key,sizeof(key),"%s.expert_weights_norm",prefix);int found=metadata_bool(&m->gguf,key,1);if(!found&&gguf_kv_find(&m->gguf,key))goto invalid;m->expert_weights_norm=1;snprintf(key,sizeof(key),"%s.expert_weights_scale",prefix);if(gguf_kv_find(&m->gguf,key)&&!gguf_get_f32(&m->gguf,key,&m->expert_weights_scale))goto invalid;snprintf(key,sizeof(key),"%s.expert_gating_func",prefix);if(gguf_kv_find(&m->gguf,key)&&(!gguf_get_u32(&m->gguf,key,&m->expert_gating_func)||m->expert_gating_func!=1))goto invalid;}
    return q35_load_schedule(m,prefix);
invalid: fail(m,"unsupported or invalid Qwen3.5 metadata");return 0;
}
static int q35_load_embedding(dsm_model *m) {
    m->token_embd=gguf_tensor_find(&m->gguf,"token_embd.weight");m->output_norm=gguf_tensor_find(&m->gguf,"output_norm.weight");m->output=gguf_tensor_find(&m->gguf,"output.weight");if(!m->output)m->output=m->token_embd;if(!m->token_embd||m->token_embd->ndim!=2||m->token_embd->dims[0]!=m->dim||!dsm_quant_supported(m->token_embd->type)||!q35_vector(m->output_norm,m->dim)||!q35_matrix(m->output,m->dim,m->token_embd->dims[1])){fail(m,"invalid embedding or output tensor");return 0;}m->vocab=(uint32_t)m->token_embd->dims[1];return 1;
}
static int q35_load_layers(dsm_model *m) {
    uint32_t qkv_width=2*m->ssm_state*m->ssm_groups+m->ssm_inner;
    m->layer=calloc(m->layers,sizeof(*m->layer));if(!m->layer){fail(m,"out of memory");return 0;}
    for(uint32_t i=0;i<m->layers;++i){dsm_layer *l=&m->layer[i];l->attn_norm=q35_need(m,"blk.%u.attn_norm.weight",i);l->ffn_norm=q35_need(m,"blk.%u.post_attention_norm.weight",i);
        if(m->is_recurrent[i]){l->qkv=q35_need(m,"blk.%u.attn_qkv.weight",i);l->z=q35_need(m,"blk.%u.attn_gate.weight",i);l->conv1d=q35_need(m,"blk.%u.ssm_conv1d.weight",i);l->dt_bias=q35_need(m,"blk.%u.ssm_dt.bias",i);l->ssm_a=q35_need(m,"blk.%u.ssm_a",i);l->alpha=q35_need(m,"blk.%u.ssm_alpha.weight",i);l->beta=q35_need(m,"blk.%u.ssm_beta.weight",i);l->ssm_norm=q35_need(m,"blk.%u.ssm_norm.weight",i);l->ssm_out=q35_need(m,"blk.%u.ssm_out.weight",i);
        }else{l->q=q35_need(m,"blk.%u.attn_q.weight",i);l->k=q35_need(m,"blk.%u.attn_k.weight",i);l->v=q35_need(m,"blk.%u.attn_v.weight",i);l->o=q35_need(m,"blk.%u.attn_output.weight",i);l->q_norm=q35_need(m,"blk.%u.attn_q_norm.weight",i);l->k_norm=q35_need(m,"blk.%u.attn_k_norm.weight",i);}
        if(m->is_moe){l->gate_inp=q35_need(m,"blk.%u.ffn_gate_inp.weight",i);l->gate_exps=q35_need(m,"blk.%u.ffn_gate_exps.weight",i);l->up_exps=q35_need(m,"blk.%u.ffn_up_exps.weight",i);l->down_exps=q35_need(m,"blk.%u.ffn_down_exps.weight",i);l->gate_inp_shexp=q35_need(m,"blk.%u.ffn_gate_inp_shexp.weight",i);l->gate_shexp=q35_need(m,"blk.%u.ffn_gate_shexp.weight",i);l->up_shexp=q35_need(m,"blk.%u.ffn_up_shexp.weight",i);l->down_shexp=q35_need(m,"blk.%u.ffn_down_shexp.weight",i);
        }else{l->gate=q35_need(m,"blk.%u.ffn_gate.weight",i);l->up=q35_need(m,"blk.%u.ffn_up.weight",i);l->down=q35_need(m,"blk.%u.ffn_down.weight",i);}
        int valid=l->attn_norm&&l->ffn_norm&&q35_vector(l->attn_norm,m->dim)&&q35_vector(l->ffn_norm,m->dim);
        if(m->is_recurrent[i])valid=valid&&q35_matrix(l->qkv,m->dim,qkv_width)&&q35_matrix(l->z,m->dim,m->ssm_inner)&&l->conv1d&&l->conv1d->ndim==2&&l->conv1d->dims[0]==m->ssm_conv&&l->conv1d->dims[1]==qkv_width&&l->conv1d->type==GGML_TYPE_F32&&q35_vector(l->dt_bias,m->ssm_vheads)&&q35_vector(l->ssm_a,m->ssm_vheads)&&q35_matrix(l->alpha,m->dim,m->ssm_vheads)&&q35_matrix(l->beta,m->dim,m->ssm_vheads)&&q35_vector(l->ssm_norm,m->ssm_state)&&q35_matrix(l->ssm_out,m->ssm_inner,m->dim);
        else valid=valid&&q35_matrix(l->q,m->dim,2*m->q_width)&&q35_matrix(l->k,m->dim,m->kv_heads*m->key_dim)&&q35_matrix(l->v,m->dim,m->kv_heads*m->value_dim)&&q35_matrix(l->o,m->attn_width,m->dim)&&q35_vector(l->q_norm,m->key_dim)&&q35_vector(l->k_norm,m->key_dim);
        if(m->is_moe)valid=valid&&q35_matrix(l->gate_inp,m->dim,m->experts)&&q35_expert_matrix(l->gate_exps,m->dim,m->expert_hidden,m->experts)&&q35_expert_matrix(l->up_exps,m->dim,m->expert_hidden,m->experts)&&q35_expert_matrix(l->down_exps,m->expert_hidden,m->dim,m->experts)&&q35_vector(l->gate_inp_shexp,m->dim)&&q35_matrix(l->gate_shexp,m->dim,m->shared_hidden)&&q35_matrix(l->up_shexp,m->dim,m->shared_hidden)&&q35_matrix(l->down_shexp,m->shared_hidden,m->dim);else valid=valid&&q35_matrix(l->gate,m->dim,m->hidden)&&q35_matrix(l->up,m->dim,m->hidden)&&q35_matrix(l->down,m->hidden,m->dim);
        if(!valid){fail(m,"invalid Qwen3.5 layer tensor shape or type");return 0;}}
    return 1;
}
static int q35_mapped(const dsm_model *m,const gguf_tensor *t){if(t==m->token_embd||t==m->output_norm||t==m->output)return 1;for(uint32_t i=0;i<m->layers;++i){const dsm_layer*l=&m->layer[i];const gguf_tensor*all[]={l->attn_norm,l->q,l->k,l->v,l->o,l->q_norm,l->k_norm,l->ffn_norm,l->gate,l->up,l->down,l->gate_inp,l->gate_exps,l->up_exps,l->down_exps,l->qkv,l->z,l->conv1d,l->dt_bias,l->ssm_a,l->alpha,l->beta,l->ssm_norm,l->ssm_out,l->gate_shexp,l->up_shexp,l->down_shexp,l->gate_inp_shexp};for(size_t j=0;j<sizeof(all)/sizeof(*all);++j)if(t==all[j])return 1;}return 0;}
int dsm_model_tensor_summary(const dsm_model *m,uint64_t *mapped,uint64_t *unknown){uint64_t a=0,b=0;for(uint64_t i=0;i<m->gguf.n_tensors;++i){const gguf_tensor*t=&m->gguf.tensors[i];if(q35_mapped(m,t))++a;else if(!((t->name.len>=6&&!memcmp(t->name.ptr,"nextn.",6))||(t->name.len>=4&&!memcmp(t->name.ptr,"mtp.",4))||(t->name.len>=7&&!memcmp(t->name.ptr,"mmproj.",7))))++b;}if(mapped)*mapped=a;if(unknown)*unknown=b;return b==0;}
int dsm_model_open(dsm_model *m, const char *path) {
    memset(m, 0, sizeof(*m));
    if (!gguf_open(&m->gguf, path)) { snprintf(m->error, sizeof(m->error), "%s", m->gguf.error); return 0; }
    gguf_str arch;
    if (!gguf_get_string(&m->gguf, "general.architecture", &arch) ||
        !((arch.len == 5 && !memcmp(arch.ptr, "qwen3", 5)) ||
          (arch.len == 8 && !memcmp(arch.ptr, "qwen3moe", 8)) ||
          (arch.len == 6 && !memcmp(arch.ptr, "qwen35", 6)) ||
          (arch.len == 9 && !memcmp(arch.ptr, "qwen35moe", 9)))) {
        fail(m, "unsupported architecture"); goto bad;
    }
    if (arch.len == 5) { m->arch = DSM_ARCH_QWEN3; m->is_moe = 0; }
    else if (arch.len == 8) { m->arch = DSM_ARCH_QWEN3MOE; m->is_moe = 1; }
    else if (arch.len == 6) { m->arch = DSM_ARCH_QWEN35; m->is_moe = 0; }
    else { m->arch = DSM_ARCH_QWEN35MOE; m->is_moe = 1; }
    const char *prefix = m->arch == DSM_ARCH_QWEN3 ? "qwen3" :
                         m->arch == DSM_ARCH_QWEN3MOE ? "qwen3moe" :
                         m->arch == DSM_ARCH_QWEN35 ? "qwen35" : "qwen35moe";
    if (q35_arch(m)) {
        if (!q35_load_metadata(m, prefix) || !q35_load_embedding(m) ||
            !q35_load_layers(m)) goto bad;
        uint64_t mapped, unknown;
        dsm_model_tensor_summary(m, &mapped, &unknown);
        if (unknown) { fail(m, "unknown Qwen3.5 tensor (only nextn/mtp/mmproj may be ignored)"); goto bad; }
        return 1;
    }
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
void dsm_model_close(dsm_model *m) { free(m->is_recurrent); free(m->attn_index); free(m->layer); gguf_close(&m->gguf); m->is_recurrent = NULL; m->attn_index = NULL; m->layer = NULL; }
