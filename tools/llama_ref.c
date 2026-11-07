/*
 * llama-ref is intentionally a separate executable.  llama.cpp's libggml
 * exports gguf_* symbols which collide with DSM's private GGUF reader.
 *
 * Logit dump format (little endian):
 *   bytes 0..7   "DSMREF1\0"
 *   u32 version (2), u32 header_bytes (32), u32 vocab, u32 n_positions,
 *   u32 record_bytes (= 4 + 4*vocab), u32 reserved; then one record per row:
 *   u32 position, vocab f32 logits.  Positions are zero-based input-token
 * positions; each row predicts the token after its corresponding input token.
 */
#include <llama.h>

#include <errno.h>
#include <fnmatch.h>
#include <inttypes.h>
#include <limits.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { REF_VERSION = 2, REF_HEADER_BYTES = 32, REF_BATCH = 512 };

typedef struct {
    char magic[8];
    uint32_t version;
    uint32_t header_bytes;
    uint32_t vocab;
    uint32_t n_positions;
    uint32_t record_bytes;
    uint32_t reserved;
} ref_header;

typedef struct {
    const char *patterns;
    const char *out_dir;
    uint32_t position;
    bool warned_non_f32;
} dump_callback;

static void usage(const char *argv0) {
    fprintf(stderr,
            "usage: %s MODEL.gguf IDS.txt OUT.bin [options]\n"
            "       %s MODEL.gguf ID [ID ...] --out OUT.bin [options]\n"
            "options: --ngl 0|all|N  --logits-at all|last|K,K,...\n"
            "         --dump-tensors PATTERNS --tensor-dir DIR --greedy N\n",
            argv0, argv0);
}

static bool append_id(int32_t **ids, size_t *len, size_t *cap, long value) {
    if (value < 0 || value > INT32_MAX) return false;
    if (*len == *cap) {
        size_t new_cap = *cap ? *cap * 2 : 256;
        if (new_cap < *cap || new_cap > SIZE_MAX / sizeof(**ids)) return false;
        int32_t *new_ids = realloc(*ids, new_cap * sizeof(**ids));
        if (!new_ids) return false;
        *ids = new_ids;
        *cap = new_cap;
    }
    (*ids)[(*len)++] = (int32_t)value;
    return true;
}

static bool parse_id(const char *text, int32_t **ids, size_t *len, size_t *cap) {
    char *end = NULL;
    errno = 0;
    long value = strtol(text, &end, 10);
    return !errno && end != text && !*end && append_id(ids, len, cap, value);
}

static bool read_ids(const char *path, int32_t **ids, size_t *len, size_t *cap) {
    FILE *file = fopen(path, "r");
    if (!file) return false;
    char token[64];
    bool ok = true;
    while (fscanf(file, " %63[^, \t\r\n]", token) == 1) {
        if (!parse_id(token, ids, len, cap)) {
            ok = false;
            break;
        }
        int separator = fgetc(file);
        if (separator != ',' && separator != EOF) ungetc(separator, file);
    }
    if (ferror(file)) ok = false;
    fclose(file);
    return ok;
}

static bool selected_position(const char *selection, uint32_t position, uint32_t n_ids) {
    if (!strcmp(selection, "all")) return true;
    if (!strcmp(selection, "last")) return position + 1 == n_ids;
    const char *p = selection;
    while (*p) {
        char *end = NULL;
        errno = 0;
        unsigned long parsed = strtoul(p, &end, 10);
        if (errno || end == p || (end[0] && end[0] != ',')) return false;
        if (parsed == position) return true;
        p = *end ? end + 1 : end;
    }
    return false;
}

static bool tensor_matches(const char *patterns, const char *name) {
    if (!patterns || !*patterns) return false;
    const char *p = patterns;
    while (*p) {
        const char *comma = strchr(p, ',');
        size_t len = comma ? (size_t)(comma - p) : strlen(p);
        char pattern[GGML_MAX_NAME];
        if (len >= sizeof(pattern)) return false;
        memcpy(pattern, p, len);
        pattern[len] = '\0';
        if (!fnmatch(pattern, name, 0)) return true;
        p = comma ? comma + 1 : p + len;
    }
    return false;
}

static bool dump_eval_tensor(struct ggml_tensor *tensor, bool ask, void *data) {
    dump_callback *callback = data;
    if (ask) return tensor_matches(callback->patterns, tensor->name);
    if (!tensor_matches(callback->patterns, tensor->name)) return true;
    if (tensor->type != GGML_TYPE_F32) {
        if (!callback->warned_non_f32) {
            fprintf(stderr, "llama-ref: skipped non-F32 callback tensor(s)\n");
            callback->warned_non_f32 = true;
        }
        return true;
    }
    int written = snprintf(NULL, 0, "%s/%s-%u.f32", callback->out_dir,
                           tensor->name, callback->position);
    if (written < 0) return false;
    char *path = malloc((size_t)written + 1);
    if (!path) return false;
    snprintf(path, (size_t)written + 1, "%s/%s-%u.f32", callback->out_dir,
             tensor->name, callback->position);
    FILE *file = fopen(path, "wb");
    free(path);
    if (!file) return false;
    size_t bytes = ggml_nbytes(tensor);
    /* Callback tensors can live on Metal.  ggml_backend_tensor_get copies them. */
    float *copy = malloc(bytes);
    if (!copy) {
        fclose(file);
        return false;
    }
    ggml_backend_tensor_get(tensor, copy, 0, bytes);
    bool ok = fwrite(copy, 1, bytes, file) == bytes && !fclose(file);
    free(copy);
    return ok;
}

static uint32_t argmax(const float *logits, uint32_t vocab) {
    uint32_t best = 0;
    for (uint32_t i = 1; i < vocab; ++i)
        if (logits[i] > logits[best]) best = i;
    return best;
}

int main(int argc, char **argv) {
    const char *model_path = NULL, *ids_path = NULL, *out_path = NULL;
    const char *selection = "all", *patterns = NULL, *tensor_dir = NULL;
    int32_t n_gpu_layers = -1;
    uint32_t greedy = 0;
    int32_t *ids = NULL;
    size_t n_ids = 0, ids_cap = 0, n_selected = 0;
    struct llama_model *model = NULL;
    struct llama_context *context = NULL;
    struct llama_batch batch = {0};
    FILE *out = NULL;
    float *logit_row = NULL;
    int status = 1;

    if (argc < 4) {
        usage(argv[0]);
        return 2;
    }
    model_path = argv[1];
    ids_path = argv[2];
    int arg = 3;
    if (strchr(ids_path, '/') || strchr(ids_path, '.')) {
        out_path = argv[arg++];
    } else {
        while (arg < argc && strncmp(argv[arg], "--", 2)) {
            if (!parse_id(argv[arg++], &ids, &n_ids, &ids_cap)) {
                fprintf(stderr, "llama-ref: invalid token id\n");
                goto done;
            }
        }
    }
    while (arg < argc) {
        if (!strcmp(argv[arg], "--out") && arg + 1 < argc) out_path = argv[++arg];
        else if (!strcmp(argv[arg], "--ngl") && arg + 1 < argc) {
            const char *value = argv[++arg];
            n_gpu_layers = !strcmp(value, "all") ? -1 : (int32_t)strtol(value, NULL, 10);
        } else if (!strcmp(argv[arg], "--logits-at") && arg + 1 < argc) selection = argv[++arg];
        else if (!strcmp(argv[arg], "--dump-tensors") && arg + 1 < argc) patterns = argv[++arg];
        else if (!strcmp(argv[arg], "--tensor-dir") && arg + 1 < argc) tensor_dir = argv[++arg];
        else if (!strcmp(argv[arg], "--greedy") && arg + 1 < argc) greedy = (uint32_t)strtoul(argv[++arg], NULL, 10);
        else {
            usage(argv[0]);
            status = 2;
            goto done;
        }
        ++arg;
    }
    if (!out_path || (patterns && !tensor_dir) || (!patterns && tensor_dir)) {
        usage(argv[0]);
        status = 2;
        goto done;
    }
    if (!ids && !read_ids(ids_path, &ids, &n_ids, &ids_cap)) {
        fprintf(stderr, "llama-ref: cannot parse token id file %s\n", ids_path);
        goto done;
    }
    if (!n_ids || n_ids > UINT32_MAX) {
        fprintf(stderr, "llama-ref: need 1..%u token ids\n", UINT32_MAX);
        goto done;
    }
    for (uint32_t i = 0; i < n_ids; ++i) {
        if (selected_position(selection, i, (uint32_t)n_ids)) ++n_selected;
    }
    if (!n_selected) {
        fprintf(stderr, "llama-ref: --logits-at selected no positions\n");
        goto done;
    }

    llama_backend_init();
    struct llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = n_gpu_layers;
    model = llama_model_load_from_file(model_path, model_params);
    if (!model) {
        fprintf(stderr, "llama-ref: cannot load %s\n", model_path);
        goto done;
    }
    uint32_t vocab = (uint32_t)llama_vocab_n_tokens(llama_model_get_vocab(model));
    if (!vocab) {
        fprintf(stderr, "llama-ref: model has no vocabulary\n");
        goto done;
    }
    struct llama_context_params context_params = llama_context_default_params();
    context_params.n_ctx = (uint32_t)n_ids + greedy;
    context_params.n_batch = REF_BATCH;
    context_params.n_ubatch = REF_BATCH;
    context_params.type_k = GGML_TYPE_F32;
    context_params.type_v = GGML_TYPE_F32;
    context_params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    context_params.no_perf = true;
    dump_callback callback = {.patterns = patterns, .out_dir = tensor_dir};
    if (patterns) {
        context_params.cb_eval = dump_eval_tensor;
        context_params.cb_eval_user_data = &callback;
    }
    context = llama_init_from_model(model, context_params);
    if (!context) {
        fprintf(stderr, "llama-ref: cannot create context\n");
        goto done;
    }
    batch = llama_batch_init(REF_BATCH, 0, 1);
    if (!batch.token) {
        fprintf(stderr, "llama-ref: cannot allocate batch\n");
        goto done;
    }
    out = fopen(out_path, "wb");
    if (!out) {
        fprintf(stderr, "llama-ref: cannot open %s\n", out_path);
        goto done;
    }
    ref_header header = {.magic = "DSMREF1", .version = REF_VERSION,
                         .header_bytes = REF_HEADER_BYTES, .vocab = vocab,
                         .n_positions = n_selected,
                         .record_bytes = sizeof(uint32_t) + vocab * sizeof(float)};
    if (fwrite(&header, sizeof(header), 1, out) != 1) goto done;
    logit_row = malloc((size_t)vocab * sizeof(*logit_row));
    if (!logit_row) goto done;

    uint32_t output = 0;
    for (uint32_t start = 0; start < n_ids; start += REF_BATCH) {
        uint32_t count = (uint32_t)n_ids - start;
        if (count > REF_BATCH) count = REF_BATCH;
        batch.n_tokens = (int32_t)count;
        for (uint32_t i = 0; i < count; ++i) {
            batch.token[i] = ids[start + i];
            batch.pos[i] = (llama_pos)(start + i);
            batch.n_seq_id[i] = 1;
            batch.seq_id[i][0] = 0;
            batch.logits[i] = selected_position(selection, start + i, (uint32_t)n_ids);
        }
        callback.position = start;
        if (llama_decode(context, batch)) {
            fprintf(stderr, "llama-ref: inference failed at input position %u\n", start);
            goto done;
        }
        for (uint32_t i = 0; i < count; ++i) {
            if (!batch.logits[i]) continue;
            float *source = llama_get_logits_ith(context, (int32_t)output++);
            if (!source) {
                fprintf(stderr, "llama-ref: missing logits at input position %u\n", start + i);
                goto done;
            }
            memcpy(logit_row, source, (size_t)vocab * sizeof(*logit_row));
            uint32_t position = start + i;
            if (fwrite(&position, sizeof(position), 1, out) != 1 ||
                fwrite(logit_row, sizeof(*logit_row), vocab, out) != vocab) goto done;
        }
    }
    for (uint32_t i = 0; i < greedy; ++i) {
        uint32_t token = argmax(logit_row, vocab);
        char piece[256];
        int length = llama_token_to_piece(llama_model_get_vocab(model), token, piece,
                                          sizeof(piece), 0, false);
        if (length > 0) fwrite(piece, 1, (size_t)length, stdout);
        batch.n_tokens = 1;
        batch.token[0] = (llama_token)token;
        batch.pos[0] = (llama_pos)(n_ids + i);
        batch.n_seq_id[0] = 1;
        batch.seq_id[0][0] = 0;
        batch.logits[0] = 1;
        if (llama_decode(context, batch)) break;
        float *source = llama_get_logits_ith(context, 0);
        if (!source) break;
        memcpy(logit_row, source, (size_t)vocab * sizeof(*logit_row));
    }
    if (greedy) putchar('\n');
    if (fclose(out)) {
        out = NULL;
        goto done;
    }
    out = NULL;
    status = 0;
done:
    if (out) fclose(out);
    free(logit_row);
    llama_batch_free(batch);
    llama_free(context);
    llama_model_free(model);
    llama_backend_free();
    free(ids);
    return status;
}
