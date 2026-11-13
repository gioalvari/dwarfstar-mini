#include "forward.h"
#include "json.h"
#include "metal.h"
#include "model.h"
#include "sampler.h"
#include "server.h"
#include "tokenizer.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static void usage(const char *argv0) { fprintf(stderr,"usage: %s -m MODEL -p PROMPT [-n N] [--backend metal|cpu] [--threads N] [--temp T] [--top-k K] [--top-p P] [--min-p P] [--seed N] [--chat] [--dump-logits FILE] [--tokens-only]\n       %s -m MODEL --render-chat REQUEST.json\n       %s serve -m MODEL [--backend metal|cpu] [--host 127.0.0.1] [--port 8080] [--ctx N] [--threads N] [--alias NAME]\n",argv0,argv0,argv0); }
static double now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec/1e9;}
typedef struct { char *data; size_t len, cap; } string;
static int append(string *out, const char *text, size_t length) {
    if (out->len + length + 1 > out->cap) {
        size_t cap = out->cap ? out->cap : 256;
        while (cap < out->len + length + 1) cap *= 2;
        char *data = realloc(out->data, cap);
        if (!data) return 0;
        out->data = data;
        out->cap = cap;
    }
    memcpy(out->data + out->len, text, length);
    out->len += length;
    out->data[out->len] = '\0';
    return 1;
}
static int append_lit(string *out, const char *text) { return append(out, text, strlen(text)); }
static int json_is(const dsm_json *value, const char *text) {
    return value && value->type == DSM_JSON_STRING && value->len == strlen(text) &&
           !memcmp(value->text, text, value->len);
}
/* The shipped Qwen3.5 template trims message content before rendering it. */
static int append_trimmed_content(string *out, const dsm_json *content, int strip_thinking) {
    if (!content || content->type != DSM_JSON_STRING) return 0;
    char *owned = dsm_json_string_dup(content);
    if (!owned) return 0;
    const char *text = owned;
    size_t length = strlen(text);
    if (strip_thinking) {
        const char *end = strstr(text, "</think>");
        if (end) { text = end + strlen("</think>"); length = strlen(text); }
    }
    while (length && (*text == ' ' || *text == '\t' || *text == '\r' || *text == '\n')) { ++text; --length; }
    while (length && (text[length - 1] == ' ' || text[length - 1] == '\t' || text[length - 1] == '\r' || text[length - 1] == '\n')) --length;
    int ok = append(out, text, length);
    free(owned);
    return ok;
}
static int render_qwen35_chat(const char *request_path, char **rendered) {
    FILE *file = fopen(request_path, "rb");
    dsm_json *root = NULL;
    string out = {0};
    char error[128], *request = NULL;
    long length;
    int ok = 0;
    *rendered = NULL;
    if (!file || fseek(file, 0, SEEK_END) || (length = ftell(file)) < 0 || fseek(file, 0, SEEK_SET) ||
        !(request = malloc((size_t)length + 1)) || fread(request, 1, (size_t)length, file) != (size_t)length) goto done;
    request[length] = '\0';
    if (!dsm_json_parse(request, (size_t)length, &root, error, sizeof(error))) goto done;
    const dsm_json *messages = dsm_json_object_get(root, "messages");
    const dsm_json *kwargs = dsm_json_object_get(root, "chat_template_kwargs");
    const dsm_json *thinking = kwargs ? dsm_json_object_get(kwargs, "enable_thinking") : NULL;
    if (!messages || messages->type != DSM_JSON_ARRAY || !messages->count) goto done;
    for (size_t i = 0; i < messages->count; ++i) {
        const dsm_json *message = messages->items[i];
        const dsm_json *role = dsm_json_object_get(message, "role");
        const dsm_json *content = dsm_json_object_get(message, "content");
        if (!role || role->type != DSM_JSON_STRING || !append_lit(&out, "<|im_start|>") ||
            !append(&out, role->text, role->len) || !append_lit(&out, "\n") ||
            !append_trimmed_content(&out, content, json_is(role, "assistant")) ||
            !append_lit(&out, "<|im_end|>\n")) goto done;
    }
    if (!append_lit(&out, "<|im_start|>assistant\n") ||
        !append_lit(&out, thinking && thinking->type == DSM_JSON_BOOL && !thinking->boolean ?
                    "<think>\n\n</think>\n\n" : "<think>\n")) goto done;
    *rendered = out.data;
    out.data = NULL;
    ok = 1;
done:
    if (file) fclose(file);
    free(request);
    free(out.data);
    dsm_json_free(root);
    return ok;
}
static int qwen35_arch(const gguf_file *gguf) {
    gguf_str arch;
    return gguf_get_string(gguf, "general.architecture", &arch) &&
           ((arch.len == 6 && !memcmp(arch.ptr, "qwen35", 6)) ||
            (arch.len == 9 && !memcmp(arch.ptr, "qwen35moe", 9)));
}
static int tokenize_only(const char *path, const char *prompt) {
    gguf_file gguf;
    dsm_tokenizer *tok = NULL;
    dsm_tokens ids;
    char error[256];
    int ok = gguf_open(&gguf, path) &&
             dsm_tokenizer_init(&tok, &gguf, error, sizeof(error)) &&
             dsm_tokenize(tok, prompt, 0, &ids);
    if (!ok) fprintf(stderr, "dsm: %s\n", tok ? "tokenization failed" : (gguf.error[0] ? gguf.error : error));
    if (ok) {
        for (size_t i = 0; i < ids.len; ++i) printf("%s%d", i ? " " : "", ids.ids[i]);
        puts("");
        dsm_tokens_free(&ids);
    }
    dsm_tokenizer_free(tok);
    gguf_close(&gguf);
    return ok;
}
int main(int argc, char **argv) {
    if (argc > 1 && (!strcmp(argv[1], "serve") || !strcmp(argv[1], "--serve"))) {
        dsm_server_config config = {
            .host = "127.0.0.1", .port = 8080, .threads = 8,
        };
        const char *path = NULL;
        for (int i = 2; i < argc; ++i) {
            if (!strcmp(argv[i], "-m") && i + 1 < argc) path = argv[++i];
            else if (!strcmp(argv[i], "--backend") && i + 1 < argc) config.backend = argv[++i];
            else if (!strcmp(argv[i], "--host") && i + 1 < argc) config.host = argv[++i];
            else if (!strcmp(argv[i], "--port") && i + 1 < argc) config.port = (uint16_t)strtoul(argv[++i], NULL, 10);
            else if (!strcmp(argv[i], "--ctx") && i + 1 < argc) config.context = (uint32_t)strtoul(argv[++i], NULL, 10);
            else if (!strcmp(argv[i], "--threads") && i + 1 < argc) config.threads = (uint32_t)strtoul(argv[++i], NULL, 10);
            else if (!strcmp(argv[i], "--alias") && i + 1 < argc) config.alias = argv[++i];
            else { usage(argv[0]); return 2; }
        }
        if (!path || !config.port || !config.threads) { usage(argv[0]); return 2; }
        return dsm_server_run(path, &config);
    }
    const char *path = NULL, *prompt = NULL, *dump = NULL, *backend = NULL, *render_chat = NULL;
    int n = 16, chat = 0, tokens_only = 0;
    uint32_t threads = 8;
    float temp = 0, top_p = 0.95f, min_p = 0;
    uint32_t top_k = 0;
    uint64_t seed = 1;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "-m") && i + 1 < argc) path = argv[++i];
        else if (!strcmp(argv[i], "-p") && i + 1 < argc) prompt = argv[++i];
        else if (!strcmp(argv[i], "-n") && i + 1 < argc) n = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--temp") && i + 1 < argc) temp = strtof(argv[++i], NULL);
        else if (!strcmp(argv[i], "--top-k") && i + 1 < argc) top_k = (uint32_t)strtoul(argv[++i], NULL, 10);
        else if (!strcmp(argv[i], "--top-p") && i + 1 < argc) top_p = strtof(argv[++i], NULL);
        else if (!strcmp(argv[i], "--min-p") && i + 1 < argc) min_p = strtof(argv[++i], NULL);
        else if (!strcmp(argv[i], "--seed") && i + 1 < argc) seed = strtoull(argv[++i], NULL, 10);
        else if (!strcmp(argv[i], "--backend") && i + 1 < argc) backend = argv[++i];
        else if (!strcmp(argv[i], "--threads") && i + 1 < argc) threads = (uint32_t)strtoul(argv[++i], NULL, 10);
        else if (!strcmp(argv[i], "--chat")) chat = 1;
        else if (!strcmp(argv[i], "--render-chat") && i + 1 < argc) render_chat = argv[++i];
        else if (!strcmp(argv[i], "--dump-logits") && i + 1 < argc) dump = argv[++i];
        else if (!strcmp(argv[i], "--tokens-only")) tokens_only = 1;
        else { usage(argv[0]); return 2; }
    }
    if (!path || (!prompt && !render_chat) || (prompt && render_chat) || n < 0 || !threads) { usage(argv[0]); return 2; }
    char *rendered = NULL;
    if (render_chat) {
        gguf_file gguf;
        if (!gguf_open(&gguf, path)) { fprintf(stderr, "dsm: %s\n", gguf.error); return 1; }
        int is_qwen35 = qwen35_arch(&gguf);
        gguf_close(&gguf);
        if (!is_qwen35 || !render_qwen35_chat(render_chat, &rendered)) {
            fprintf(stderr, "dsm: failed to render Qwen3.5 chat request\n");
            return 1;
        }
        fputs(rendered, stdout);
        free(rendered);
        return 0;
    }
    if (chat) {
        gguf_file gguf;
        if (!gguf_open(&gguf, path)) { fprintf(stderr, "dsm: %s\n", gguf.error); return 1; }
        size_t len = strlen(prompt) + 80;
        rendered = malloc(len);
        if (!rendered) { gguf_close(&gguf); return 1; }
        snprintf(rendered, len, "<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n%s", prompt,
                 qwen35_arch(&gguf) ? "<think>\n" : "");
        gguf_close(&gguf);
        prompt = rendered;
    }
    if (tokens_only) {
        int ok = tokenize_only(path, prompt);
        free(rendered);
        return ok ? 0 : 1;
    }
    dsm_model model;
    if (!dsm_model_open(&model, path)) { fprintf(stderr, "dsm: %s\n", model.error); free(rendered); return 1; }
    dsm_tokenizer *tok; char err[256];
    if (!dsm_tokenizer_init(&tok, &model.gguf, err, sizeof(err))) { fprintf(stderr, "dsm: %s\n", err); dsm_model_close(&model); free(rendered); return 1; }
    dsm_tokens ids;
    if (!dsm_tokenize(tok, prompt, 0, &ids)) { fprintf(stderr, "dsm: tokenization failed\n"); dsm_tokenizer_free(tok); dsm_model_close(&model); free(rendered); return 1; }
    int use_metal = backend ? !strcmp(backend, "metal") : dsm_metal_available();
    if (backend && strcmp(backend, "cpu") && strcmp(backend, "metal")) { fprintf(stderr, "dsm: unknown backend '%s'\n", backend); return 2; }
    dsm_forward *f = NULL; dsm_metal *metal = NULL; uint32_t context = (uint32_t)(ids.len + (size_t)n + 1); char metal_error[256];
    if (use_metal) {
        if (!dsm_metal_init(&metal, &model, context, metal_error, sizeof(metal_error))) {
            if (backend) { fprintf(stderr, "dsm: Metal initialization failed: %s\n", metal_error); return 1; }
            fprintf(stderr, "dsm: Metal unavailable (%s); falling back to CPU\n", metal_error); use_metal = 0;
        }
    }
    if (!use_metal && !dsm_forward_init(&f, &model, context, threads)) { fprintf(stderr, "dsm: cannot allocate CPU forward state\n"); return 1; }
    float *logits = malloc((size_t)model.vocab * sizeof(*logits)); if (!logits) return 1;
    double prefill_started = now();
    if (use_metal) {
        if (!dsm_metal_prefill(metal, ids.ids, (uint32_t)ids.len, logits)) { fprintf(stderr, "dsm: prompt inference failed\n"); return 1; }
    } else for (size_t i = 0; i < ids.len; ++i) if (!dsm_forward_token(f, ids.ids[i], logits)) { fprintf(stderr, "dsm: prompt inference failed\n"); return 1; }
    double prefill_elapsed = now() - prefill_started;
    if (dump) { FILE *fp = fopen(dump, "wb"); if (!fp || fwrite(logits, sizeof(*logits), model.vocab, fp) != model.vocab) { fprintf(stderr, "dsm: failed to write logits\n"); return 1; } fclose(fp); }
    dsm_sampler sampler;
    dsm_sampler_init(&sampler, temp, top_p, top_k, min_p, seed, 0, 0);
    double decode_started = now();
    int generated = 0;
    for (int i = 0; i < n; ++i) { int32_t id = dsm_sample(&sampler, logits, model.vocab); if ((chat && dsm_token_is_im_end(tok, id)) || !dsm_token_decode(tok, id, stdout) || !(use_metal ? dsm_metal_token(metal, id, logits) : dsm_forward_token(f, id, logits))) break; ++generated; }
    double decode_elapsed = now() - decode_started;
    fprintf(stderr, "\ndsm: %s prefill %.2f tokens/s (%zu tokens, %.3f s); decode %.2f tokens/s (%d tokens, %.3f s)\n", use_metal ? "metal" : "cpu", ids.len / prefill_elapsed, ids.len, prefill_elapsed, generated / decode_elapsed, generated, decode_elapsed);
    dsm_sampler_free(&sampler); free(logits); dsm_forward_free(f); dsm_metal_free(metal); dsm_tokens_free(&ids); dsm_tokenizer_free(tok); dsm_model_close(&model); free(rendered); return 0;
}
