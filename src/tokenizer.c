#include "tokenizer.h"

#include <ctype.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct { const char *p; uint32_t len; int32_t id; uint8_t used; } entry;
typedef struct { const char *p; uint32_t len, rank; uint8_t used; } merge;
typedef struct { const char *p; uint32_t len; int32_t id; } special_token;
struct dsm_tokenizer {
    const gguf_file *gguf; gguf_array tokens, types, merges;
    entry *vocab; size_t vocab_cap; merge *merge_table; size_t merge_cap;
    gguf_str *token_strings;
    special_token *specials;
    size_t special_first[257];
    int qwen35;
    int32_t bos, im_start, im_end, eot;
};
static uint64_t hash_bytes(const char *p, size_t n) { uint64_t h = 1469598103934665603ULL; while (n--) { h ^= (uint8_t)*p++; h *= 1099511628211ULL; } return h; }
static size_t pow2(size_t n) { size_t p = 1; while (p < n) p <<= 1; return p; }
static int push(dsm_tokens *v, int32_t id) { if (v->len == v->cap) { size_t n = v->cap ? v->cap * 2 : 32; int32_t *p = realloc(v->ids, n * sizeof(*p)); if (!p) return 0; v->ids = p; v->cap = n; } v->ids[v->len++] = id; return 1; }
void dsm_tokens_free(dsm_tokens *v) { free(v->ids); memset(v, 0, sizeof(*v)); }
static int lookup(const dsm_tokenizer *t, const char *p, size_t n, int32_t *id) {
    size_t i = hash_bytes(p, n) & (t->vocab_cap - 1);
    while (t->vocab[i].used) { if (t->vocab[i].len == n && !memcmp(t->vocab[i].p, p, n)) { *id = t->vocab[i].id; return 1; } i = (i + 1) & (t->vocab_cap - 1); }
    return 0;
}
static int next_string(const gguf_file *g, uint64_t *offset, gguf_str *s) {
    if (*offset > g->size - 8) return 0;
    memcpy(&s->len, g->map + *offset, 8); *offset += 8;
    if (s->len > g->size - *offset) return 0;
    s->ptr = (const char *)g->map + *offset; *offset += s->len; return 1;
}
static uint32_t merge_rank(const dsm_tokenizer *t, const char *p, size_t n) {
    size_t i = hash_bytes(p, n) & (t->merge_cap - 1);
    while (t->merge_table[i].used) { if (t->merge_table[i].len == n && !memcmp(t->merge_table[i].p, p, n)) return t->merge_table[i].rank; i = (i + 1) & (t->merge_cap - 1); }
    return UINT32_MAX;
}
static size_t utf8_decode(const char *s, size_t n, size_t p, uint32_t *cp) {
    if (p >= n) { *cp = UINT32_MAX; return p; }
    unsigned char c = (unsigned char)s[p];
    if (c < 0x80) { *cp = c; return p + 1; }
    size_t width = c < 0xe0 ? 2 : c < 0xf0 ? 3 : c < 0xf8 ? 4 : 1;
    if (width == 1 || p + width > n) { *cp = c; return p + 1; }
    uint32_t value = c & ((1U << (8 - width - 1)) - 1);
    for (size_t i = 1; i < width; ++i) {
        unsigned char continuation = (unsigned char)s[p + i];
        if ((continuation & 0xc0) != 0x80) { *cp = c; return p + 1; }
        value = (value << 6) | (continuation & 0x3f);
    }
    if ((width == 2 && value < 0x80) || (width == 3 && value < 0x800) ||
        (width == 4 && (value < 0x10000 || value > 0x10ffff))) {
        *cp = c; return p + 1;
    }
    *cp = value;
    return p + width;
}
static size_t utf8_next(const char *s, size_t n, size_t p) {
    uint32_t cp;
    return utf8_decode(s, n, p, &cp);
}
static int is_space_cp(uint32_t cp) {
    return cp <= 0x20 || (cp >= 0x2000 && cp <= 0x200a) || cp == 0x2028 ||
           cp == 0x2029 || cp == 0x202f || cp == 0x205f || cp == 0x3000;
}
static int is_letter_cp(uint32_t cp) {
    return (cp >= 'A' && cp <= 'Z') || (cp >= 'a' && cp <= 'z') ||
           (cp >= 0x00c0 && cp <= 0x02af) || (cp >= 0x0370 && cp <= 0x1fff) ||
           (cp >= 0x3040 && cp <= 0x30ff) || (cp >= 0x3100 && cp <= 0x312f) ||
           (cp >= 0x3130 && cp <= 0x318f) || (cp >= 0x31a0 && cp <= 0x31bf) ||
           (cp >= 0x3400 && cp <= 0x4dbf) || (cp >= 0x4e00 && cp <= 0x9fff) ||
           (cp >= 0xac00 && cp <= 0xd7af) || (cp >= 0xf900 && cp <= 0xfaff) ||
           (cp >= 0x10000 && cp <= 0x1fffd) || (cp >= 0x20000 && cp <= 0x2fffd);
}
static int is_mark_cp(uint32_t cp) {
    return (cp >= 0x0300 && cp <= 0x036f) || (cp >= 0x1ab0 && cp <= 0x1aff) ||
           (cp >= 0x1dc0 && cp <= 0x1dff) || (cp >= 0x20d0 && cp <= 0x20ff) ||
           (cp >= 0xfe00 && cp <= 0xfe0f) || (cp >= 0xfe20 && cp <= 0xfe2f);
}
static int is_space_at(const char *s, size_t n, size_t p) {
    if (!n) return 0;
    if ((unsigned char)s[p] < 0x80) return (unsigned char)s[p] <= 0x20;
    uint32_t cp;
    utf8_decode(s, n, p, &cp);
    return is_space_cp(cp);
}
static int is_digit_at(const char *s, size_t n, size_t p) {
    uint32_t cp;
    utf8_decode(s, n, p, &cp);
    return cp >= '0' && cp <= '9';
}
static int is_letter_or_mark_at(const char *s, size_t n, size_t p, int qwen35) {
    if (!qwen35) {
        unsigned char byte = (unsigned char)s[p];
        return isalpha(byte) || (byte >= 0xc0 && byte < 0xf0);
    }
    uint32_t cp;
    utf8_decode(s, n, p, &cp);
    return is_letter_cp(cp) || (qwen35 && is_mark_cp(cp));
}
/* qwen2: contractions | optional prefix plus letters | digits | punctuation | whitespace. */
static size_t qwen_piece(const char *s, size_t n, size_t p, int qwen35) {
    size_t start = p, q = utf8_next(s, n, p);
    uint32_t cp;
    utf8_decode(s, n, p, &cp);
    if (s[p] == '\'' && q < n) { char c = (char)tolower((unsigned char)s[q]); size_t r = utf8_next(s, n, q); if (strchr("stmd", c) || ((c == 'r' || c == 'v' || c == 'l') && r < n && (tolower((unsigned char)s[r]) == 'e' || (c == 'l' && tolower((unsigned char)s[r]) == 'l')))) return c == 'r' || c == 'v' || c == 'l' ? utf8_next(s, n, r) : r; }
    if (is_letter_or_mark_at(s, n, p, qwen35) ||
        (q < n && cp != '\r' && cp != '\n' && !is_digit_at(s, n, p) &&
         is_letter_or_mark_at(s, n, q, qwen35))) {
        q = utf8_next(s, n, p);
        while (q < n && is_letter_or_mark_at(s, n, q, qwen35)) q = utf8_next(s, n, q);
        return q;
    }
    if (is_digit_at(s,n,p)) {
        unsigned count = 1;
        while (q < n && is_digit_at(s, n, q) && (!qwen35 && count++ < 3)) q = utf8_next(s, n, q);
        return q;
    }
    int punctuation = !is_space_at(s, n, p);
    if (cp == ' ' && q < n) {
        punctuation = !is_space_at(s, n, q) && !is_letter_or_mark_at(s, n, q, qwen35) &&
                      !is_digit_at(s, n, q);
    }
    if (punctuation) {
        if (cp == ' ') q = utf8_next(s, n, q);
        while (q < n && !is_space_at(s, n, q) && !is_letter_or_mark_at(s, n, q, qwen35) && !is_digit_at(s, n, q)) q = utf8_next(s,n,q);
        while (q < n && (s[q] == '\r' || s[q] == '\n')) q = utf8_next(s,n,q);
        return q;
    }
    size_t whitespace = 0, last_newline_end = 0;
    for (size_t r = p; r < n && is_space_at(s, n, r); r = utf8_next(s, n, r)) {
        uint32_t whitespace_cp;
        size_t next = utf8_decode(s, n, r, &whitespace_cp);
        if (whitespace_cp == '\r' || whitespace_cp == '\n') last_newline_end = next;
        ++whitespace;
        q = next;
    }
    if (last_newline_end) return last_newline_end;
    if (whitespace > 1 && q < n) {
        size_t last = p;
        for (size_t r = p; r < q; r = utf8_next(s, n, r)) last = r;
        return last;
    }
    return q > start ? q : utf8_next(s,n,p);
}
static int byte_encode(const char *in, size_t n, char **encoded, size_t *encoded_n) {
    char *out = malloc(n * 2 + 1); if (!out) return 0; size_t k = 0;
    for (size_t i = 0; i < n; ++i) {
        unsigned b = (unsigned char)in[i]; unsigned cp = b;
        if (!(b >= 33 && b <= 126) && !(b >= 161 && b <= 172) && !(b >= 174 && b <= 255)) {
            unsigned rank = 0;
            for (unsigned j = 0; j < b; ++j) if (!((j >= 33 && j <= 126) || (j >= 161 && j <= 172) || (j >= 174 && j <= 255))) ++rank;
            cp = 256 + rank;
        }
        if (cp < 128) out[k++] = (char)cp;
        else { out[k++] = (char)(0xc0 | (cp >> 6)); out[k++] = (char)(0x80 | (cp & 63)); }
    }
    out[k] = '\0'; *encoded = out; *encoded_n = k; return 1;
}
typedef struct { char *p; size_t len; int32_t id; } symbol;
static int emit_piece(const dsm_tokenizer *t, const char *raw, size_t raw_n, dsm_tokens *out) {
    char *owned = NULL, *text; size_t n;
    if (!byte_encode(raw, raw_n, &owned, &n)) return 0; text = owned;
    int32_t whole; if (lookup(t, text, n, &whole)) { free(owned); return push(out, whole); }
    symbol *sym = calloc(n + 1, sizeof(*sym)); if (!sym) { free(owned); return 0; } size_t count = 0;
    for (size_t p = 0; p < n; ) { size_t q = utf8_next(text,n,p); sym[count].p = malloc(q-p); if (!sym[count].p) goto fail; memcpy(sym[count].p,text+p,q-p); sym[count].len=q-p; if (!lookup(t,sym[count].p,sym[count].len,&sym[count].id)) { free(sym[count].p); for(size_t j=p;j<q;++j) { if(!lookup(t,text+j,1,&sym[count].id)) goto fail; sym[count].p=malloc(1);if(!sym[count].p)goto fail;sym[count].p[0]=text[j];sym[count].len=1;++count; } p=q; continue; } ++count; p=q; }
    while (count > 1) { uint32_t best = UINT32_MAX; size_t at = count;
        /* GGUF merge entries are byte-token pairs. The component may already
         * have been merged, so rank by its two serialized token strings. */
        for (size_t i = 0; i + 1 < count; ++i) {
            uint32_t r = UINT32_MAX;
            {
                size_t len = sym[i].len + sym[i + 1].len + 1; char *pair = malloc(len);
                if (!pair) goto fail;
                memcpy(pair, sym[i].p, sym[i].len); pair[sym[i].len] = ' ';
                memcpy(pair + sym[i].len + 1, sym[i + 1].p, sym[i + 1].len);
                r = merge_rank(t, pair, len);
                free(pair);
            }
            if (r < best) { best = r; at = i; }
        }
        if (at == count) break; size_t len=sym[at].len+sym[at+1].len; char *joined=malloc(len); if(!joined)goto fail; memcpy(joined,sym[at].p,sym[at].len);memcpy(joined+sym[at].len,sym[at+1].p,sym[at+1].len);int32_t merged;if(!lookup(t,joined,len,&merged)){free(joined);goto fail;}free(sym[at].p);free(sym[at+1].p);sym[at]=(symbol){joined,len,merged};memmove(sym+at+1,sym+at+2,(count-at-2)*sizeof(*sym));--count;
    }
    for (size_t i = 0; i < count; ++i) { if (!push(out, sym[i].id)) goto fail; free(sym[i].p); }
    free(owned); free(sym); return 1;
fail:
    for(size_t i=0;i<count;++i) free(sym[i].p); free(owned); free(sym); return 0;
}
static int special_compare(const void *left, const void *right) {
    const special_token *a = left, *b = right;
    unsigned char af = (unsigned char)a->p[0], bf = (unsigned char)b->p[0];
    if (af != bf) return af < bf ? -1 : 1;
    if (a->len != b->len) return a->len > b->len ? -1 : 1;
    return memcmp(a->p, b->p, a->len);
}
static int find_special(const dsm_tokenizer *t, const char *text, size_t n,
                        size_t p, int32_t *id, size_t *length) {
    unsigned char first = (unsigned char)text[p];
    for (size_t i = t->special_first[first]; i < t->special_first[first + 1]; ++i) {
        const special_token *special = &t->specials[i];
        if (special->len <= n - p && !memcmp(text + p, special->p, special->len)) {
            *id = special->id;
            *length = special->len;
            return 1;
        }
    }
    return 0;
}
int dsm_tokenizer_init(dsm_tokenizer **out, const gguf_file *g, char *error, size_t error_size) {
    *out = NULL; dsm_tokenizer *t = calloc(1, sizeof(*t)); if (!t) goto oom; t->gguf = g;
    if (!gguf_get_array(g,"tokenizer.ggml.tokens",&t->tokens) || t->tokens.type != GGUF_STRING || !gguf_get_array(g,"tokenizer.ggml.token_type",&t->types) || t->types.type != GGUF_INT32 || !gguf_get_array(g,"tokenizer.ggml.merges",&t->merges) || t->merges.type != GGUF_STRING || t->types.len != t->tokens.len || !gguf_get_u32(g,"tokenizer.ggml.bos_token_id",(uint32_t *)&t->bos)) goto invalid;
    gguf_str pre = {0};
    if (gguf_get_string(g, "tokenizer.ggml.pre", &pre)) {
        t->qwen35 = pre.len == 6 && !memcmp(pre.ptr, "qwen35", 6);
        if (!t->qwen35 && !(pre.len == 5 && !memcmp(pre.ptr, "qwen2", 5))) goto invalid;
    }
    t->vocab_cap = pow2(t->tokens.len * 2); t->vocab = calloc(t->vocab_cap,sizeof(*t->vocab)); t->token_strings = calloc(t->tokens.len,sizeof(*t->token_strings)); t->specials = calloc(t->tokens.len,sizeof(*t->specials)); t->merge_cap = pow2(t->merges.len * 2); t->merge_table = calloc(t->merge_cap,sizeof(*t->merge_table)); if (!t->vocab || !t->token_strings || !t->specials || !t->merge_table) goto oom;
    uint64_t offset=t->tokens.data_offset;
    size_t special_count = 0;
    for (uint64_t i=0;i<t->tokens.len;++i) { int32_t kind; gguf_str s; if (!next_string(g,&offset,&s) || !gguf_array_i32_at(g, &t->types, i, &kind)) goto invalid; t->token_strings[i]=s; size_t at=hash_bytes(s.ptr,s.len)&(t->vocab_cap-1); while(t->vocab[at].used) at=(at+1)&(t->vocab_cap-1); t->vocab[at]=(entry){s.ptr,(uint32_t)s.len,(int32_t)i,1}; if ((kind == 3 || (kind == 4 && s.len > 2 && s.ptr[0] == '<' && s.ptr[s.len - 1] == '>')) && s.len) t->specials[special_count++] = (special_token){s.ptr, (uint32_t)s.len, (int32_t)i}; }
    qsort(t->specials, special_count, sizeof(*t->specials), special_compare);
    size_t at = 0;
    for (unsigned first = 0; first < 256; ++first) {
        t->special_first[first] = at;
        while (at < special_count && (unsigned char)t->specials[at].p[0] == first) ++at;
    }
    t->special_first[256] = special_count;
    offset=t->merges.data_offset;
    for (uint64_t i=0;i<t->merges.len;++i) { gguf_str s; if (!next_string(g,&offset,&s) || !memchr(s.ptr,' ',s.len)) goto invalid; size_t at=hash_bytes(s.ptr,(size_t)s.len)&(t->merge_cap-1); while(t->merge_table[at].used) at=(at+1)&(t->merge_cap-1); t->merge_table[at]=(merge){s.ptr,(uint32_t)s.len,(uint32_t)i,1}; }
    t->im_start=t->im_end=t->eot=-1;
    (void)lookup(t,"<|im_start|>",12,&t->im_start); (void)lookup(t,"<|im_end|>",10,&t->im_end); (void)lookup(t,"<|endoftext|>",13,&t->eot);
    *out=t; return 1;
oom: snprintf(error,error_size,"out of memory"); goto bad;
invalid: snprintf(error,error_size,"invalid Qwen2 tokenizer metadata");
bad: dsm_tokenizer_free(t); return 0;
}
void dsm_tokenizer_free(dsm_tokenizer *t) { if (!t) return; free(t->vocab); free(t->token_strings); free(t->specials); free(t->merge_table); free(t); }
int dsm_tokenize(const dsm_tokenizer *t, const char *text, int add_bos, dsm_tokens *out) {
    memset(out,0,sizeof(*out)); if(add_bos && !push(out,t->bos)) return 0; size_t n=strlen(text);
    for(size_t p=0;p<n;) {
        int32_t special = -1; size_t special_n = 0;
        (void)find_special(t, text, n, p, &special, &special_n);
        if(special >= 0) { if(!push(out,special)) { dsm_tokens_free(out); return 0; } p+=special_n; continue; }
        size_t end=qwen_piece(text,n,p,t->qwen35); if(end<=p || !emit_piece(t,text+p,end-p,out)) { dsm_tokens_free(out); return 0; } p=end;
    } return 1;
}
int dsm_token_decode(const dsm_tokenizer *t, int32_t id, FILE *out) {
    gguf_str s; int32_t kind; if(id<0 || (uint64_t)id>=t->tokens.len || !gguf_array_i32_at(t->gguf,&t->types,(uint64_t)id,&kind)) return 0; s=t->token_strings[id];
    if(kind == 3) return fwrite(s.ptr,1,(size_t)s.len,out)==s.len;
    for(size_t p=0;p<s.len;) { unsigned char c=(unsigned char)s.ptr[p++]; unsigned cp=c; if(c>=0xc0 && p<s.len) { unsigned char c2=(unsigned char)s.ptr[p++]; cp=((unsigned)(c&31)<<6)|(c2&63); } unsigned byte=cp; if(cp>=256&&cp<256+68) { unsigned rank=cp-256, seen=0; for(unsigned j=0;j<256;++j) if(!((j>=33&&j<=126)||(j>=161&&j<=172)||(j>=174&&j<=255)) && seen++==rank) { byte=j; break; } } if(fputc((int)byte,out)==EOF) return 0; } return 1;
}
int32_t dsm_token_im_start(const dsm_tokenizer *t) { return t ? t->im_start : -1; }
int dsm_token_is_im_end(const dsm_tokenizer *t, int32_t id) { return id == t->im_end; }
