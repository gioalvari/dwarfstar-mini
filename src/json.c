#include "json.h"

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct { const char *p, *end; char *error; size_t error_size; } parser;

static void fail(parser *p, const char *message) {
    if (!p->error[0]) snprintf(p->error, p->error_size, "%s", message);
}
static void space(parser *p) { while (p->p < p->end && isspace((unsigned char)*p->p)) ++p->p; }
static dsm_json *value(parser *p);
static dsm_json *new_value(dsm_json_type type) {
    dsm_json *v = calloc(1, sizeof(*v)); if (v) v->type = type; return v;
}
static int append(dsm_json *v, dsm_json *item) {
    dsm_json **items = realloc(v->items, (v->count + 1) * sizeof(*items));
    if (!items) return 0;
    v->items = items; v->items[v->count++] = item; return 1;
}
static int string(parser *p, dsm_json *v) {
    if (p->p == p->end || *p->p++ != '"') return 0;
    v->text = p->p;
    while (p->p < p->end) {
        unsigned char c = (unsigned char)*p->p++;
        if (c == '"') { v->len = (size_t)((p->p - 1) - v->text); return 1; }
        if (c < 0x20) break;
        if (c == '\\') {
            if (p->p == p->end || !strchr("\"\\/bfnrtu", *p->p++)) break;
            if (p->p[-1] == 'u') for (int i = 0; i < 4; ++i)
                if (p->p == p->end || !isxdigit((unsigned char)*p->p++)) goto bad;
        }
    }
bad: fail(p, "invalid JSON string"); return 0;
}
static dsm_json *array(parser *p) {
    dsm_json *v = new_value(DSM_JSON_ARRAY); if (!v) return NULL; ++p->p; space(p);
    if (p->p < p->end && *p->p == ']') { ++p->p; return v; }
    for (;;) {
        dsm_json *item = value(p);
        if (!item || !append(v, item)) { dsm_json_free(item); dsm_json_free(v); fail(p, "out of memory"); return NULL; }
        space(p);
        if (p->p < p->end && *p->p == ']') { ++p->p; return v; }
        if (p->p == p->end || *p->p++ != ',') { dsm_json_free(v); fail(p, "expected ',' or ']'"); return NULL; }
        space(p);
    }
}
static dsm_json *object(parser *p) {
    dsm_json *v = new_value(DSM_JSON_OBJECT); if (!v) return NULL; ++p->p; space(p);
    if (p->p < p->end && *p->p == '}') { ++p->p; return v; }
    for (;;) {
        dsm_json *key = new_value(DSM_JSON_STRING);
        if (!key || !string(p, key)) { dsm_json_free(key); dsm_json_free(v); return NULL; }
        space(p);
        if (p->p == p->end || *p->p++ != ':') { dsm_json_free(key); dsm_json_free(v); fail(p, "expected ':'"); return NULL; }
        space(p); dsm_json *item = value(p);
        if (!item || !append(v, key) || !append(v, item)) { dsm_json_free(key); dsm_json_free(item); dsm_json_free(v); fail(p, "out of memory"); return NULL; }
        space(p);
        if (p->p < p->end && *p->p == '}') { ++p->p; return v; }
        if (p->p == p->end || *p->p++ != ',') { dsm_json_free(v); fail(p, "expected ',' or '}'"); return NULL; }
        space(p);
    }
}
static dsm_json *value(parser *p) {
    space(p); if (p->p == p->end) { fail(p, "unexpected end of JSON"); return NULL; }
    if (*p->p == '"') { dsm_json *v = new_value(DSM_JSON_STRING); if (!v || !string(p, v)) { dsm_json_free(v); return NULL; } return v; }
    if (*p->p == '[') return array(p); if (*p->p == '{') return object(p);
    if (p->end - p->p >= 4 && !memcmp(p->p, "true", 4)) { p->p += 4; dsm_json *v = new_value(DSM_JSON_BOOL); if(v)v->boolean=1; return v; }
    if (p->end - p->p >= 5 && !memcmp(p->p, "false", 5)) { p->p += 5; return new_value(DSM_JSON_BOOL); }
    if (p->end - p->p >= 4 && !memcmp(p->p, "null", 4)) { p->p += 4; return new_value(DSM_JSON_NULL); }
    char *end; dsm_json *v = new_value(DSM_JSON_NUMBER); if (!v) return NULL;
    v->number = strtod(p->p, &end); if (end == p->p || end > p->end) { dsm_json_free(v); fail(p, "invalid JSON value"); return NULL; }
    p->p = end; return v;
}
int dsm_json_parse(const char *text, size_t len, dsm_json **out, char *error, size_t error_size) {
    parser p = {text, text + len, error, error_size}; *out = NULL; if (error_size) error[0] = '\0';
    dsm_json *v = value(&p); space(&p); if (!v || p.p != p.end) { if (!error[0]) fail(&p, "trailing JSON data"); dsm_json_free(v); return 0; }
    *out = v; return 1;
}
void dsm_json_free(dsm_json *v) { if (!v) return; for (size_t i = 0; i < v->count; ++i) dsm_json_free(v->items[i]); free(v->items); free(v); }
const dsm_json *dsm_json_object_get(const dsm_json *object, const char *key) {
    if (!object || object->type != DSM_JSON_OBJECT) return NULL; size_t n = strlen(key);
    for (size_t i = 0; i + 1 < object->count; i += 2) if (object->items[i]->len == n && !memcmp(object->items[i]->text, key, n)) return object->items[i + 1];
    return NULL;
}
char *dsm_json_string_dup(const dsm_json *v) {
    if (!v || v->type != DSM_JSON_STRING) return NULL; char *out = malloc(v->len + 1); if (!out) return NULL;
    size_t j = 0; for (size_t i = 0; i < v->len; ++i) { char c = v->text[i]; if (c != '\\') { out[j++] = c; continue; }
        if (++i == v->len) goto bad; c = v->text[i]; if (c == 'b') out[j++]='\b'; else if(c=='f')out[j++]='\f'; else if(c=='n')out[j++]='\n'; else if(c=='r')out[j++]='\r'; else if(c=='t')out[j++]='\t'; else if(c=='"'||c=='\\'||c=='/')out[j++]=c; else if(c=='u') { if(i+4>=v->len)goto bad; unsigned x=0; for(int k=0;k<4;++k){char h=v->text[++i];x=x*16+(h>='0'&&h<='9'?h-'0':h>='a'&&h<='f'?h-'a'+10:h-'A'+10);} if(x<0x80)out[j++]=(char)x; else if(x<0x800){out[j++]=(char)(0xc0|(x>>6));out[j++]=(char)(0x80|(x&63));} else {out[j++]=(char)(0xe0|(x>>12));out[j++]=(char)(0x80|((x>>6)&63));out[j++]=(char)(0x80|(x&63));}} else goto bad;
    } out[j]='\0'; return out;
bad: free(out); return NULL;
}
