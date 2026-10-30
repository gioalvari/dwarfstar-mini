#ifndef DSM_TOKENIZER_H
#define DSM_TOKENIZER_H

#include "gguf.h"

#include <stdio.h>

typedef struct { int32_t *ids; size_t len, cap; } dsm_tokens;
typedef struct dsm_tokenizer dsm_tokenizer;

int dsm_tokenizer_init(dsm_tokenizer **out, const gguf_file *gguf, char *error, size_t error_size);
void dsm_tokenizer_free(dsm_tokenizer *tokenizer);
int dsm_tokenize(const dsm_tokenizer *tokenizer, const char *text, int add_bos,
                 dsm_tokens *tokens);
int dsm_token_decode(const dsm_tokenizer *tokenizer, int32_t id, FILE *out);
void dsm_tokens_free(dsm_tokens *tokens);

#endif
