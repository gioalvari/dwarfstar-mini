#ifndef DSM_METAL_H
#define DSM_METAL_H

#include "model.h"

#include <stdint.h>

typedef struct dsm_metal dsm_metal;

int dsm_metal_available(void);
int dsm_metal_init(dsm_metal **out, const dsm_model *model, uint32_t context,
                   char *error, uint32_t error_size);
void dsm_metal_free(dsm_metal *state);
int dsm_metal_token(dsm_metal *state, int32_t token, float *logits);
uint32_t dsm_metal_pos(const dsm_metal *state);
int dsm_metal_truncate(dsm_metal *state, uint32_t pos);
int dsm_metal_matvec_test(dsm_metal *state, const gguf_tensor *weights,
                          const float *input, float *output);

#endif
