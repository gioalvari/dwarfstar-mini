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
int dsm_metal_prefill(dsm_metal *state, const int32_t *tokens, uint32_t n,
                      float *logits);
uint32_t dsm_metal_pos(const dsm_metal *state);
/* Returns the checkpoint position restored at or before pos, or UINT32_MAX. */
uint32_t dsm_metal_rewind(dsm_metal *state, uint32_t pos);
/* Saves the current hybrid recurrent state after all preceding Metal work. */
int dsm_metal_checkpoint(dsm_metal *state);
int dsm_metal_truncate(dsm_metal *state, uint32_t pos);
int dsm_metal_matvec_test(dsm_metal *state, const gguf_tensor *weights,
                           const float *input, float *output);
int dsm_metal_matmul_test(dsm_metal *state, const gguf_tensor *weights,
                          const float *input, uint32_t rows, float *output);
int dsm_metal_expert_matvec_test(dsm_metal *state, const gguf_tensor *weights,
                                 uint32_t expert, const float *input,
                                 float *output);
int dsm_metal_expert_matmul_test(dsm_metal *state, const gguf_tensor *weights,
                                 const float *input, const uint32_t *ids,
                                 uint32_t rows, uint32_t input_rows,
                                 float *output);

#endif
