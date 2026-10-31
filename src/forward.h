#ifndef DSM_FORWARD_H
#define DSM_FORWARD_H

#include "model.h"

typedef struct dsm_forward dsm_forward;
int dsm_forward_init(dsm_forward **out, const dsm_model *model, uint32_t context,
                     uint32_t threads);
void dsm_forward_free(dsm_forward *state);
int dsm_forward_token(dsm_forward *state, int32_t token, float *logits);
uint32_t dsm_forward_pos(const dsm_forward *state);
/* Discard cached tokens at and after pos. Cached tokens before pos remain valid. */
int dsm_forward_truncate(dsm_forward *state, uint32_t pos);

#endif
