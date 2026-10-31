#ifndef DSM_QUANTS_H
#define DSM_QUANTS_H

#include <stdint.h>

float dsm_f16_to_f32(uint16_t value);
int dsm_quant_supported(uint32_t type);
void dsm_dequant_row(uint32_t type, const void *input, float *output, uint64_t count);
float dsm_dot_row(uint32_t type, const void *input, const float *vector,
                  uint64_t count);

#endif
