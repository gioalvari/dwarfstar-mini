#include "quants.h"
#include "gguf.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

static void put16(uint8_t *at, uint16_t value) { memcpy(at, &value, sizeof(value)); }
static int close(float a, float b) { return fabsf(a - b) < 1e-6f; }
int main(void) {
    uint8_t q2[84] = {0}, q4[144] = {0}, q6[210] = {0}; float values[256], input[256];
    for (unsigned i = 0; i < 256; ++i) input[i] = (float)(i % 5 - 2);
    q2[0] = 0x11; q2[16] = 0xe4; put16(q2 + 80, 0x3c00); put16(q2 + 82, 0x3800); dsm_dequant_row(GGML_TYPE_Q2_K, q2, values, 256);
    if (!close(values[0], -0.5f) || !close(values[1], -0.5f)) { fprintf(stderr, "q2 %.6g %.6g\n", values[0], values[1]); return 1; }
    put16(q4, 0x3c00); put16(q4 + 2, 0x3800); q4[4] = 2; q4[8] = 3; q4[16] = 0x21; dsm_dequant_row(GGML_TYPE_Q4_K, q4, values, 256);
    if (!close(values[0], 0.5f) || !close(values[32], 0.0f)) { fprintf(stderr, "q4 %.6g %.6g\n", values[0], values[32]); return 1; }
    put16(q6 + 208, 0x3c00); q6[192] = 2; q6[0] = 1; q6[128] = 1; dsm_dequant_row(GGML_TYPE_Q6_K, q6, values, 256);
    if (!close(values[0], -30.0f)) { fprintf(stderr, "q6 %.6g\n", values[0]); return 1; }
    float dot = 0; for (unsigned i = 0; i < 256; ++i) dot += values[i] * input[i];
    if (!close(dot, dsm_dot_row(GGML_TYPE_Q6_K, q6, input, 256))) { fprintf(stderr, "dot %.6g %.6g\n", dot, dsm_dot_row(GGML_TYPE_Q6_K, q6, input, 256)); return 1; }
    puts("quant tests: ok"); return 0;
}
