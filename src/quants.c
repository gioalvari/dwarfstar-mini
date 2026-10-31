#include "quants.h"

#include "gguf.h"

#include <assert.h>
#include <string.h>

float dsm_f16_to_f32(uint16_t value) {
    union { uint32_t u; float f; } out;
    uint32_t sign = (uint32_t)(value & 0x8000) << 16;
    uint32_t exponent = (value >> 10) & 31;
    uint32_t fraction = value & 1023;
    if (exponent == 0) {
        if (fraction == 0) { out.u = sign; return out.f; }
        while (!(fraction & 1024)) { fraction <<= 1; --exponent; }
        fraction &= 1023; out.u = sign | ((exponent + 113) << 23) | (fraction << 13);
    } else if (exponent == 31) out.u = sign | 0x7f800000 | (fraction << 13);
    else out.u = sign | ((exponent + 112) << 23) | (fraction << 13);
    return out.f;
}

static uint16_t get_u16(const uint8_t *input) { uint16_t out; memcpy(&out, input, sizeof(out)); return out; }
static void get_scale_min_k4(unsigned index, const uint8_t *scales, uint8_t *scale, uint8_t *min) {
    if (index < 4) { *scale = scales[index] & 63; *min = scales[index + 4] & 63; }
    else { *scale = (scales[index + 4] & 15) | ((scales[index - 4] >> 6) << 4); *min = (scales[index + 4] >> 4) | ((scales[index] >> 6) << 4); }
}

int dsm_quant_supported(uint32_t type) {
    return type == GGML_TYPE_F32 || type == GGML_TYPE_Q8_0 ||
           type == GGML_TYPE_Q2_K || type == GGML_TYPE_Q3_K ||
           type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q5_K || type == GGML_TYPE_Q6_K;
}

static void dequant_q8_0(const uint8_t *input, float *output, uint64_t count) {
    for (uint64_t block = 0; block < count / 32; ++block) {
        const uint8_t *src = input + block * 34; float d = dsm_f16_to_f32(get_u16(src));
        for (unsigned i = 0; i < 32; ++i) output[block * 32 + i] = d * (int8_t)src[2 + i];
    }
}
static void dequant_q2_k(const uint8_t *input, float *output, uint64_t count) {
    for (uint64_t block = 0; block < count / 256; ++block) {
        const uint8_t *src = input + block * 84, *q = src + 16; float d = dsm_f16_to_f32(get_u16(src + 80)), min = dsm_f16_to_f32(get_u16(src + 82));
        unsigned out = 0, scale = 0;
        for (unsigned n = 0; n < 256; n += 128, q += 32) for (unsigned shift = 0; shift < 8; shift += 2) {
            uint8_t sc = src[scale++]; float dl = d * (sc & 15), ml = min * (sc >> 4);
            for (unsigned l = 0; l < 16; ++l) output[block * 256 + out++] = dl * ((q[l] >> shift) & 3) - ml;
            sc = src[scale++]; dl = d * (sc & 15); ml = min * (sc >> 4);
            for (unsigned l = 0; l < 16; ++l) output[block * 256 + out++] = dl * ((q[l + 16] >> shift) & 3) - ml;
        }
    }
}
static void dequant_q3_k(const uint8_t *input, float *output, uint64_t count) {
    for (uint64_t block = 0; block < count / 256; ++block) {
        const uint8_t *src = input + block * 110, *hm = src, *q = src + 32; uint32_t aux[4] = {0}; memcpy(aux, src + 96, 12);
        const int8_t *scales = (const int8_t *)aux; const uint32_t mask1 = 0x03030303, mask2 = 0x0f0f0f0f; uint32_t tmp = aux[2];
        aux[2] = ((aux[0] >> 4) & mask2) | (((tmp >> 4) & mask1) << 4); aux[3] = ((aux[1] >> 4) & mask2) | (((tmp >> 6) & mask1) << 4);
        aux[0] = (aux[0] & mask2) | (((tmp >> 0) & mask1) << 4); aux[1] = (aux[1] & mask2) | (((tmp >> 2) & mask1) << 4);
        float d = dsm_f16_to_f32(get_u16(src + 108)); unsigned out = 0, scale = 0; uint8_t m = 1;
        for (unsigned n = 0; n < 256; n += 128, q += 32) for (unsigned shift = 0; shift < 8; shift += 2, m <<= 1) {
            float dl = d * (scales[scale++] - 32); for (unsigned l = 0; l < 16; ++l) output[block * 256 + out++] = dl * (((q[l] >> shift) & 3) - ((hm[l] & m) ? 0 : 4));
            dl = d * (scales[scale++] - 32); for (unsigned l = 0; l < 16; ++l) output[block * 256 + out++] = dl * (((q[l + 16] >> shift) & 3) - ((hm[l + 16] & m) ? 0 : 4));
        }
    }
}
static void dequant_q4_or_q5_k(const uint8_t *input, float *output, uint64_t count, int q5) {
    uint64_t bytes = q5 ? 176 : 144;
    for (uint64_t block = 0; block < count / 256; ++block) {
        const uint8_t *src = input + block * bytes, *scales = src + 4, *qh = q5 ? src + 16 : NULL, *q = src + (q5 ? 48 : 16);
        float d = dsm_f16_to_f32(get_u16(src)), min = dsm_f16_to_f32(get_u16(src + 2)); unsigned out = 0, index = 0; uint8_t u1 = 1, u2 = 2;
        for (unsigned n = 0; n < 256; n += 64, q += 32, index += 2, u1 <<= 2, u2 <<= 2) {
            uint8_t sc, m; get_scale_min_k4(index, scales, &sc, &m); float d1 = d * sc, m1 = min * m; get_scale_min_k4(index + 1, scales, &sc, &m); float d2 = d * sc, m2 = min * m;
            for (unsigned l = 0; l < 32; ++l) output[block * 256 + out++] = d1 * ((q[l] & 15) + (q5 && (qh[l] & u1) ? 16 : 0)) - m1;
            for (unsigned l = 0; l < 32; ++l) output[block * 256 + out++] = d2 * ((q[l] >> 4) + (q5 && (qh[l] & u2) ? 16 : 0)) - m2;
        }
    }
}
static void dequant_q6_k(const uint8_t *input, float *output, uint64_t count) {
    for (uint64_t block = 0; block < count / 256; ++block) {
        const uint8_t *src = input + block * 210, *ql = src, *qh = src + 128; const int8_t *sc = (const int8_t *)(src + 192); float d = dsm_f16_to_f32(get_u16(src + 208));
        for (unsigned n = 0; n < 256; n += 128, ql += 64, qh += 32, sc += 8) for (unsigned l = 0; l < 32; ++l) {
            unsigned index = block * 256 + n + l, is = l / 16;
            int8_t q1 = ((ql[l] & 15) | (((qh[l] >> 0) & 3) << 4)) - 32, q2 = ((ql[l + 32] & 15) | (((qh[l] >> 2) & 3) << 4)) - 32;
            int8_t q3 = ((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32, q4 = ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
            output[index] = d * sc[is] * q1; output[index + 32] = d * sc[is + 2] * q2; output[index + 64] = d * sc[is + 4] * q3; output[index + 96] = d * sc[is + 6] * q4;
        }
    }
}

void dsm_dequant_row(uint32_t type, const void *input, float *output, uint64_t count) {
    const uint8_t *src = input; assert(dsm_quant_supported(type));
    if (type == GGML_TYPE_F32) { memcpy(output, input, count * sizeof(*output)); return; }
    if (type == GGML_TYPE_Q8_0) dequant_q8_0(src, output, count);
    else if (type == GGML_TYPE_Q2_K) dequant_q2_k(src, output, count);
    else if (type == GGML_TYPE_Q3_K) dequant_q3_k(src, output, count);
    else if (type == GGML_TYPE_Q4_K) dequant_q4_or_q5_k(src, output, count, 0);
    else if (type == GGML_TYPE_Q5_K) dequant_q4_or_q5_k(src, output, count, 1);
    else dequant_q6_k(src, output, count);
}
float dsm_dot_row(uint32_t type, const void *input, const float *vector, uint64_t count) {
    float values[256], sum = 0; const ggml_type_info *info = gguf_type_info(type); uint64_t block = info->block_elems;
    for (uint64_t i = 0; i < count; i += block) { dsm_dequant_row(type, (const uint8_t *)input + i / block * info->block_bytes, values, block); for (uint64_t j = 0; j < block; ++j) sum += values[j] * vector[i + j]; }
    return sum;
}
