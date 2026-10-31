#include <metal_stdlib>
using namespace metal;

struct ggml_metal_kargs_mul_mv {
    int ne00, ne01, ne02;
    ulong nb00, nb01, nb02, nb03;
    int ne10, ne11, ne12;
    ulong nb10, nb11, nb12, nb13;
    int ne0, ne1, nr0;
    short r2, r3;
};
struct Gemv { uint in, out; };
struct Norm { uint width, groups; float eps; };
struct Rope { uint heads, dim, pos; float base; };
struct Cache { uint layer, pos, context, width; };
struct Attention { uint heads, kv_heads, dim, pos, context, layer; };

// Ported from llama.cpp ggml-metal/kernels/mul_mv.metal (MIT license).
struct block_q8_0 { half d; char qs[32]; };

template<short NR0>
static inline void helper_mv_reduce_and_write(
        device float *dst_f32, float sumf[NR0], const int r0, const int ne01,
        ushort tiisg, ushort sgitg, threadgroup char *shmem) {
    constexpr short NW = 32;
    threadgroup float *shmem_f32[NR0];
    for (short row = 0; row < NR0; ++row) {
        shmem_f32[row] = (threadgroup float *)shmem + NW * row;
        if (sgitg == 0) shmem_f32[row][tiisg] = 0.0f;
        sumf[row] = simd_sum(sumf[row]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (short row = 0; row < NR0; ++row) {
        if (tiisg == 0) shmem_f32[row][sgitg] = sumf[row];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (short row = 0; row < NR0 && r0 + row < ne01; ++row) {
        float tot = simd_sum(shmem_f32[row][tiisg]);
        if (tiisg == 0 && sgitg == 0) dst_f32[r0 + row] = tot;
    }
}

kernel void q8_gemv(
        constant ggml_metal_kargs_mul_mv &args [[buffer(3)]],
        device const char *src0 [[buffer(0)]], device const char *src1 [[buffer(1)]],
        device char *dst [[buffer(2)]], threadgroup char *shmem [[threadgroup(0)]],
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    constexpr short NR0 = 2;
    constexpr short NSG = 4;
    constexpr short NW = 32;
    constexpr short NQ = 8;
    const int nb = args.ne00 / 32;
    const int r0 = tgpig.x * NR0;
    const int r1 = tgpig.y;
    const int im = tgpig.z;
    const uint i12 = im;
    const uint i13 = 0;
    const uint64_t offset1 = r1 * args.nb11 + i12 * args.nb12 + i13 * args.nb13;
    device const float *y = (device const float *)(src1 + offset1);
    device const block_q8_0 *ax[NR0];
    for (short row = 0; row < NR0; ++row) {
        const uint64_t offset0 = (r0 + row) * args.nb01 + i12 * args.nb02 + i13 * args.nb03;
        ax[row] = (device const block_q8_0 *)(src0 + offset0);
    }
    float sumf[NR0] = {0.f};
    const short ix = tiisg / (NW / NQ);
    const short il = tiisg % (NW / NQ);
    const int ib0 = sgitg * NQ + ix;
    float yl[NQ];
    device const float *yb = y + ib0 * 32 + il * NQ;
    for (int ib = ib0; ib < nb; ib += NSG * NQ) {
        for (short i = 0; i < NQ; ++i) yl[i] = yb[i];
        for (short row = 0; row < NR0; ++row) {
            device const char *qs = ax[row][ib].qs + il * NQ;
            float sumq = 0.f;
            for (short i = 0; i < NQ; ++i) sumq += qs[i] * yl[i];
            sumf[row] += sumq * ax[row][ib].d;
        }
        yb += NSG * NQ * 32;
    }
    device float *dst_f32 = (device float *)dst + (uint64_t)im * args.ne0 * args.ne1 + (uint64_t)r1 * args.ne0;
    helper_mv_reduce_and_write<NR0>(dst_f32, sumf, r0, args.ne01, tiisg, sgitg, shmem);
}

struct block_q4_K { half d; half dmin; uchar scales[12]; uchar qs[128]; };

kernel void q4k_gemv(
        constant ggml_metal_kargs_mul_mv &args [[buffer(3)]],
        device const char *src0 [[buffer(0)]], device const char *src1 [[buffer(1)]],
        device char *dst [[buffer(2)]], uint3 tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    constexpr short nr0 = 2;
    constexpr short NSG = 2;
    constexpr uint16_t kmask1 = 0x3f3f;
    constexpr uint16_t kmask2 = 0x0f0f;
    constexpr uint16_t kmask3 = 0xc0c0;
    const short ix = tiisg / 8;
    const short it = tiisg % 8;
    const short iq = it / 4;
    const short ir = it % 4;
    const int nb = args.ne00 / 256;
    const int r0 = tgpig.x;
    const int r1 = tgpig.y;
    const int im = tgpig.z;
    const int first_row = (r0 * NSG + sgitg) * nr0;
    const uint i12 = im;
    const uint i13 = 0;
    const uint64_t offset0 = first_row * args.nb01 + i12 * args.nb02 + i13 * args.nb03;
    const uint64_t offset1 = r1 * args.nb11 + i12 * args.nb12 + i13 * args.nb13;
    device const block_q4_K *x = (device const block_q4_K *)(src0 + offset0);
    device const float *y = (device const float *)(src1 + offset1);
    float yl[16];
    float yh[16];
    float sumf[nr0] = {0.f};
    device const float *y4 = y + ix * 256 + 64 * iq + 8 * ir;
    uint16_t sc16[4];
    thread const uint8_t *sc8 = (thread const uint8_t *)sc16;
    for (int ib = ix; ib < nb; ib += 4) {
        float4 sumy = {0.f, 0.f, 0.f, 0.f};
        for (short i = 0; i < 8; ++i) {
            yl[i + 0] = y4[i + 0]; sumy[0] += yl[i + 0];
            yl[i + 8] = y4[i + 32]; sumy[1] += yl[i + 8];
            yh[i + 0] = y4[i + 128]; sumy[2] += yh[i + 0];
            yh[i + 8] = y4[i + 160]; sumy[3] += yh[i + 8];
        }
        device const uint16_t *sc = (device const uint16_t *)x[ib].scales + iq;
        device const uint16_t *q1 = (device const uint16_t *)x[ib].qs + 16 * iq + 4 * ir;
        device const half *dh = &x[ib].d;
        for (short row = 0; row < nr0; row++) {
            sc16[0] = sc[0] & kmask1;
            sc16[1] = sc[2] & kmask1;
            sc16[2] = ((sc[4] >> 0) & kmask2) | ((sc[0] & kmask3) >> 2);
            sc16[3] = ((sc[4] >> 4) & kmask2) | ((sc[2] & kmask3) >> 2);
            device const uint16_t *q2 = q1 + 32;
            float4 acc1 = {0.f, 0.f, 0.f, 0.f};
            float4 acc2 = {0.f, 0.f, 0.f, 0.f};
            for (short i = 0; i < 4; ++i) {
                acc1[0] += yl[2 * i + 0] * (q1[i] & 0x000F);
                acc1[1] += yl[2 * i + 1] * (q1[i] & 0x0F00);
                acc1[2] += yl[2 * i + 8] * (q1[i] & 0x00F0);
                acc1[3] += yl[2 * i + 9] * (q1[i] & 0xF000);
                acc2[0] += yh[2 * i + 0] * (q2[i] & 0x000F);
                acc2[1] += yh[2 * i + 1] * (q2[i] & 0x0F00);
                acc2[2] += yh[2 * i + 8] * (q2[i] & 0x00F0);
                acc2[3] += yh[2 * i + 9] * (q2[i] & 0xF000);
            }
            sumf[row] += dh[0] * ((acc1[0] + 1.f / 256.f * acc1[1]) * sc8[0] +
                                  (acc1[2] + 1.f / 256.f * acc1[3]) * sc8[1] / 16.f +
                                  (acc2[0] + 1.f / 256.f * acc2[1]) * sc8[4] +
                                  (acc2[2] + 1.f / 256.f * acc2[3]) * sc8[5] / 16.f) -
                         dh[1] * (sumy[0] * sc8[2] + sumy[1] * sc8[3] + sumy[2] * sc8[6] + sumy[3] * sc8[7]);
            q1 += args.nb01 / 2;
            sc += args.nb01 / 2;
            dh += args.nb01 / 2;
        }
        y4 += 4 * 256;
    }
    device float *dst_f32 = (device float *)dst + (uint64_t)im * args.ne0 * args.ne1 + (uint64_t)r1 * args.ne0;
    for (int row = 0; row < nr0 && first_row + row < args.ne0; ++row) {
        float sum_all = simd_sum(sumf[row]);
        if (tiisg == 0) dst_f32[first_row + row] = sum_all;
    }
}

struct block_q6_K { uchar ql[128]; uchar qh[64]; char scales[16]; half d; };

kernel void q6k_gemv(
        constant ggml_metal_kargs_mul_mv &args [[buffer(3)]],
        device const char *src0 [[buffer(0)]], device const char *src1 [[buffer(1)]],
        device char *dst [[buffer(2)]], uint3 tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    constexpr short nr0 = 2;
    constexpr short NSG = 2;
    constexpr uchar kmask1 = 0x03;
    constexpr uchar kmask2 = 0x0C;
    constexpr uchar kmask3 = 0x30;
    constexpr uchar kmask4 = 0xC0;
    const int nb = args.ne00 / 256;
    const int r0 = tgpig.x;
    const int r1 = tgpig.y;
    const int im = tgpig.z;
    const int first_row = (r0 * NSG + sgitg) * nr0;
    const uint i12 = im;
    const uint i13 = 0;
    const uint64_t offset0 = first_row * args.nb01 + i12 * args.nb02 + i13 * args.nb03;
    const uint64_t offset1 = r1 * args.nb11 + i12 * args.nb12 + i13 * args.nb13;
    device const block_q6_K *x = (device const block_q6_K *)(src0 + offset0);
    device const float *yy = (device const float *)(src1 + offset1);
    float sumf[nr0] = {0.f};
    float yl[16];
    const short tid = tiisg / 2;
    const short ix = tiisg % 2;
    const short ip = tid / 8;
    const short il = tid % 8;
    const short l0 = 4 * il;
    const short is = 8 * ip + l0 / 16;
    const short y_offset = 128 * ip + l0;
    const short q_offset_l = 64 * ip + l0;
    const short q_offset_h = 32 * ip + l0;
    for (int i = ix; i < nb; i += 2) {
        device const uchar *q1 = x[i].ql + q_offset_l;
        device const uchar *q2 = q1 + 32;
        device const uchar *qh = x[i].qh + q_offset_h;
        device const char *sc = x[i].scales + is;
        device const half *dh = &x[i].d;
        device const float *y = yy + i * 256 + y_offset;
        for (short l = 0; l < 4; ++l) {
            yl[4 * l + 0] = y[l + 0];
            yl[4 * l + 1] = y[l + 32];
            yl[4 * l + 2] = y[l + 64];
            yl[4 * l + 3] = y[l + 96];
        }
        for (short row = 0; row < nr0; ++row) {
            float4 sums = {0.f, 0.f, 0.f, 0.f};
            for (short l = 0; l < 4; ++l) {
                sums[0] += yl[4 * l + 0] * ((char)((q1[l] & 0xF) | ((qh[l] & kmask1) << 4)) - 32);
                sums[1] += yl[4 * l + 1] * ((char)((q2[l] & 0xF) | ((qh[l] & kmask2) << 2)) - 32);
                sums[2] += yl[4 * l + 2] * ((char)((q1[l] >> 4) | ((qh[l] & kmask3) << 0)) - 32);
                sums[3] += yl[4 * l + 3] * ((char)((q2[l] >> 4) | ((qh[l] & kmask4) >> 2)) - 32);
            }
            sumf[row] += dh[0] * (sums[0] * sc[0] + sums[1] * sc[2] + sums[2] * sc[4] + sums[3] * sc[6]);
            q1 += args.nb01;
            q2 += args.nb01;
            qh += args.nb01;
            sc += args.nb01;
            dh += args.nb01 / 2;
        }
    }
    device float *dst_f32 = (device float *)dst + (uint64_t)im * args.ne0 * args.ne1 + (uint64_t)r1 * args.ne0;
    for (int row = 0; row < nr0 && first_row + row < args.ne0; ++row) {
        float sum_all = simd_sum(sumf[row]);
        if (tiisg == 0) dst_f32[first_row + row] = sum_all;
    }
}

struct block_q2_K { uchar scales[16]; uchar qs[64]; half d; half dmin; };

kernel void q2k_gemv(
        constant ggml_metal_kargs_mul_mv &args [[buffer(3)]],
        device const char *src0 [[buffer(0)]], device const char *src1 [[buffer(1)]],
        device char *dst [[buffer(2)]], uint3 tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    constexpr short nr0 = 4;
    constexpr short NSG = 2;
    const int nb = args.ne00 / 256;
    const int r0 = tgpig.x;
    const int r1 = tgpig.y;
    const int im = tgpig.z;
    const int first_row = (r0 * NSG + sgitg) * nr0;
    const uint i12 = im;
    const uint64_t offset0 = first_row * args.nb01 + i12 * args.nb02;
    const uint64_t offset1 = r1 * args.nb11 + i12 * args.nb12;
    device const block_q2_K *x = (device const block_q2_K *)(src0 + offset0);
    device const float *y = (device const float *)(src1 + offset1);
    float yl[32];
    float sumf[nr0] = {0.f};
    const short ix = tiisg / 8;
    const short it = tiisg % 8;
    const short iq = it / 4;
    const short ir = it % 4;
    const short is = (8 * ir) / 16;
    device const float *y4 = y + ix * 256 + 128 * iq + 8 * ir;
    for (int ib = ix; ib < nb; ib += 4) {
        float4 sumy = {0.f, 0.f, 0.f, 0.f};
        for (short i = 0; i < 8; ++i) {
            yl[i + 0] = y4[i + 0]; sumy[0] += yl[i + 0];
            yl[i + 8] = y4[i + 32]; sumy[1] += yl[i + 8];
            yl[i + 16] = y4[i + 64]; sumy[2] += yl[i + 16];
            yl[i + 24] = y4[i + 96]; sumy[3] += yl[i + 24];
        }
        device const uchar *sc = x[ib].scales + 8 * iq + is;
        device const uint16_t *qs = (device const uint16_t *)x[ib].qs + 16 * iq + 4 * ir;
        device const half *dh = &x[ib].d;
        for (short row = 0; row < nr0; row++) {
            float4 acc1 = {0.f, 0.f, 0.f, 0.f};
            float4 acc2 = {0.f, 0.f, 0.f, 0.f};
            for (int i = 0; i < 8; i += 2) {
                acc1[0] += yl[i + 0] * (qs[i / 2] & 0x0003);
                acc2[0] += yl[i + 1] * (qs[i / 2] & 0x0300);
                acc1[1] += yl[i + 8] * (qs[i / 2] & 0x000c);
                acc2[1] += yl[i + 9] * (qs[i / 2] & 0x0c00);
                acc1[2] += yl[i + 16] * (qs[i / 2] & 0x0030);
                acc2[2] += yl[i + 17] * (qs[i / 2] & 0x3000);
                acc1[3] += yl[i + 24] * (qs[i / 2] & 0x00c0);
                acc2[3] += yl[i + 25] * (qs[i / 2] & 0xc000);
            }
            float dall = dh[0];
            float dmin = dh[1] / 16.f;
            sumf[row] += dall * ((acc1[0] + acc2[0] / 256.f) * (sc[0] & 0xF) +
                                 (acc1[1] + acc2[1] / 256.f) * (sc[2] & 0xF) / 4.f +
                                 (acc1[2] + acc2[2] / 256.f) * (sc[4] & 0xF) / 16.f +
                                 (acc1[3] + acc2[3] / 256.f) * (sc[6] & 0xF) / 64.f) -
                         dmin * (sumy[0] * (sc[0] & 0xF0) + sumy[1] * (sc[2] & 0xF0) +
                                 sumy[2] * (sc[4] & 0xF0) + sumy[3] * (sc[6] & 0xF0));
            qs += args.nb01 / 2;
            sc += args.nb01;
            dh += args.nb01 / 2;
        }
        y4 += 4 * 256;
    }
    device float *dst_f32 = (device float *)dst + (uint64_t)im * args.ne0 * args.ne1 + (uint64_t)r1 * args.ne0;
    for (int row = 0; row < nr0 && first_row + row < args.ne0; ++row) {
        float sum_all = simd_sum(sumf[row]);
        if (tiisg == 0) dst_f32[first_row + row] = sum_all;
    }
}

kernel void f32_gemv(device const float *w [[buffer(0)]], device const float *x [[buffer(1)]],
                     device float *y [[buffer(2)]], constant Gemv &p [[buffer(3)]],
                     ushort tid [[thread_index_in_threadgroup]], uint row [[threadgroup_position_in_grid]],
                     threadgroup float *sum [[threadgroup(0)]]) {
    if (row >= p.out) return; float v = 0.0f; const device float *r = w + row * p.in;
    for (uint j = tid; j < p.in; j += 256) v += r[j] * x[j];
    sum[tid] = v; threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint step = 128; step; step >>= 1) { if (tid < step) sum[tid] += sum[tid + step]; threadgroup_barrier(mem_flags::mem_threadgroup); }
    if (!tid) y[row] = sum[0];
}

inline float k_f16(device const uchar *p) { return float(as_type<half>(*reinterpret_cast<const device ushort *>(p))); }
inline uchar k4_scale(device const uchar *s, uint index, bool minv) {
    if (index < 4) return s[index + (minv ? 4 : 0)] & 63;
    return minv ? (s[index + 4] >> 4) | ((s[index] >> 6) << 4) : (s[index + 4] & 15) | ((s[index - 4] >> 6) << 4);
}
inline float k_weight(device const uchar *b, uint type, uint i) {
    if (type == 0) {
        uint g=i/16,q=i%16,shift=2*((g%8)/2),qi=(i/128)*32+(g&1)*16+q; uchar sc=b[g];
        return k_f16(b+80)*(sc&15)*((b[16+qi]>>shift)&3)-k_f16(b+82)*(sc>>4);
    }
    if (type == 1) {
        uint g=i/16,q=i%16,shift=2*((g%8)/2),qi=(i/128)*32+(g&1)*16+q;
        uint sc=(g<8?b[96+g]&15:b[96+g-8]>>4)|(((b[104+g%4]>>(2*(g/4)))&3)<<4);
        int v=((b[32+qi]>>shift)&3)-((b[qi%32]&(1u<<(g/2)))?0:4);
        return k_f16(b+108)*(int(sc)-32)*v;
    }
    if (type == 2 || type == 3) {
        uint g=i/32,q=i%32,pair=i/64,qs=type==3?48:16; uchar packed=b[qs+pair*32+q]; uint v=(g&1)?packed>>4:packed&15;
        if(type==3&&(b[16+q]&(1u<<((g&1)+2*pair))))v+=16;
        return k_f16(b)*k4_scale(b+4,g,false)*v-k_f16(b+2)*k4_scale(b+4,g,true);
    }
    uint g=i/128,w=i%128,l=w%32,part=w/32;
    uchar lo=part==0?(b[g*64+l]&15):part==1?(b[g*64+32+l]&15):part==2?(b[g*64+l]>>4):(b[g*64+32+l]>>4);
    uint hi=(b[128+g*32+l]>>(2*part))&3;
    int scale=int(b[192+g*8+l/16+2*part]); if (scale >= 128) scale -= 256;
    return k_f16(b+208)*scale*(int(lo|(hi<<4))-32);
}
#define K_GEMV(NAME, TYPE, BYTES) \
kernel void NAME(device const uchar *w [[buffer(0)]], device const float *x [[buffer(1)]], device float *y [[buffer(2)]], constant Gemv &p [[buffer(3)]], ushort tid [[thread_index_in_threadgroup]], uint row [[threadgroup_position_in_grid]], threadgroup float *sum [[threadgroup(0)]]) { \
    float v=0; if(row<p.out) { device const uchar *r=w+row*(p.in/256)*BYTES; for(uint j=tid;j<p.in;j+=256) v+=k_weight(r+(j/256)*BYTES,TYPE,j%256)*x[j]; } sum[tid]=v; threadgroup_barrier(mem_flags::mem_threadgroup); for(uint step=128;step;step>>=1){if(tid<step)sum[tid]+=sum[tid+step];threadgroup_barrier(mem_flags::mem_threadgroup);} if(!tid&&row<p.out)y[row]=sum[0]; }
K_GEMV(q5k_gemv, 3, 176)

struct block_q3_K { uchar hmask[32]; uchar qs[64]; uchar scales[12]; half d; };

template<short nr0>
static inline void q3k_gemv_impl(device const block_q3_K *x, device const float *yy,
                                  device float *dst, uint in, uint out,
                                  uint64_t row_bytes, int first_row,
                                  ushort tiisg) {
    const int nb = in / 256;
    float yl[32];
    const short tid = tiisg / 4;
    const short ix = tiisg % 4;
    const short ip = tid / 4;
    const short il = 2 * ((tid % 4) / 2);
    const short ir = tid % 2;
    const short l0 = 8 * ir;
    const ushort4 mm[4] = {{0x0001, 0x0100, 0x0002, 0x0200},
                           {0x0004, 0x0400, 0x0008, 0x0800},
                           {0x0010, 0x1000, 0x0020, 0x2000},
                           {0x0040, 0x4000, 0x0080, 0x8000}};
    const int4 qm[2] = {{0x0003, 0x0300, 0x000c, 0x0c00},
                        {0x0030, 0x3000, 0x00c0, 0xc000}};
    const ushort4 hm = mm[2 * ip + il / 2];
    const short shift = 2 * il;
    const float v1 = il == 0 ? 4.f : 64.f;
    const float v2 = 4.f * v1;
    const uint16_t s_shift1 = 4 * ip;
    const uint16_t s_shift2 = s_shift1 + il;
    const short q_offset = 32 * ip + l0;
    const short y_offset = 128 * ip + 32 * il + l0;
    device const float *y1 = yy + ix * 256 + y_offset;
    uint32_t scales32, aux32;
    thread uint16_t *scales16 = (thread uint16_t *)&scales32;
    thread const int8_t *scales = (thread const int8_t *)&scales32;
    float sumf1[nr0] = {0.f};
    float sumf2[nr0] = {0.f};
    for (int i = ix; i < nb; i += 4) {
        for (short l = 0; l < 8; ++l) {
            yl[l + 0] = y1[l + 0];
            yl[l + 8] = y1[l + 16];
            yl[l + 16] = y1[l + 32];
            yl[l + 24] = y1[l + 48];
        }
        device const uint16_t *q = (device const uint16_t *)(x[i].qs + q_offset);
        device const uint16_t *h = (device const uint16_t *)(x[i].hmask + l0);
        device const uint16_t *a = (device const uint16_t *)(x[i].scales);
        device const half *dh = &x[i].d;
        for (short row = 0; row < nr0; ++row) {
            const float d_all = (float)dh[0];
            scales16[0] = a[4]; scales16[1] = a[5];
            aux32 = ((scales32 >> s_shift2) << 4) & 0x30303030;
            scales16[0] = a[il + 0]; scales16[1] = a[il + 1];
            scales32 = ((scales32 >> s_shift1) & 0x0f0f0f0f) | aux32;
            float s1 = 0, s2 = 0, s3 = 0, s4 = 0, s5 = 0, s6 = 0;
            for (short l = 0; l < 8; l += 2) {
                const int32_t qs = q[l / 2];
                s1 += yl[l + 0] * (qs & qm[il / 2][0]);
                s2 += yl[l + 1] * (qs & qm[il / 2][1]);
                s3 += ((h[l / 2] & hm[0]) ? 0.f : yl[l + 0]) + ((h[l / 2] & hm[1]) ? 0.f : yl[l + 1]);
                s4 += yl[l + 16] * (qs & qm[il / 2][2]);
                s5 += yl[l + 17] * (qs & qm[il / 2][3]);
                s6 += ((h[l / 2] & hm[2]) ? 0.f : yl[l + 16]) + ((h[l / 2] & hm[3]) ? 0.f : yl[l + 17]);
            }
            float d1 = d_all * (s1 + 1.f / 256.f * s2 - s3 * v1);
            float d2 = d_all * (s4 + 1.f / 256.f * s5 - s6 * v2);
            sumf1[row] += d1 * (scales[0] - 32); sumf2[row] += d2 * (scales[2] - 32);
            s1 = s2 = s3 = s4 = s5 = s6 = 0;
            for (short l = 0; l < 8; l += 2) {
                const int32_t qs = q[l / 2 + 8];
                s1 += yl[l + 8] * (qs & qm[il / 2][0]);
                s2 += yl[l + 9] * (qs & qm[il / 2][1]);
                s3 += ((h[l / 2 + 8] & hm[0]) ? 0.f : yl[l + 8]) + ((h[l / 2 + 8] & hm[1]) ? 0.f : yl[l + 9]);
                s4 += yl[l + 24] * (qs & qm[il / 2][2]);
                s5 += yl[l + 25] * (qs & qm[il / 2][3]);
                s6 += ((h[l / 2 + 8] & hm[2]) ? 0.f : yl[l + 24]) + ((h[l / 2 + 8] & hm[3]) ? 0.f : yl[l + 25]);
            }
            d1 = d_all * (s1 + 1.f / 256.f * s2 - s3 * v1);
            d2 = d_all * (s4 + 1.f / 256.f * s5 - s6 * v2);
            sumf1[row] += d1 * (scales[1] - 32); sumf2[row] += d2 * (scales[3] - 32);
            q += row_bytes / 2; h += row_bytes / 2; a += row_bytes / 2; dh += row_bytes / 2;
        }
        y1 += 4 * 256;
    }
    for (int row = 0; row < nr0; ++row) sumf1[row] = simd_sum((sumf1[row] + 0.25f * sumf2[row]) / (1 << shift));
    if (tiisg == 0) for (int row = 0; row < nr0 && first_row + row < out; ++row) dst[first_row + row] = sumf1[row];
}

kernel void q3k_gemv(constant ggml_metal_kargs_mul_mv &args [[buffer(3)]],
                     device const char *src0 [[buffer(0)]], device const char *src1 [[buffer(1)]],
                     device char *dst [[buffer(2)]], uint3 tgpig [[threadgroup_position_in_grid]],
                     ushort tiisg [[thread_index_in_simdgroup]], ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    constexpr short nr0 = 2; constexpr short NSG = 2;
    const int first_row = (tgpig.x * NSG + sgitg) * nr0;
    const uint64_t row_bytes = args.nb01;
    device const block_q3_K *x = (device const block_q3_K *)(src0 + first_row * row_bytes);
    device const float *y = (device const float *)src1;
    q3k_gemv_impl<nr0>(x, y, (device float *)dst, args.ne00, args.ne0, row_bytes, first_row, tiisg);
}

kernel void rmsnorm(device const float *x [[buffer(0)]], device const float *w [[buffer(1)]],
                    device float *out [[buffer(2)]], constant Norm &p [[buffer(3)]],
                    ushort tid [[thread_index_in_threadgroup]], uint group [[threadgroup_position_in_grid]],
                    threadgroup float *sum [[threadgroup(0)]]) {
    uint base = group * p.width; float v = 0.0f;
    for (uint i = tid; i < p.width; i += 256) v += x[base + i] * x[base + i];
    sum[tid] = v; threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint step = 128; step; step >>= 1) { if (tid < step) sum[tid] += sum[tid + step]; threadgroup_barrier(mem_flags::mem_threadgroup); }
    float scale = rsqrt(sum[0] / float(p.width) + p.eps);
    for (uint i = tid; i < p.width; i += 256) out[base + i] = x[base + i] * scale * w[i];
}

kernel void rope_neox(device float *x [[buffer(0)]], constant Rope &p [[buffer(1)]], uint index [[thread_position_in_grid]]) {
    uint n = p.heads * (p.dim / 2); if (index >= n) return; uint h = index / (p.dim / 2), i = index % (p.dim / 2);
    float theta = float(p.pos) / pow(p.base, float(i) / float(p.dim / 2)); float c = cos(theta), s = sin(theta); uint base = h * p.dim;
    float a = x[base + i], b = x[base + i + p.dim / 2]; x[base + i] = a * c - b * s; x[base + i + p.dim / 2] = a * s + b * c;
}

kernel void cache_write(device const float *src [[buffer(0)]], device float *cache [[buffer(1)]], constant Cache &p [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    if (i < p.width) cache[((p.layer * p.context + p.pos) * p.width) + i] = src[i];
}

kernel void attention_decode(device const float *q [[buffer(0)]], device const float *keys [[buffer(1)]],
                              device const float *values [[buffer(2)]], device float *out [[buffer(3)]],
                              device float *scores [[buffer(4)]], constant Attention &p [[buffer(5)]],
                              ushort tid [[thread_index_in_threadgroup]], uint head [[threadgroup_position_in_grid]],
                              threadgroup float *reduce [[threadgroup(0)]]) {
    if (head >= p.heads) return; uint kvh = head / (p.heads / p.kv_heads), width = p.kv_heads * p.dim;
    device float *head_scores = scores + head * p.context;
    float local_max = -INFINITY;
    for (uint pos = tid; pos <= p.pos; pos += 256) { const device float *k = keys + ((p.layer * p.context + pos) * width + kvh * p.dim); float dot = 0; for (uint z = 0; z < p.dim; ++z) dot += q[head * p.dim + z] * k[z]; float score = dot * rsqrt(float(p.dim)); head_scores[pos] = score; local_max = max(local_max, score); }
    reduce[tid] = local_max; threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint step = 128; step; step >>= 1) { if (tid < step) reduce[tid] = max(reduce[tid], reduce[tid + step]); threadgroup_barrier(mem_flags::mem_threadgroup); }
    float maxv = reduce[0], local_sum = 0;
    for (uint pos = tid; pos <= p.pos; pos += 256) { float value = exp(head_scores[pos] - maxv); head_scores[pos] = value; local_sum += value; }
    reduce[tid] = local_sum; threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint step = 128; step; step >>= 1) { if (tid < step) reduce[tid] += reduce[tid + step]; threadgroup_barrier(mem_flags::mem_threadgroup); }
    float inv_sum = 1.0f / reduce[0];
    for (uint j = tid; j < p.dim; j += 256) { float value = 0; for (uint pos = 0; pos <= p.pos; ++pos) value += head_scores[pos] * inv_sum * values[((p.layer * p.context + pos) * width + kvh * p.dim + j)]; out[head * p.dim + j] = value; }
}

kernel void add_inplace(device float *x [[buffer(0)]], device const float *y [[buffer(1)]], constant uint &n [[buffer(2)]], uint i [[thread_position_in_grid]]) { if (i < n) x[i] += y[i]; }
kernel void swiglu(device float *gate [[buffer(0)]], device const float *up [[buffer(1)]], constant uint &n [[buffer(2)]], uint i [[thread_position_in_grid]]) { if (i < n) { float x=gate[i]; gate[i]=(x/(1.0f+exp(-x)))*up[i]; } }

struct ExpertGemv { uint in, out, expert_stride, out_stride, input_stride; };

kernel void route_topk(device const float *route [[buffer(0)]], device uint *ids [[buffer(1)]],
                       device float *weights [[buffer(2)]], constant uint &experts [[buffer(3)]],
                       constant uint &used [[buffer(4)]], ushort tid [[thread_index_in_threadgroup]],
                       threadgroup float *scratch [[threadgroup(0)]]) {
    if (tid < experts) scratch[tid] = route[tid];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid != 0) return;
    float maxv = -INFINITY;
    for (uint e = 0; e < experts; ++e) maxv = max(maxv, scratch[e]);
    float total = 0.0f;
    for (uint e = 0; e < experts; ++e) { scratch[e] = exp(scratch[e] - maxv); total += scratch[e]; }
    for (uint e = 0; e < experts; ++e) scratch[e] /= total;
    float selected_total = 0.0f;
    for (uint slot = 0; slot < used; ++slot) {
        uint best = 0;
        for (uint e = 1; e < experts; ++e) if (scratch[e] > scratch[best]) best = e;
        ids[slot] = best; weights[slot] = scratch[best]; selected_total += scratch[best]; scratch[best] = -1.0f;
    }
    for (uint slot = 0; slot < used; ++slot) weights[slot] /= selected_total;
}

#define EXPERT_K_GEMV(NAME, TYPE, BYTES) \
kernel void NAME(device const uchar *w [[buffer(0)]], device const float *x [[buffer(1)]], device float *y [[buffer(2)]], constant ExpertGemv &p [[buffer(3)]], device const uint *ids [[buffer(4)]], ushort tid [[thread_index_in_threadgroup]], uint3 tgpig [[threadgroup_position_in_grid]], threadgroup float *sum [[threadgroup(0)]]) { \
    uint row=tgpig.x,slot=tgpig.z; float v=0; if(row<p.out) { device const uchar *r=w+uint64_t(ids[slot])*p.expert_stride+row*(p.in/256)*BYTES; device const float *input=x+slot*p.input_stride; for(uint j=tid;j<p.in;j+=256) v+=k_weight(r+(j/256)*BYTES,TYPE,j%256)*input[j]; } sum[tid]=v; threadgroup_barrier(mem_flags::mem_threadgroup); for(uint step=128;step;step>>=1){if(tid<step)sum[tid]+=sum[tid+step];threadgroup_barrier(mem_flags::mem_threadgroup);} if(!tid&&row<p.out)y[slot*p.out_stride+row]=sum[0]; \
}
kernel void expert_q4k_gemv(
        device const char *w [[buffer(0)]], device const float *input [[buffer(1)]],
        device float *output [[buffer(2)]], constant ExpertGemv &p [[buffer(3)]],
        device const uint *ids [[buffer(4)]], uint3 tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    constexpr short nr0 = 2;
    constexpr short NSG = 2;
    constexpr uint16_t kmask1 = 0x3f3f;
    constexpr uint16_t kmask2 = 0x0f0f;
    constexpr uint16_t kmask3 = 0xc0c0;
    const short ix = tiisg / 8;
    const short it = tiisg % 8;
    const short iq = it / 4;
    const short ir = it % 4;
    const int nb = p.in / 256;
    const int r0 = tgpig.x;
    const int slot = tgpig.z;
    const int first_row = (r0 * NSG + sgitg) * nr0;
    const uint64_t row_bytes = (uint64_t)(p.in / 256) * sizeof(block_q4_K);
    device const block_q4_K *x = (device const block_q4_K *)(w + (uint64_t)ids[slot] * p.expert_stride + first_row * row_bytes);
    device const float *y = input + slot * p.input_stride;
    float yl[16];
    float yh[16];
    float sumf[nr0] = {0.f};
    device const float *y4 = y + ix * 256 + 64 * iq + 8 * ir;
    uint16_t sc16[4];
    thread const uint8_t *sc8 = (thread const uint8_t *)sc16;
    for (int ib = ix; ib < nb; ib += 4) {
        float4 sumy = {0.f, 0.f, 0.f, 0.f};
        for (short i = 0; i < 8; ++i) {
            yl[i + 0] = y4[i + 0]; sumy[0] += yl[i + 0];
            yl[i + 8] = y4[i + 32]; sumy[1] += yl[i + 8];
            yh[i + 0] = y4[i + 128]; sumy[2] += yh[i + 0];
            yh[i + 8] = y4[i + 160]; sumy[3] += yh[i + 8];
        }
        device const uint16_t *sc = (device const uint16_t *)x[ib].scales + iq;
        device const uint16_t *q1 = (device const uint16_t *)x[ib].qs + 16 * iq + 4 * ir;
        device const half *dh = &x[ib].d;
        for (short row = 0; row < nr0; row++) {
            sc16[0] = sc[0] & kmask1;
            sc16[1] = sc[2] & kmask1;
            sc16[2] = ((sc[4] >> 0) & kmask2) | ((sc[0] & kmask3) >> 2);
            sc16[3] = ((sc[4] >> 4) & kmask2) | ((sc[2] & kmask3) >> 2);
            device const uint16_t *q2 = q1 + 32;
            float4 acc1 = {0.f, 0.f, 0.f, 0.f};
            float4 acc2 = {0.f, 0.f, 0.f, 0.f};
            for (short i = 0; i < 4; ++i) {
                acc1[0] += yl[2 * i + 0] * (q1[i] & 0x000F);
                acc1[1] += yl[2 * i + 1] * (q1[i] & 0x0F00);
                acc1[2] += yl[2 * i + 8] * (q1[i] & 0x00F0);
                acc1[3] += yl[2 * i + 9] * (q1[i] & 0xF000);
                acc2[0] += yh[2 * i + 0] * (q2[i] & 0x000F);
                acc2[1] += yh[2 * i + 1] * (q2[i] & 0x0F00);
                acc2[2] += yh[2 * i + 8] * (q2[i] & 0x00F0);
                acc2[3] += yh[2 * i + 9] * (q2[i] & 0xF000);
            }
            sumf[row] += dh[0] * ((acc1[0] + 1.f / 256.f * acc1[1]) * sc8[0] +
                                  (acc1[2] + 1.f / 256.f * acc1[3]) * sc8[1] / 16.f +
                                  (acc2[0] + 1.f / 256.f * acc2[1]) * sc8[4] +
                                  (acc2[2] + 1.f / 256.f * acc2[3]) * sc8[5] / 16.f) -
                         dh[1] * (sumy[0] * sc8[2] + sumy[1] * sc8[3] + sumy[2] * sc8[6] + sumy[3] * sc8[7]);
            q1 += row_bytes / 2;
            sc += row_bytes / 2;
            dh += row_bytes / 2;
        }
        y4 += 4 * 256;
    }
    for (int row = 0; row < nr0 && first_row + row < p.out; ++row) {
        float sum_all = simd_sum(sumf[row]);
        if (tiisg == 0) output[slot * p.out_stride + first_row + row] = sum_all;
    }
}

kernel void expert_q3k_gemv(
        device const char *w [[buffer(0)]], device const float *input [[buffer(1)]],
        device float *output [[buffer(2)]], constant ExpertGemv &p [[buffer(3)]],
        device const uint *ids [[buffer(4)]], uint3 tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    constexpr short nr0 = 2; constexpr short NSG = 2;
    const int slot = tgpig.z;
    const int first_row = (tgpig.x * NSG + sgitg) * nr0;
    const uint64_t row_bytes = (uint64_t)(p.in / 256) * sizeof(block_q3_K);
    device const block_q3_K *x = (device const block_q3_K *)(w + (uint64_t)ids[slot] * p.expert_stride + first_row * row_bytes);
    q3k_gemv_impl<nr0>(x, input + slot * p.input_stride, output + slot * p.out_stride,
                        p.in, p.out, row_bytes, first_row, tiisg);
}

kernel void moe_add(device float *x [[buffer(0)]], device const float *expert [[buffer(1)]],
                    device const float *weights [[buffer(2)]], constant uint &slots [[buffer(3)]],
                    constant uint &n [[buffer(4)]], uint i [[thread_position_in_grid]]) {
    if (i < n) { float value=0; for(uint slot=0;slot<slots;++slot)value+=weights[slot]*expert[slot*n+i]; x[i]+=value; }
}

kernel void expert_swiglu(device float *gate [[buffer(0)]], device const float *up [[buffer(1)]],
                          constant uint &width [[buffer(2)]], constant uint &slots [[buffer(3)]],
                          uint i [[thread_position_in_grid]]) {
    if (i < width*slots) { float x=gate[i]; gate[i]=(x/(1.0f+exp(-x)))*up[i]; }
}
