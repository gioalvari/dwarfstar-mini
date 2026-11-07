#include <metal_stdlib>
using namespace metal;

struct MulMm { int ne00, ne02; ulong nb01, nb02, nb03; int ne12; ulong nb10, nb11, nb12, nb13; int ne0, ne1; short r2, r3; };
struct MulMmIdMap0 { int ne10, ne11; ulong nb11, nb12; int ne21, ne20; ulong nb21; };
struct MulMmId { int ne00, ne02; ulong nb01, nb02, nb03; int ne11; ulong nb10, nb11, nb12, nb13; int ne20, ne0, ne1; short r2, r3; };

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
struct AttentionSplit { uint heads, kv_heads, dim, pos, context, layer, splits; };
struct RopeBatch { uint heads, dim, pos0, rows; float base; };
struct CacheBatch { uint layer, pos0, context, width, rows; };
struct AttentionPrefill { uint heads, kv_heads, dim, pos0, context, layer, rows; };
struct RopePartial { uint heads, dim, rot, pos; float base; };
struct SsmConv { uint width, history; };
struct Gdn { uint qk_heads, v_heads, state_dim, layer; float eps; };

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

// Ported from llama.cpp ggml-metal/kernels/mul_mm.metal (MIT license).
#define QK_NL 16
template <typename type4x4>
void dequantize_f32(device const float4x4 * src, short il, thread type4x4 & reg) {
    reg = (type4x4)(*src);
}

template <typename type4x4>
void dequantize_q8_0(device const block_q8_0 *xb, short il, thread type4x4 & reg) {
    device const packed_char4 * qs = (device const packed_char4 *) xb->qs;
    const float d = xb->d;

    float4x4 reg_f;

    for (int i = 0; i < 4; ++i) {
        reg_f[i] = float4(qs[4*il + i]) * d;
    }

    reg = (type4x4) reg_f;
}

template <typename type4>
void dequantize_q8_0_t4(device const block_q8_0 *xb, short il, thread type4 & reg) {
    device const packed_char4 * qs = (device const packed_char4 *) xb->qs;
    const float d = xb->d;

    reg = (type4) (float4(qs[il]) * d);
}

template <typename type4x4>
void dequantize_q2_K(device const block_q2_K *xb, short il, thread type4x4 & reg) {
    const float d = float(xb->d);
    const float min = float(xb->dmin);
    device const uint8_t * q = (device const uint8_t *)xb->qs;
    float dl, ml;
    uint8_t sc = xb->scales[il];

    q = q + 32*(il/8) + 16*(il&1);
    il = (il/2)%4;

    float coef = il > 1 ? (il > 2 ? 1.f / 64.f : 1.f / 16.f) :
                          (il > 0 ? 1.f / 4.f : 1.f);
    uchar mask = il>1 ? (il>2 ? 192    : 48)     : (il>0 ? 12    : 3);
    dl = d * (sc & 0xF) * coef, ml = min * (sc >> 4);
    for (int i = 0; i < 16; ++i) {
        reg[i/4][i%4] = dl * (q[i] & mask) - ml;
    }
}

template <typename type4x4>
void dequantize_q3_K(device const block_q3_K *xb, short il, thread type4x4 & reg) {
    const float d_all = float(xb->d);
    device const uint8_t * q = (device const uint8_t *)xb->qs;
    device const uint8_t * h = (device const uint8_t *)xb->hmask;
    device const int8_t * scales = (device const int8_t *)xb->scales;

    q = q + 32 * (il/8) + 16 * (il&1);
    h = h + 16 * (il&1);
    uint8_t m = 1 << (il/2);
    uint16_t kmask1 = (il/4)>1 ? ((il/4)>2 ? 192 : 48) : \
                                 ((il/4)>0 ? 12  : 3);
    uint16_t kmask2 = il/8 ? 0xF0 : 0x0F;
    uint16_t scale_2 = scales[il%8], scale_1 = scales[8 + il%4];
    int16_t  dl_int = (il/4)&1 ? (scale_2&kmask2) | ((scale_1&kmask1) << 2)
                               : (scale_2&kmask2) | ((scale_1&kmask1) << 4);
    float dl = il<8 ? d_all * (dl_int - 32.f) : d_all * (dl_int / 16.f - 32.f);
    const float ml = 4.f * dl;

    il = (il/2) & 3;
    const float   coef = il > 1 ? (il > 2 ? 1.f / 64.f : 1.f / 16.f) :
                                  (il > 0 ? 1.f / 4.f : 1.f);
    const uint8_t mask = il>1 ? (il>2 ? 192    : 48)     : (il>0 ? 12    : 3);
    dl *= coef;

    for (int i = 0; i < 16; ++i) {
        reg[i/4][i%4] = dl * (q[i] & mask) - (h[i] & m ? 0 : ml);
    }
}

static inline uchar2 get_scale_min_k4_just2(int j, int k, device const uchar * q) {
    return j < 4 ? uchar2{uchar(q[j+0+k] & 63), uchar(q[j+4+k] & 63)}
                 : uchar2{uchar((q[j+4+k] & 0xF) | ((q[j-4+k] & 0xc0) >> 2)), uchar((q[j+4+k] >> 4) | ((q[j-0+k] & 0xc0) >> 2))};
}

template <typename type4x4>
void dequantize_q4_K(device const block_q4_K * xb, short il, thread type4x4 & reg) {
    device const uchar * q = xb->qs;

    short is = (il/4) * 2;
    q = q + (il/4) * 32 + 16 * (il&1);
    il = il & 3;
    const uchar2 sc = get_scale_min_k4_just2(is, il/2, xb->scales);
    const float d   = float(xb->d) / (il < 2 ? 1.f : 16.f);
    const float min = float(xb->dmin);
    const float dl = d * sc[0];
    const float ml = min * sc[1];

    const ushort mask = il < 2 ? 0x0F : 0xF0;
    for (int i = 0; i < 16; ++i) {
        reg[i/4][i%4] = dl * (q[i] & mask) - ml;
    }
}

template <typename type4x4>
void dequantize_q6_K(device const block_q6_K *xb, short il, thread type4x4 & reg) {
    const half d_all = xb->d;
    device const uint16_t * ql = (device const uint16_t *)xb->ql;
    device const uint16_t * qh = (device const uint16_t *)xb->qh;
    device const int8_t * scales = (device const int8_t *)xb->scales;

    ql = ql + 32*(il/8) + 16*((il/2)&1) + 8*(il&1);
    qh = qh + 16*(il/8) + 8*(il&1);
    float sc = scales[(il%2) + 2 * ((il/2))];
    il = (il/2) & 3;

    const uint32_t kmask1 = il>1 ? (il>2 ? 0xC0C0C0C0 : 0x30303030) : (il>0 ? 0x0C0C0C0C : 0x03030303);
    const uint32_t kmask2 = il>1 ? 0xF0F0F0F0                       : 0x0F0F0F0F;
    const float ml = d_all * sc * 32.f;
    const float dl0 = d_all * sc;
    const float dl1 = dl0 / 256.f;
    const float dl2 = dl0 / (256.f * 256.f);
    const float dl3 = dl0 / (256.f * 256.f * 256.f);
    const uint8_t shr_h = il>2 ? 2 : 0;
    const uint8_t shl_h = il>1 ? 0 : (il>0 ? 2 : 4);
    const uint8_t shr_l = il>1 ? 4 : 0;
    for (int i = 0; i < 4; ++i) {
        const uint32_t  low = (ql[2*i] | (uint32_t)(ql[2*i+1] << 16)) & kmask2;
        const uint32_t high = (qh[2*i] | (uint32_t)(qh[2*i+1] << 16)) & kmask1;
        const uint32_t q = ((high << shl_h) >> shr_h) | (low >> shr_l);
        reg[i][0] = dl0 *  ((half)(q & 0xFF))       - ml;
        reg[i][1] = dl1 * ((float)(q & 0xFF00))     - ml;
        reg[i][2] = dl2 * ((float)(q & 0xFF0000))   - ml;
        reg[i][3] = dl3 * ((float)(q & 0xFF000000)) - ml;
    }
}

template<
    typename S0, typename S0_4x4, typename S0_8x8,
    typename S1, typename S1_2x4, typename S1_8x8,
    typename block_q, short nl, void (*dequantize_func)(device const block_q *, short, thread S0_4x4 &),
    typename T0, typename T0_4x4, typename T1, typename T1_2x4>
kernel void kernel_mul_mm(
        constant MulMm & args [[buffer(0)]],
        device const char * src0 [[buffer(1)]],
        device const char * src1 [[buffer(2)]],
        device       char * dst [[buffer(3)]],
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {

    threadgroup S0 * sa = (threadgroup S0 *)(shmem);
    threadgroup S1 * sb = (threadgroup S1 *)(shmem + 4096);

    constexpr int NR0 = 64;
    constexpr int NR1 = 32;

    constexpr int NK  = 32;
    constexpr int NL0 = NK/16;
    constexpr int NL1 = NK/8;

    const int im = tgpig.z;
    const int r0 = tgpig.y*NR0;
    const int r1 = tgpig.x*NR1;

    // if this block is of 64x32 shape or smaller
    const short nr0 = (args.ne0 - r0 < NR0) ? (args.ne0 - r0) : NR0;
    const short nr1 = (args.ne1 - r1 < NR1) ? (args.ne1 - r1) : NR1;

    // a thread shouldn't load data outside of the matrix
    const short lr0 = ((short)tiitg/NL0) < nr0 ? ((short)tiitg/NL0) : nr0 - 1; // 0 .. 63
    const short lr1 = ((short)tiitg/NL1) < nr1 ? ((short)tiitg/NL1) : nr1 - 1; // 0 .. 31

    const short il0 = (tiitg % NL0);

    short il = il0;

    const int i12 = im % 1;
    const int i13 = im / 1;

    const uint64_t offset0 = (i12/1)*args.nb02 + (i13/1)*args.nb03;
    const short    offset1 = il0/nl;

    device const block_q * x = (device const block_q *)(src0 + args.nb01*(r0 + lr0) + offset0) + offset1;

    const short iy = 8*(tiitg % NL1);

    device const T1 * y = (device const T1 *)(src1
        + args.nb13*i13
        + args.nb12*i12
        + args.nb11*(r1 + lr1)
        + args.nb10*iy);

    S0_8x8 ma[4];
    S1_8x8 mb[2];

    simdgroup_float8x8 mc[8];

    for (short i = 0; i < 8; i++){
        mc[i] = make_filled_simdgroup_matrix<float, 8>(0.f);
    }

    for (int loop_k = 0; loop_k < args.ne00; loop_k += NK) {
        // load data and store to threadgroup memory
        if (is_same<T0_4x4, block_q>::value && false) {
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // no need for dequantization
            for (short i = 0; i < 16; i++) {
                const short sx = 2*il0 + i/8;
                const short sy = (tiitg/NL0)/8;

              //const short lx = i%8;
              //const short ly = (tiitg/NL0)%8;
                const short lx = (tiitg/NL0)%8;
                const short ly = i%8;

                const short ib = 8*sx + sy;

                *(sa + 64*ib + 8*ly + lx) = loop_k + 16*il + i < args.ne00 ? (S0) *((device T0 *) x + i) : 0;
            }
        } else {
            S0_4x4 temp_a;
            dequantize_func(x, il, temp_a);

            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (short i = 0; i < 16; i++) {
                const short sx = 2*il0 + i/8;
                const short sy = (tiitg/NL0)/8;

              //const short lx = i%8;
              //const short ly = (tiitg/NL0)%8;
                const short lx = (tiitg/NL0)%8;
                const short ly = i%8;

                const short ib = 8*sx + sy;

                // NOTE: this is massively slower.. WTF?
                //sa[64*ib + 8*ly + lx] = temp_a[i/4][i%4];

                *(sa + 64*ib + 8*ly + lx) = temp_a[i/4][i%4];
            }
        }

        if (false) {
            for (short i = 0; i < 8; ++i) {
                const short sx = (tiitg%NL1);
                const short sy = (tiitg/NL1)/8;

                const short lx = i;
                const short ly = (tiitg/NL1)%8;
              //const short lx = (tiitg/NL1)%8;
              //const short ly = i;

                const short ib = 4*sx + sy;

                *(sb + 64*ib + 8*ly + lx) = loop_k + iy + i < args.ne00 ? (S1) *((device T1 *) y + i) : 0;
            }
        } else {
            const short sx = (tiitg%NL1);
            const short sy = (tiitg/NL1)/8;

          //const short dx = sx;
          //const short dy = sy;

            const short ly = (tiitg/NL1)%8;

            const short ib = 4*sx + sy;

            *(threadgroup S1_2x4 *)(sb + 64*ib + 8*ly) = (S1_2x4)(*((device T1_2x4 *) y));
        }

        il = (il + 2 < nl) ? il + 2 : il % 2;
        x  = (il < 2) ? x + (2 + nl - 1)/nl : x;

        y += NK;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // load matrices from threadgroup memory and conduct outer products
        threadgroup const S0 * lsma = (sa + 4*64*(sgitg%2));
        threadgroup const S1 * lsmb = (sb + 2*64*(sgitg/2));

        for (short ik = 0; ik < NK/8; ik++) {
            simdgroup_barrier(mem_flags::mem_none);

            for (short i = 0; i < 4; i++) {
                simdgroup_load(ma[i], lsma + 64*i, 8, 0, false);
            }

            simdgroup_barrier(mem_flags::mem_none);

            for (short i = 0; i < 2; i++) {
                simdgroup_load(mb[i], lsmb + 64*i, 8, 0, false);
            }

            simdgroup_barrier(mem_flags::mem_none);

            for (short i = 0; i < 8; i++){
                simdgroup_multiply_accumulate(mc[i], mb[i/4], ma[i%4], mc[i]);
            }

            lsma += 8*64;
            lsmb += 4*64;
        }
    }

    if (!true || (r0 + NR0 <= args.ne0 && r1 + NR1 <= args.ne1)) {
        // if no bounds checks on the output are needed, we can directly write to device memory
        device float * C = (device float *) dst +
            (r0 + 32*(sgitg &  1)) + \
            (uint64_t) (r1 + 16*(sgitg >> 1))*args.ne0 + (uint64_t) im*args.ne1*args.ne0;

        for (short i = 0; i < 8; i++) {
            simdgroup_store(mc[i], C + 8*(i%4) + (uint64_t) 8*args.ne0*(i/4), args.ne0, 0, false);
        }
    } else {
        // block is smaller than 64x32, we should avoid writing data outside of the matrix
        threadgroup_barrier(mem_flags::mem_threadgroup);

        threadgroup float * temp_str = ((threadgroup float *) shmem) + 32*(sgitg&1) + (16*(sgitg >> 1))*NR0;

        for (short i = 0; i < 8; i++) {
            simdgroup_store(mc[i], temp_str + 8*(i%4) + 8*NR0*(i/4), NR0, 0, false);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (sgitg == 0) {
            for (int j = tiitg; j < nr1; j += NR1) {
                device float  * D  = (device float  *) dst + r0 + (uint64_t) (r1 + j)*args.ne0 + (uint64_t) im*args.ne1*args.ne0;
                device float4 * D4 = (device float4 *) D;

                threadgroup float  * C  = temp_str + (j*NR0);
                threadgroup float4 * C4 = (threadgroup float4 *) C;

                int i = 0;
                for (; i < nr0/4; i++) {
                    *(D4 + i) = *(C4 + i);
                }

                i *= 4;
                for (; i < nr0; i++) {
                    *(D + i) = *(C + i);
                }
            }
        }
    }
}

typedef decltype(kernel_mul_mm<half, half4x4, simdgroup_half8x8, half, half2x4, simdgroup_half8x8, float4x4, 1, dequantize_f32, float, float4x4, float, float2x4>) dsm_mul_mm_t;
template [[host_name("mul_mm_q8_0")]] kernel dsm_mul_mm_t kernel_mul_mm<half, half4x4, simdgroup_half8x8, half, half2x4, simdgroup_half8x8, block_q8_0, 2, dequantize_q8_0, float, float4x4, float, float2x4>;
template [[host_name("mul_mm_q2_K")]] kernel dsm_mul_mm_t kernel_mul_mm<half, half4x4, simdgroup_half8x8, half, half2x4, simdgroup_half8x8, block_q2_K, QK_NL, dequantize_q2_K, float, float4x4, float, float2x4>;
template [[host_name("mul_mm_q3_K")]] kernel dsm_mul_mm_t kernel_mul_mm<half, half4x4, simdgroup_half8x8, half, half2x4, simdgroup_half8x8, block_q3_K, QK_NL, dequantize_q3_K, float, float4x4, float, float2x4>;
template [[host_name("mul_mm_q4_K")]] kernel dsm_mul_mm_t kernel_mul_mm<half, half4x4, simdgroup_half8x8, half, half2x4, simdgroup_half8x8, block_q4_K, QK_NL, dequantize_q4_K, float, float4x4, float, float2x4>;
template [[host_name("mul_mm_q6_K")]] kernel dsm_mul_mm_t kernel_mul_mm<half, half4x4, simdgroup_half8x8, half, half2x4, simdgroup_half8x8, block_q6_K, QK_NL, dequantize_q6_K, float, float4x4, float, float2x4>;

kernel void mul_mm_id_map0(
        constant MulMmIdMap0 & args [[buffer(0)]],
        device const int32_t * src2 [[buffer(1)]],
        device uint * tpe [[buffer(2)]],
        device int32_t * hids [[buffer(3)]],
        threadgroup uint * shmem [[threadgroup(0)]],
        ushort tid [[thread_index_in_threadgroup]]) {
    const uint expert = tid;
    const uint experts = uint(args.ne10);
    const uint slots = uint(args.ne20);
    threadgroup uint * counts = shmem;
    threadgroup uint * bases = counts + experts;

    uint count = 0;
    for (uint token = 0; token < uint(args.ne21); ++token) {
        device const int32_t * ids = (device const int32_t *)((device const char *)src2 + token * args.nb21);
        for (uint slot = 0; slot < slots; ++slot) count += ids[slot] == int(expert);
    }
    counts[expert] = count;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0) {
        uint base = 0;
        uint tile = 0;
        for (uint e = 0; e < experts; ++e) {
            bases[e] = base;
            for (uint k = 0; k < counts[e]; k += 32) {
                tpe[1 + 3 * tile] = e;
                tpe[1 + 3 * tile + 1] = base + k;
                tpe[1 + 3 * tile + 2] = min(counts[e] - k, 32u);
                ++tile;
            }
            base += counts[e] + 1;
        }
        tpe[0] = tile;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint entry = bases[expert];
    for (uint token = 0; token < uint(args.ne21); ++token) {
        device const int32_t * ids = (device const int32_t *)((device const char *)src2 + token * args.nb21);
        for (uint slot = 0; slot < slots; ++slot) {
            if (ids[slot] == int(expert)) hids[entry++] = int32_t(token * slots + slot);
        }
    }
}

template<
    typename S0, typename S0_4x4, typename S0_8x8,
    typename S1, typename S1_2x4, typename S1_8x8,
    typename block_q, short nl, void (*dequantize_func)(device const block_q *, short, thread S0_4x4 &),
    typename T0, typename T0_4x4, typename T1, typename T1_2x4>
kernel void kernel_mul_mm_id(
        constant MulMmId & args [[buffer(0)]],
        device const char * src0 [[buffer(1)]],
        device const char * src1 [[buffer(2)]],
        device const uint * tpe [[buffer(3)]],
        device const int32_t * hids [[buffer(4)]],
        device char * dst [[buffer(5)]],
        threadgroup char * shmem [[threadgroup(0)]],
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort tiitg [[thread_index_in_threadgroup]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    threadgroup S0 * sa = (threadgroup S0 *)shmem;
    threadgroup S1 * sb = (threadgroup S1 *)(shmem + 4096);
    constexpr int NR0 = 64;
    constexpr int NR1 = 32;
    constexpr int NK = 32;
    constexpr int NL0 = NK / 16;
    constexpr int NL1 = NK / 8;
    const int tile = tgpig.x;
    const int r0 = tgpig.y * NR0;
    if (tile >= int(tpe[0])) return;
    const int expert = tpe[1 + 3 * tile];
    const int id_base = tpe[1 + 3 * tile + 1];
    const short nr1 = short(min(tpe[1 + 3 * tile + 2], uint(NR1)));
    const short nr0 = args.ne0 - r0 < NR0 ? args.ne0 - r0 : NR0;
    const short lr0 = short(tiitg / NL0) < nr0 ? short(tiitg / NL0) : nr0 - 1;
    const short il0 = tiitg % NL0;
    const short lr1 = short(tiitg / NL1) < nr1 ? short(tiitg / NL1) : nr1 - 1;
    const short iy = 8 * (tiitg % NL1);
    const int id = hids[id_base + lr1];
    const short i11 = (id % args.ne20) % args.ne11;
    const int i12 = id / args.ne20;
    const uint64_t offset0 = uint64_t(expert) * args.nb02;
    const short offset1 = il0 / nl;
    device const block_q * x = (device const block_q *)(src0 + args.nb01 * (r0 + lr0) + offset0) + offset1;
    device const T1 * y = (device const T1 *)(src1 + args.nb12 * i12 + args.nb11 * i11 + args.nb10 * iy);
    S0_8x8 ma[4];
    S1_8x8 mb[2];
    simdgroup_float8x8 mc[8];
    for (short i = 0; i < 8; ++i) mc[i] = make_filled_simdgroup_matrix<float, 8>(0.f);
    const bool sg_active = nr1 > NR1 / 2 || sgitg < 2;
    short il = il0;
    for (int loop_k = 0; loop_k < args.ne00; loop_k += NK) {
        S0_4x4 temp_a;
        dequantize_func(x, il, temp_a);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (short i = 0; i < 16; ++i) {
            const short sx = 2 * il0 + i / 8;
            const short sy = (tiitg / NL0) / 8;
            const short lx = (tiitg / NL0) % 8;
            const short ly = i % 8;
            const short ib = 8 * sx + sy;
            *(sa + 64 * ib + 8 * ly + lx) = temp_a[i / 4][i % 4];
        }
        const short sx = tiitg % NL1;
        const short sy = (tiitg / NL1) / 8;
        const short ly = (tiitg / NL1) % 8;
        const short ib = 4 * sx + sy;
        *(threadgroup S1_2x4 *)(sb + 64 * ib + 8 * ly) = (S1_2x4)(*((device T1_2x4 *)y));
        il = il + 2 < nl ? il + 2 : il % 2;
        x = il < 2 ? x + (2 + nl - 1) / nl : x;
        y += NK;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg_active) {
            threadgroup const S0 * lsma = sa + 4 * 64 * (sgitg % 2);
            threadgroup const S1 * lsmb = sb + 2 * 64 * (sgitg / 2);
            for (short ik = 0; ik < NK / 8; ++ik) {
                simdgroup_barrier(mem_flags::mem_none);
                for (short i = 0; i < 4; ++i) simdgroup_load(ma[i], lsma + 64 * i, 8, 0, false);
                simdgroup_barrier(mem_flags::mem_none);
                for (short i = 0; i < 2; ++i) simdgroup_load(mb[i], lsmb + 64 * i, 8, 0, false);
                simdgroup_barrier(mem_flags::mem_none);
                for (short i = 0; i < 8; ++i) simdgroup_multiply_accumulate(mc[i], mb[i / 4], ma[i % 4], mc[i]);
                lsma += 8 * 64;
                lsmb += 4 * 64;
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg_active) {
        threadgroup float * temp = (threadgroup float *)shmem + 32 * (sgitg & 1) + 16 * (sgitg >> 1) * NR0;
        for (short i = 0; i < 8; ++i) simdgroup_store(mc[i], temp + 8 * (i % 4) + 8 * NR0 * (i / 4), NR0, 0, false);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (short j = sgitg; j < nr1; j += 4) {
        const int output_id = hids[id_base + j];
        const short slot = output_id % args.ne20;
        const int token = output_id / args.ne20;
        device float * out = (device float *)dst + r0 + uint64_t(slot) * args.ne0 + uint64_t(token) * args.ne1 * args.ne0;
        threadgroup float * in = (threadgroup float *)shmem + j * NR0;
        for (int i = tiisg; i < nr0; i += 32) out[i] = in[i];
    }
}

typedef decltype(kernel_mul_mm_id<half, half4x4, simdgroup_half8x8, half, half2x4, simdgroup_half8x8, float4x4, 1, dequantize_f32, float, float4x4, float, float2x4>) dsm_mul_mm_id_t;
template [[host_name("mul_mm_id_q8_0")]] kernel dsm_mul_mm_id_t kernel_mul_mm_id<half, half4x4, simdgroup_half8x8, half, half2x4, simdgroup_half8x8, block_q8_0, 2, dequantize_q8_0, float, float4x4, float, float2x4>;
template [[host_name("mul_mm_id_q2_K")]] kernel dsm_mul_mm_id_t kernel_mul_mm_id<half, half4x4, simdgroup_half8x8, half, half2x4, simdgroup_half8x8, block_q2_K, QK_NL, dequantize_q2_K, float, float4x4, float, float2x4>;
template [[host_name("mul_mm_id_q3_K")]] kernel dsm_mul_mm_id_t kernel_mul_mm_id<half, half4x4, simdgroup_half8x8, half, half2x4, simdgroup_half8x8, block_q3_K, QK_NL, dequantize_q3_K, float, float4x4, float, float2x4>;
template [[host_name("mul_mm_id_q4_K")]] kernel dsm_mul_mm_id_t kernel_mul_mm_id<half, half4x4, simdgroup_half8x8, half, half2x4, simdgroup_half8x8, block_q4_K, QK_NL, dequantize_q4_K, float, float4x4, float, float2x4>;
template [[host_name("mul_mm_id_q6_K")]] kernel dsm_mul_mm_id_t kernel_mul_mm_id<half, half4x4, simdgroup_half8x8, half, half2x4, simdgroup_half8x8, block_q6_K, QK_NL, dequantize_q6_K, float, float4x4, float, float2x4>;

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

kernel void rope_neox_batch(device float *x [[buffer(0)]], constant RopeBatch &p [[buffer(1)]], uint index [[thread_position_in_grid]]) {
    uint per_row = p.heads * (p.dim / 2);
    uint total = p.rows * per_row;
    if (index >= total) return;
    uint row = index / per_row;
    uint within = index % per_row;
    uint h = within / (p.dim / 2);
    uint i = within % (p.dim / 2);
    float theta = float(p.pos0 + row) / pow(p.base, float(i) / float(p.dim / 2));
    float c = cos(theta), s = sin(theta);
    uint base = row * p.heads * p.dim + h * p.dim;
    float a = x[base + i], b = x[base + i + p.dim / 2];
    x[base + i] = a * c - b * s;
    x[base + i + p.dim / 2] = a * s + b * c;
}

// Text-only IMRoPE is NEOX rotation over its first 64 dimensions.  The
// remaining 192 dimensions of each 256-wide attention head are unchanged.
kernel void rope_neox_partial(device float *x [[buffer(0)]],
                              constant RopePartial &p [[buffer(1)]],
                              uint index [[thread_position_in_grid]]) {
    uint pairs = p.rot / 2;
    uint n = p.heads * pairs;
    if (index >= n) return;
    uint h = index / pairs, i = index % pairs;
    float theta = float(p.pos) / pow(p.base, float(i) / float(pairs));
    float c = cos(theta), s = sin(theta);
    uint base = h * p.dim;
    float a = x[base + i], b = x[base + i + pairs];
    x[base + i] = a * c - b * s;
    x[base + i + pairs] = a * s + b * c;
}

kernel void split_q_gate(device const float *qg [[buffer(0)]], device float *q [[buffer(1)]],
                         device float *gate [[buffer(2)]], constant uint &dim [[buffer(3)]],
                         uint index [[thread_position_in_grid]]) {
    uint head = index / dim, element = index % dim;
    uint base = head * dim * 2;
    q[index] = qg[base + element];
    gate[index] = qg[base + dim + element];
}

kernel void split_gdn_qk(device const float *qkv [[buffer(0)]], device float *q [[buffer(1)]],
                         device float *k [[buffer(2)]], constant uint &qk_width [[buffer(3)]],
                         uint index [[thread_position_in_grid]]) {
    if (index < qk_width) {
        q[index] = qkv[index];
        k[index] = qkv[qk_width + index];
    }
}

kernel void l2norm_heads(device float *x [[buffer(0)]], constant Norm &p [[buffer(1)]],
                         ushort tid [[thread_index_in_threadgroup]],
                         uint group [[threadgroup_position_in_grid]],
                         threadgroup float *sum [[threadgroup(0)]]) {
    uint base = group * p.width;
    float value = 0.0f;
    for (uint i = tid; i < p.width; i += 256) value += x[base + i] * x[base + i];
    sum[tid] = value;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint step = 128; step; step >>= 1) {
        if (tid < step) sum[tid] += sum[tid + step];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float scale = rsqrt(sum[0] + p.eps);
    for (uint i = tid; i < p.width; i += 256) x[base + i] *= scale;
}

kernel void ssm_conv_step(device const float *input [[buffer(0)]],
                          device float *history [[buffer(1)]],
                          device const float *weights [[buffer(2)]],
                          device float *output [[buffer(3)]],
                          constant SsmConv &p [[buffer(4)]],
                          uint channel [[thread_position_in_grid]]) {
    if (channel >= p.width) return;
    device float *state = history + channel;
    device const float *w = weights + channel * (p.history + 1);
    float total = state[0] * w[0] + state[p.width] * w[1] +
                  state[2 * p.width] * w[2] + input[channel] * w[3];
    output[channel] = total / (1.0f + exp(-total));
    state[0] = state[p.width];
    state[p.width] = state[2 * p.width];
    state[2 * p.width] = input[channel];
}

kernel void gdn_gates(device float *alpha [[buffer(0)]], device float *beta [[buffer(1)]],
                      device const float *dt_bias [[buffer(2)]], device const float *ssm_a [[buffer(3)]],
                      uint index [[thread_position_in_grid]]) {
    float x = alpha[index] + dt_bias[index];
    float softplus = x > 20.0f ? x : log(1.0f + exp(x));
    alpha[index] = ssm_a[index] * softplus;
    beta[index] = 1.0f / (1.0f + exp(-beta[index]));
}

// FP32 K=1 gated DeltaNet recurrence. State is [layer][vhead][row][column],
// with contiguous rows; each simdgroup owns one state row.
kernel void gated_delta_net(device const float *q [[buffer(0)]], device const float *k [[buffer(1)]],
                            device const float *v [[buffer(2)]], device const float *g [[buffer(3)]],
                            device const float *beta [[buffer(4)]], device float *state [[buffer(5)]],
                            device float *output [[buffer(6)]], constant Gdn &p [[buffer(7)]],
                            ushort lane [[thread_index_in_simdgroup]],
                            ushort simdgroup [[simdgroup_index_in_threadgroup]],
                            uint2 group [[threadgroup_position_in_grid]]) {
    uint head = group.y, row = group.x * 4 + simdgroup;
    if (head >= p.v_heads || row >= p.state_dim) return;
    uint qk_head = head % p.qk_heads;
    uint state_base = ((p.layer * p.v_heads + head) * p.state_dim + row) * p.state_dim;
    device float *s = state + state_base;
    device const float *qv = q + qk_head * p.state_dim;
    device const float *kv = k + qk_head * p.state_dim;
    float decay = exp(g[head]);
    float sk = 0.0f;
    for (uint column = lane; column < p.state_dim; column += 32) {
        s[column] *= decay;
        sk += s[column] * kv[column];
    }
    sk = simd_sum(sk);
    float delta = (v[head * p.state_dim + row] - sk) * beta[head];
    float y = 0.0f;
    for (uint column = lane; column < p.state_dim; column += 32) {
        s[column] += kv[column] * delta;
        y += s[column] * qv[column];
    }
    y = simd_sum(y);
    if (lane == 0) output[head * p.state_dim + row] = y * rsqrt(float(p.state_dim));
}

kernel void gated_rmsnorm_silu(device float *x [[buffer(0)]], device const float *z [[buffer(1)]],
                               device const float *weight [[buffer(2)]], constant Norm &p [[buffer(3)]],
                               ushort tid [[thread_index_in_threadgroup]],
                               uint group [[threadgroup_position_in_grid]],
                               threadgroup float *sum [[threadgroup(0)]]) {
    uint base = group * p.width;
    float value = 0.0f;
    for (uint i = tid; i < p.width; i += 256) value += x[base + i] * x[base + i];
    sum[tid] = value;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint step = 128; step; step >>= 1) {
        if (tid < step) sum[tid] += sum[tid + step];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float scale = rsqrt(sum[0] / float(p.width) + p.eps);
    for (uint i = tid; i < p.width; i += 256) {
        float gate = z[base + i];
        x[base + i] = x[base + i] * scale * weight[i] * gate / (1.0f + exp(-gate));
    }
}

kernel void sigmoid_mul(device float *x [[buffer(0)]], device const float *gate [[buffer(1)]],
                        constant uint &n [[buffer(2)]], uint index [[thread_position_in_grid]]) {
    if (index < n) x[index] *= 1.0f / (1.0f + exp(-gate[index]));
}

kernel void shexp_add(device float *x [[buffer(0)]], device const float *shared [[buffer(1)]],
                      device const float *gate [[buffer(2)]], constant uint &n [[buffer(3)]],
                      uint index [[thread_position_in_grid]]) {
    if (index < n) x[index] += shared[index] / (1.0f + exp(-gate[0]));
}

kernel void cache_write(device const float *src [[buffer(0)]], device float *cache [[buffer(1)]], constant Cache &p [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    if (i < p.width) cache[((p.layer * p.context + p.pos) * p.width) + i] = src[i];
}

kernel void cache_write_batch(device const float *src [[buffer(0)]], device float *cache [[buffer(1)]], constant CacheBatch &p [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    uint total = p.rows * p.width;
    if (i >= total) return;
    uint row = i / p.width;
    uint col = i % p.width;
    cache[((p.layer * p.context + p.pos0 + row) * p.width) + col] = src[i];
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
    threadgroup_barrier(mem_flags::mem_threadgroup); // all threads read reduce[0] before reuse
    for (uint pos = tid; pos <= p.pos; pos += 256) { float value = exp(head_scores[pos] - maxv); head_scores[pos] = value; local_sum += value; }
    reduce[tid] = local_sum; threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device); // head_scores read cross-thread below
    for (uint step = 128; step; step >>= 1) { if (tid < step) reduce[tid] += reduce[tid + step]; threadgroup_barrier(mem_flags::mem_threadgroup); }
    float inv_sum = 1.0f / reduce[0];
    for (uint j = tid; j < p.dim; j += 256) { float value = 0; for (uint pos = 0; pos <= p.pos; ++pos) value += head_scores[pos] * inv_sum * values[((p.layer * p.context + pos) * width + kvh * p.dim + j)]; out[head * p.dim + j] = value; }
}

// Decode attention is split across 64-token chunks.  The first pass leaves
// each chunk in online-softmax form (maximum, denominator, numerator vector),
// so the second pass can combine chunks without materializing all scores.
kernel void attention_decode_split(device const float *q [[buffer(0)]],
                                   device const float *keys [[buffer(1)]],
                                   device const float *values [[buffer(2)]],
                                   device float *partial [[buffer(3)]],
                                   constant AttentionSplit &p [[buffer(4)]],
                                   ushort tid [[thread_index_in_threadgroup]],
                                   ushort lane [[thread_index_in_simdgroup]],
                                   ushort simdgroup [[simdgroup_index_in_threadgroup]],
                                   uint2 group [[threadgroup_position_in_grid]],
                                   threadgroup float *scratch [[threadgroup(0)]]) {
    constexpr uint CHUNK = 64;
    uint split = group.x, head = group.y;
    if (head >= p.heads || split >= p.splits) return;
    uint start = split * CHUNK;
    uint count = min(CHUNK, p.pos + 1 - start);
    uint kvh = head / (p.heads / p.kv_heads);
    uint width = p.kv_heads * p.dim;
    threadgroup float *scores = scratch;
    threadgroup float *reduce = scratch + CHUNK;

    // Eight simdgroups compute eight 128-wide QK dots at a time.  A lane owns
    // four elements, with the loop retaining support for non-128 dimensions.
    for (uint base = 0; base < count; base += 8) {
        uint index = base + simdgroup;
        float dot = 0.0f;
        if (index < count) {
            device const float *key = keys +
                ((p.layer * p.context + start + index) * width + kvh * p.dim);
            device const float *query = q + head * p.dim;
            for (uint z = lane * 4; z < p.dim; z += 128) {
                dot += query[z] * key[z];
                if (z + 1 < p.dim) dot += query[z + 1] * key[z + 1];
                if (z + 2 < p.dim) dot += query[z + 2] * key[z + 2];
                if (z + 3 < p.dim) dot += query[z + 3] * key[z + 3];
            }
        }
        dot = simd_sum(dot);
        if (lane == 0 && index < count) scores[index] = dot * rsqrt(float(p.dim));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    reduce[tid] = tid < count ? scores[tid] : -INFINITY;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint step = 128; step; step >>= 1) {
        if (tid < step) reduce[tid] = max(reduce[tid], reduce[tid + step]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float maximum = reduce[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float local_sum = tid < count ? exp(scores[tid] - maximum) : 0.0f;
    reduce[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint step = 128; step; step >>= 1) {
        if (tid < step) reduce[tid] += reduce[tid + step];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float denominator = reduce[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    device float *dst = partial + (head * p.splits + split) * (p.dim + 2);
    if (tid == 0) { dst[0] = maximum; dst[1] = denominator; }
    for (uint j = tid; j < p.dim; j += 256) {
        float numerator = 0.0f;
        for (uint i = 0; i < count; ++i) {
            float weight = exp(scores[i] - maximum);
            numerator += weight * values[((p.layer * p.context + start + i) * width +
                                           kvh * p.dim + j)];
        }
        dst[j + 2] = numerator;
    }
}

kernel void attention_decode_merge(device const float *partial [[buffer(0)]],
                                   device float *out [[buffer(1)]],
                                   constant AttentionSplit &p [[buffer(2)]],
                                   ushort tid [[thread_index_in_threadgroup]],
                                   uint head [[threadgroup_position_in_grid]],
                                   threadgroup float *reduce [[threadgroup(0)]]) {
    if (head >= p.heads) return;
    uint stride = p.dim + 2;
    float local_maximum = -INFINITY;
    for (uint split = tid; split < p.splits; split += 256)
        local_maximum = max(local_maximum, partial[(head * p.splits + split) * stride]);
    reduce[tid] = local_maximum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint step = 128; step; step >>= 1) {
        if (tid < step) reduce[tid] = max(reduce[tid], reduce[tid + step]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float maximum = reduce[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float local_sum = 0.0f;
    for (uint split = tid; split < p.splits; split += 256) {
        device const float *src = partial + (head * p.splits + split) * stride;
        local_sum += src[1] * exp(src[0] - maximum);
    }
    reduce[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint step = 128; step; step >>= 1) {
        if (tid < step) reduce[tid] += reduce[tid + step];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float denominator = reduce[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint j = tid; j < p.dim; j += 256) {
        float numerator = 0.0f;
        for (uint split = 0; split < p.splits; ++split) {
            device const float *src = partial + (head * p.splits + split) * stride;
            numerator += src[j + 2] * exp(src[0] - maximum);
        }
        out[head * p.dim + j] = numerator / denominator;
    }
}

// Tiled online flash attention.  One threadgroup owns four query rows for one
// KV head; its simdgroups own the GQA query heads.  This is adapted from the
// online-softmax approach in llama.cpp's MIT-licensed flash-attention kernels.
kernel void attention_prefill(device const float *q [[buffer(0)]], device const float *keys [[buffer(1)]],
                              device const float *values [[buffer(2)]], device float *out [[buffer(3)]],
                              constant AttentionPrefill &p [[buffer(4)]],
                              ushort tid [[thread_index_in_threadgroup]],
                              ushort lane [[thread_index_in_simdgroup]],
                              ushort simdgroup [[simdgroup_index_in_threadgroup]],
                              uint2 group [[threadgroup_position_in_grid]],
                              threadgroup float *scratch [[threadgroup(0)]]) {
    constexpr uint ROWS = 4;
    constexpr uint KEYS = 16;
    constexpr uint DIM = 128;
    uint row0 = group.x * ROWS;
    uint kvh = group.y;
    uint q_per_kv = p.heads / p.kv_heads;
    if (kvh >= p.kv_heads || p.dim != DIM || q_per_kv > 8) return;
    uint width = p.kv_heads * p.dim;

    threadgroup float *k_tile = scratch;
    threadgroup float *v_tile = k_tile + KEYS * DIM;
    threadgroup float *q_tile = v_tile + KEYS * DIM;
    uint q_count = min(ROWS, p.rows - row0) * q_per_kv * DIM;
    for (uint i = tid; i < q_count; i += q_per_kv * 32) {
        uint row = i / (q_per_kv * DIM);
        uint in_row = i % (q_per_kv * DIM);
        uint head = kvh * q_per_kv + in_row / DIM;
        q_tile[i] = q[(row0 + row) * p.heads * DIM + head * DIM + in_row % DIM];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint row = 0; row < ROWS && row0 + row < p.rows; ++row) {
        uint qpos = p.pos0 + row0 + row;
        device float4 *dst = (device float4 *)(out +
            ((row0 + row) * p.heads + kvh * q_per_kv + simdgroup) * DIM);
        threadgroup const float4 *query = (threadgroup const float4 *)(q_tile +
            (row * q_per_kv + simdgroup) * DIM);
        float4 result = 0.0f;
        float maximum = -INFINITY;
        float sum = 0.0f;
        for (uint base = 0; base <= qpos; base += KEYS) {
            uint count = min(KEYS, qpos + 1 - base);
            uint tile_floats = count * DIM;
            for (uint i = tid; i < tile_floats; i += q_per_kv * 32) {
                uint pos = i / DIM;
                uint dim = i % DIM;
                uint offset = (p.layer * p.context + base + pos) * width + kvh * DIM + dim;
                k_tile[i] = keys[offset];
                v_tile[i] = values[offset];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            float score = -INFINITY;
            if (lane < count) {
                score = 0.0f;
                threadgroup const float4 *key = (threadgroup const float4 *)(k_tile + lane * DIM);
                for (uint i = 0; i < DIM / 4; ++i) score += dot(query[i], key[i]);
                score *= rsqrt(128.0f);
            }
            float next_maximum = max(maximum, simd_max(score));
            float old_scale = exp(maximum - next_maximum);
            float weight = exp(score - next_maximum);
            result *= old_scale;
            threadgroup const float4 *value = (threadgroup const float4 *)v_tile;
            for (uint i = 0; i < count; ++i) result += value[i * (DIM / 4) + lane] * simd_shuffle(weight, i);
            sum = sum * old_scale + simd_sum(weight);
            maximum = next_maximum;
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        dst[lane] = result / sum;
    }
}

kernel void add_inplace(device float *x [[buffer(0)]], device const float *y [[buffer(1)]], constant uint &n [[buffer(2)]], uint i [[thread_position_in_grid]]) { if (i < n) x[i] += y[i]; }
kernel void swiglu(device float *gate [[buffer(0)]], device const float *up [[buffer(1)]], constant uint &n [[buffer(2)]], uint i [[thread_position_in_grid]]) { if (i < n) { float x=gate[i]; gate[i]=(x/(1.0f+exp(-x)))*up[i]; } }

struct ExpertGemv { uint in, out, expert_stride, out_stride, input_stride; };

kernel void route_topk(device const float *route [[buffer(0)]], device uint *ids [[buffer(1)]],
                        device float *weights [[buffer(2)]], constant uint &experts [[buffer(3)]],
                        constant uint &used [[buffer(4)]], ushort tid [[thread_index_in_threadgroup]],
                        uint group [[threadgroup_position_in_grid]],
                        threadgroup float *scratch [[threadgroup(0)]]) {
    route += group * experts;
    ids += group * used;
    weights += group * used;
    if (tid < experts) scratch[tid] = route[tid];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    threadgroup float maxes[8];
    threadgroup float total;
    threadgroup float selected_total;
    threadgroup float slot_max;
    threadgroup uint slot_index;
    const uint simdgroups = (experts + 31) / 32;
    float local_max = tid < experts ? scratch[tid] : -INFINITY;
    float simd_maximum = simd_max(local_max);
    if ((tid & 31) == 0) maxes[tid / 32] = simd_maximum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float maximum = -INFINITY;
        for (uint sg = 0; sg < simdgroups; ++sg) maximum = max(maximum, maxes[sg]);
        total = maximum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < experts) scratch[tid] = exp(scratch[tid] - total);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float denominator = 0.f;
        for (uint e = 0; e < experts; ++e) denominator += scratch[e];
        total = denominator;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < experts) scratch[tid] /= total;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) selected_total = 0.f;
    for (uint slot = 0; slot < used; ++slot) {
        float candidate = tid < experts ? scratch[tid] : -INFINITY;
        float maximum = simd_max(candidate);
        uint index = candidate == maximum ? tid : UINT_MAX;
        uint local_index = simd_min(index);
        if ((tid & 31) == 0) { maxes[tid / 32] = maximum; scratch[experts + tid / 32] = as_type<float>(local_index); }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0) {
            float best = maxes[0]; uint best_index = as_type<uint>(scratch[experts]);
            for (uint sg = 1; sg < simdgroups; ++sg) {
                float value = maxes[sg]; uint value_index = as_type<uint>(scratch[experts + sg]);
                if (value > best || (value == best && value_index < best_index)) {
                    best = value; best_index = value_index;
                }
            }
            slot_max = best; slot_index = best_index;
            ids[slot] = best_index; weights[slot] = best;
            selected_total += best;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == slot_index) scratch[tid] = -1.f;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) for (uint slot = 0; slot < used; ++slot) weights[slot] /= selected_total;
}

#define EXPERT_K_GEMV(NAME, TYPE, BYTES) \
kernel void NAME(device const uchar *w [[buffer(0)]], device const float *x [[buffer(1)]], device float *y [[buffer(2)]], constant ExpertGemv &p [[buffer(3)]], device const uint *ids [[buffer(4)]], ushort tid [[thread_index_in_threadgroup]], uint3 tgpig [[threadgroup_position_in_grid]], threadgroup float *sum [[threadgroup(0)]]) { \
    uint row=tgpig.x,slot=tgpig.z; float v=0; if(row<p.out) { device const uchar *r=w+uint64_t(ids[slot])*p.expert_stride+row*(p.in/256)*BYTES; device const float *input=x+slot*p.input_stride; for(uint j=tid;j<p.in;j+=256) v+=k_weight(r+(j/256)*BYTES,TYPE,j%256)*input[j]; } sum[tid]=v; threadgroup_barrier(mem_flags::mem_threadgroup); for(uint step=128;step;step>>=1){if(tid<step)sum[tid]+=sum[tid+step];threadgroup_barrier(mem_flags::mem_threadgroup);} if(!tid&&row<p.out)y[slot*p.out_stride+row]=sum[0]; \
}
EXPERT_K_GEMV(expert_q6k_gemv, 4, 210)
kernel void expert_q8_0_gemv(
        device const block_q8_0 *w [[buffer(0)]], device const float *x [[buffer(1)]],
        device float *y [[buffer(2)]], constant ExpertGemv &p [[buffer(3)]],
        device const uint *ids [[buffer(4)]], ushort tid [[thread_index_in_threadgroup]],
        uint3 tgpig [[threadgroup_position_in_grid]], threadgroup float *sum [[threadgroup(0)]]) {
    uint row = tgpig.x, slot = tgpig.z;
    float value = 0;
    if (row < p.out) {
        uint64_t row_bytes = (uint64_t)(p.in / 32) * sizeof(block_q8_0);
        device const block_q8_0 *weights = (device const block_q8_0 *)((device const char *)w +
            (uint64_t)ids[slot] * p.expert_stride + (uint64_t)row * row_bytes);
        device const float *input = x + slot * p.input_stride;
        for (uint j = tid; j < p.in; j += 256) value += float(weights[j / 32].qs[j % 32]) *
            float(weights[j / 32].d) * input[j];
    }
    sum[tid] = value;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint step = 128; step; step >>= 1) {
        if (tid < step) sum[tid] += sum[tid + step];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (!tid && row < p.out) y[slot * p.out_stride + row] = sum[0];
}
kernel void expert_q8_0_gemv_fast(
        device const block_q8_0 *w [[buffer(0)]], device const float *x [[buffer(1)]],
        device float *y [[buffer(2)]], constant ExpertGemv &p [[buffer(3)]],
        device const uint *ids [[buffer(4)]], threadgroup char *shmem [[threadgroup(0)]],
        uint3 group [[threadgroup_position_in_grid]], ushort lane [[thread_index_in_simdgroup]],
        ushort simdgroup [[simdgroup_index_in_threadgroup]]) {
    constexpr short NR0 = 2;
    constexpr short NSG = 4;
    constexpr short NQ = 8;
    const uint slot = group.z;
    const uint first_row = group.x * NR0;
    const uint blocks = p.in / 32;
    const uint ix = lane / (32 / NQ);
    const uint il = lane % (32 / NQ);
    const uint block0 = simdgroup * NQ + ix;
    const uint64_t row_bytes = (uint64_t)blocks * sizeof(block_q8_0);
    device const block_q8_0 *weights[NR0];
    for (short row = 0; row < NR0; ++row) weights[row] = (device const block_q8_0 *)
        ((device const char *)w + (uint64_t)ids[slot] * p.expert_stride +
         (uint64_t)(first_row + row) * row_bytes);
    device const float *input = x + slot * p.input_stride;
    device const float *input_block = input + block0 * 32 + il * NQ;
    float sums[NR0] = {0.f};
    for (uint block = block0; block < blocks; block += NSG * NQ) {
        float values[NQ];
        for (short i = 0; i < NQ; ++i) values[i] = input_block[i];
        for (short row = 0; row < NR0; ++row) {
            if (first_row + row >= p.out) continue;
            device const char *qs = weights[row][block].qs + il * NQ;
            float dot = 0.f;
            for (short i = 0; i < NQ; ++i) dot += float(qs[i]) * values[i];
            sums[row] += dot * float(weights[row][block].d);
        }
        input_block += NSG * NQ * 32;
    }
    threadgroup float *reduce = (threadgroup float *)shmem;
    for (short row = 0; row < NR0; ++row) {
        if (simdgroup == 0) reduce[row * 32 + lane] = 0.f;
        sums[row] = simd_sum(sums[row]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (short row = 0; row < NR0; ++row) if (lane == 0) reduce[row * 32 + simdgroup] = sums[row];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (short row = 0; row < NR0; ++row) {
        float total = simd_sum(reduce[row * 32 + lane]);
        if (lane == 0 && simdgroup == 0 && first_row + row < p.out) {
            y[slot * p.out_stride + first_row + row] = total;
        }
    }
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

kernel void moe_add_batch(device float *x [[buffer(0)]], device const float *expert [[buffer(1)]],
                          device const float *weights [[buffer(2)]], constant uint &slots [[buffer(3)]],
                          constant uint &width [[buffer(4)]], constant uint &rows [[buffer(5)]],
                          uint i [[thread_position_in_grid]]) {
    if (i >= rows * width) return;
    uint row = i / width, col = i % width;
    float value = 0;
    for (uint slot = 0; slot < slots; ++slot) value += weights[row * slots + slot] * expert[(row * slots + slot) * width + col];
    x[i] += value;
}

kernel void expert_swiglu(device float *gate [[buffer(0)]], device const float *up [[buffer(1)]],
                           constant uint &width [[buffer(2)]], constant uint &slots [[buffer(3)]],
                           constant uint &rows [[buffer(4)]], uint i [[thread_position_in_grid]]) {
    if (i < width * slots * rows) { float x=gate[i]; gate[i]=(x/(1.0f+exp(-x)))*up[i]; }
}
