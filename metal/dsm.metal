#include <metal_stdlib>
using namespace metal;

struct Gemv { uint in, out; };
struct Norm { uint width, groups; float eps; };
struct Rope { uint heads, dim, pos; float base; };
struct Cache { uint layer, pos, context, width; };
struct Attention { uint heads, kv_heads, dim, pos, context, layer; };

kernel void q8_gemv(device const uchar *w [[buffer(0)]], device const float *x [[buffer(1)]],
                    device float *y [[buffer(2)]], constant Gemv &p [[buffer(3)]],
                    ushort tid [[thread_index_in_threadgroup]], uint row [[threadgroup_position_in_grid]],
                    threadgroup float *sum [[threadgroup(0)]]) {
    if (row >= p.out) return;
    const device uchar *r = w + row * (p.in / 32) * 34;
    float v = 0.0f;
    for (uint j = tid; j < p.in; j += 256) {
        uint block = j >> 5;
        half d = as_type<half>(*reinterpret_cast<const device ushort *>(r + block * 34));
        v += float(*reinterpret_cast<const device int8_t *>(r + block * 34 + 2 + (j & 31))) * x[j] * float(d);
    }
    sum[tid] = v; threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint step = 128; step; step >>= 1) { if (tid < step) sum[tid] += sum[tid + step]; threadgroup_barrier(mem_flags::mem_threadgroup); }
    if (!tid) y[row] = sum[0];
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
K_GEMV(q2k_gemv, 0, 84)
K_GEMV(q3k_gemv, 1, 110)
K_GEMV(q4k_gemv, 2, 144)
K_GEMV(q5k_gemv, 3, 176)
K_GEMV(q6k_gemv, 4, 210)

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
