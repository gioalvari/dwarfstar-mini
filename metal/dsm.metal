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
                             ushort tid [[thread_index_in_threadgroup]], uint head [[threadgroup_position_in_grid]]) {
    if (head >= p.heads) return; uint kvh = head / (p.heads / p.kv_heads); uint width = p.kv_heads * p.dim;
    /* Each output lane recomputes the small decode softmax. This avoids an
       inter-threadgroup global-memory dependency and is correctness-first. */
    for (uint j = tid; j < p.dim; j += 256) {
        float maxv = -INFINITY;
        for (uint pos = 0; pos <= p.pos; ++pos) { const device float *k = keys + ((p.layer * p.context + pos) * width + kvh * p.dim); float d = 0; for (uint z = 0; z < p.dim; ++z) d += q[head * p.dim + z] * k[z]; maxv = max(maxv, d * rsqrt(float(p.dim))); }
        float sum = 0, value = 0;
        for (uint pos = 0; pos <= p.pos; ++pos) { const device float *k = keys + ((p.layer * p.context + pos) * width + kvh * p.dim); float d = 0; for (uint z = 0; z < p.dim; ++z) d += q[head * p.dim + z] * k[z]; float pscore = exp(d * rsqrt(float(p.dim)) - maxv); sum += pscore; value += pscore * values[((p.layer * p.context + pos) * width + kvh * p.dim + j)]; }
        out[head * p.dim + j] = value / sum;
    }
}

kernel void add_inplace(device float *x [[buffer(0)]], device const float *y [[buffer(1)]], constant uint &n [[buffer(2)]], uint i [[thread_position_in_grid]]) { if (i < n) x[i] += y[i]; }
kernel void swiglu(device float *gate [[buffer(0)]], device const float *up [[buffer(1)]], constant uint &n [[buffer(2)]], uint i [[thread_position_in_grid]]) { if (i < n) { float x=gate[i]; gate[i]=(x/(1.0f+exp(-x)))*up[i]; } }
