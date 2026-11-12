/* Attention-only check: attention_prefill_fa128 vs attention_prefill_d128_tiled.
   Build: make check-attn
   Run:   ./check-attn metal/dsm.metal [pad_fill]   (pad_fill=nan checks unwritten KV rows)
   (and a CPU fp64 reference) on identical random inputs in the DSM cache layout. */
#import <Metal/Metal.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
typedef struct { uint32_t heads, kv_heads, dim, pos0, context, layer, rows; } attn_params;
static float frand(uint64_t *s) { *s = *s * 6364136223846793005ULL + 1442695040888963407ULL; return (float)((*s >> 33) & 0xFFFFFF) / 16777216.0f * 2.f - 1.f; }
static id<MTLComputePipelineState> mkpipe(id<MTLDevice> d, id<MTLLibrary> l, NSString *n) {
    NSError *e = nil; id<MTLFunction> f = [l newFunctionWithName:n]; if (!f) { fprintf(stderr, "missing %s\n", n.UTF8String); exit(1); }
    id<MTLComputePipelineState> p = [d newComputePipelineStateWithFunction:f error:&e]; if (!p) { fprintf(stderr, "%s\n", e.localizedDescription.UTF8String); exit(1); } return p; }
static double run(id<MTLDevice> d, id<MTLCommandQueue> qu, id<MTLComputePipelineState> p, int fa, id<MTLBuffer> q, id<MTLBuffer> k, id<MTLBuffer> v, id<MTLBuffer> o, attn_params a, int reps) {
    id<MTLCommandBuffer> cb = [qu commandBuffer]; id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
    for (int r = 0; r < reps; ++r) {
        [e setComputePipelineState:p]; [e setBuffer:q offset:0 atIndex:0]; [e setBuffer:k offset:0 atIndex:1]; [e setBuffer:v offset:0 atIndex:2]; [e setBuffer:o offset:0 atIndex:3]; [e setBytes:&a length:sizeof a atIndex:4];
        uint32_t qpk = a.heads / a.kv_heads;
        if (fa) { uint32_t G = 8 % qpk == 0 ? qpk : 1, R = 8 / G; [e setThreadgroupMemoryLength:12288 atIndex:0]; [e dispatchThreadgroups:MTLSizeMake((a.rows + R - 1) / R, a.heads / G, 1) threadsPerThreadgroup:MTLSizeMake(32, 4, 1)]; }
        else { [e setThreadgroupMemoryLength:32768 atIndex:0]; [e dispatchThreadgroups:MTLSizeMake((a.rows + 3) / 4, a.kv_heads, 1) threadsPerThreadgroup:MTLSizeMake(32, qpk, 1)]; }
    }
    [e endEncoding]; [cb commit]; [cb waitUntilCompleted];
    if (cb.status != MTLCommandBufferStatusCompleted) { fprintf(stderr, "command buffer failed: %s\n", cb.error.localizedDescription.UTF8String); exit(1); }
    return (cb.GPUEndTime - cb.GPUStartTime) / reps; }
int main(int argc, char **argv) {
    const char *path = argc > 1 ? argv[1] : "metal/dsm.metal"; float pad_fill = argc > 2 ? strtof(argv[2], NULL) : 0.f;
    id<MTLDevice> d = MTLCreateSystemDefaultDevice(); id<MTLCommandQueue> qu = [d newCommandQueue]; NSError *err = nil;
    NSString *src = [NSString stringWithContentsOfFile:@(path) encoding:NSUTF8StringEncoding error:&err];
    MTLCompileOptions *opt = [MTLCompileOptions new]; if (@available(macOS 15.0, *)) opt.mathMode = MTLMathModeSafe;
    id<MTLLibrary> lib = [d newLibraryWithSource:src options:opt error:&err]; if (!lib) { fprintf(stderr, "%s\n", err.localizedDescription.UTF8String); return 1; }
    id<MTLComputePipelineState> fa = mkpipe(d, lib, @"attention_prefill_fa128"), ref = mkpipe(d, lib, @"attention_prefill_d128_tiled");
    uint32_t cfg[3][2] = {{16, 8}, {32, 4}, {40, 8}}, rowsv[] = {9, 33, 63, 64, 65, 570, 1024}, pos0v[] = {0, 100}; int fail = 0;
    for (int c = 0; c < 3; ++c) for (size_t ri = 0; ri < sizeof rowsv / 4; ++ri) for (int pi = 0; pi < 2; ++pi) {
        uint32_t H = cfg[c][0], KV = cfg[c][1], D = 128, rows = rowsv[ri], pos0 = pos0v[pi], layers = 2, layer = 1, ctx = pos0 + rows + 3;
        uint32_t ctx_alloc = (ctx + 63) & ~63u, width = KV * D; uint64_t seed = 1234 + rows * 7 + pos0;
        id<MTLBuffer> q = [d newBufferWithLength:(NSUInteger)rows * H * D * 4 options:MTLResourceStorageModeShared];
        id<MTLBuffer> k = [d newBufferWithLength:(NSUInteger)layers * ctx_alloc * width * 4 options:MTLResourceStorageModeShared];
        id<MTLBuffer> v = [d newBufferWithLength:(NSUInteger)layers * ctx_alloc * width * 4 options:MTLResourceStorageModeShared];
        id<MTLBuffer> o1 = [d newBufferWithLength:(NSUInteger)rows * H * D * 4 options:MTLResourceStorageModeShared];
        id<MTLBuffer> o2 = [d newBufferWithLength:(NSUInteger)rows * H * D * 4 options:MTLResourceStorageModeShared];
        float *qp = q.contents, *kp = k.contents, *vp = v.contents;
        for (size_t i = 0; i < (size_t)rows * H * D; ++i) qp[i] = 2.f * frand(&seed);
        size_t total = (size_t)layers * ctx_alloc * width;
        for (size_t i = 0; i < total; ++i) { kp[i] = pad_fill; vp[i] = pad_fill; }
        for (uint32_t p = 0; p < pos0 + rows; ++p) for (uint32_t j = 0; j < width; ++j) { size_t off = ((size_t)layer * ctx + p) * width + j; kp[off] = 2.f * frand(&seed); vp[off] = frand(&seed); }
        attn_params a = {H, KV, D, pos0, ctx, layer, rows};
        double t2 = run(d, qu, ref, 0, q, k, v, o2, a, 1); double t1 = run(d, qu, fa, 1, q, k, v, o1, a, 1);
        if (rows >= 570) { t2 = run(d, qu, ref, 0, q, k, v, o2, a, 20); t1 = run(d, qu, fa, 1, q, k, v, o1, a, 20); }
        const float *x = o1.contents, *y = o2.contents; double num = 0, den = 0, mx = 0, cnum = 0, cden = 0; int nan = 0;
        for (size_t i = 0; i < (size_t)rows * H * D; ++i) { if (!isfinite(x[i])) ++nan; double dd = x[i] - y[i]; num += dd * dd; den += (double)y[i] * y[i]; if (fabs(dd) > mx) mx = fabs(dd); }
        /* CPU fp64 reference on a few rows (first, last, middle) for all heads */
        uint32_t chk[3] = {0, rows / 2, rows - 1}; double *sc = malloc(sizeof(double) * (pos0 + rows));
        for (int t = 0; t < 3; ++t) for (uint32_t h = 0; h < H; ++h) { uint32_t r = chk[t], kvh = h / (H / KV), qpos = pos0 + r; const float *qq = qp + ((size_t)r * H + h) * D; double m = -1e300, z = 0;
            for (uint32_t p = 0; p <= qpos; ++p) { const float *kk = kp + ((size_t)layer * ctx + p) * width + kvh * D; double s = 0; for (uint32_t j = 0; j < D; ++j) s += (double)qq[j] * kk[j]; sc[p] = s / sqrt((double)D); if (sc[p] > m) m = sc[p]; }
            for (uint32_t p = 0; p <= qpos; ++p) { sc[p] = exp(sc[p] - m); z += sc[p]; }
            for (uint32_t j = 0; j < D; ++j) { double acc = 0; for (uint32_t p = 0; p <= qpos; ++p) acc += sc[p] / z * vp[((size_t)layer * ctx + p) * width + kvh * D + j]; double g = x[((size_t)r * H + h) * D + j]; cnum += (g - acc) * (g - acc); cden += acc * acc; } }
        free(sc);
        double rel = sqrt(num / (den > 0 ? den : 1)), crel = sqrt(cnum / (cden > 0 ? cden : 1)); int bad = !(rel < 1e-3) || !(crel < 1e-3) || nan;
        fail |= bad;
        printf("H%2u/KV%u rows=%4u pos0=%3u: fa vs tiled rel=%.2e max_abs=%.2e | fa vs fp64 rel=%.2e nonfinite=%d%s", H, KV, rows, pos0, rel, mx, crel, nan, bad ? "  FAIL" : "");
        if (rows >= 570) printf("  | GPU us: fa %.0f tiled %.0f", t1 * 1e6, t2 * 1e6);
        printf("\n");
    }
    return fail; }
