// Shared by every engine: allocation, bf16, exp, thread sharding, the worker loop, sampling and prompt-lookup drafting.
// Include after the model's #defines (needs MAXB and VOCAB); the engine defines forward_mt(int id).
#pragma once
#include "threads.h"
#include <fcntl.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include "bench.h"

static void forward_mt(int id);
static int T = 1, B = 1, cur_tok[MAXB], cur_pos[MAXB], best_t[MAXB][256], quit;
static sbar_t bar;
static void *xalloc(size_t n) { void *p = aligned_alloc(64, (n + 63) & ~63ul); if (!p) { perror("alloc"); exit(1); } return p; }
static inline float bf(uint16_t h) { union { uint32_t u; float f; } c = { (uint32_t)h << 16 }; return c.f; }
static inline __m512 exp512(__m512 xv) {
    xv = _mm512_max_ps(xv, _mm512_set1_ps(-87.f));
    __m512 fx = _mm512_roundscale_ps(_mm512_mul_ps(xv, _mm512_set1_ps(1.44269504f)), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    __m512 r = _mm512_fnmadd_ps(fx, _mm512_set1_ps(0.693147181f), xv);
    __m512 p = _mm512_set1_ps(1.f / 720);
    p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(1.f / 120)); p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(1.f / 24));
    p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(1.f / 6));   p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(0.5f));
    p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(1.f));       p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(1.f));
    return _mm512_scalef_ps(p, fx);
}
static void shard(int blocks, int id, int *b0, int *b1) { *b0 = blocks * id / T; *b1 = blocks * (id + 1) / T; }
static int cpus[256];
static void *worker(void *arg) {
    int id = (int)(long)arg; pin(cpus[id]);
    for (;;) { sbar_sync(&bar); if (quit) break; forward_mt(id); sbar_sync(&bar); }
    return NULL;
}
static float temp = 0, rep = 1.0f; static int topk = 40; static uint64_t srng = 0x12345678abcdefULL;
static int sample(float *lg, int argmax, const int *hist, int n) {
    if (rep > 1.0f) {
        for (int i = n > 64 ? n - 64 : 0; i < n; i++) { float *l = &lg[hist[i]]; *l = *l > 0 ? *l / rep : *l * rep; }
        argmax = 0; for (int vv = 1; vv < VOCAB; vv++) if (lg[vv] > lg[argmax]) argmax = vv;
    }
    if (temp <= 0) return argmax;
    int idx[64]; float val[64]; int kk = topk < 64 ? topk : 64, cnt = 0;
    for (int vv = 0; vv < VOCAB; vv++) {
        if (cnt == kk && lg[vv] <= val[cnt - 1]) continue;
        int i = cnt < kk ? cnt++ : kk - 1;
        while (i > 0 && val[i - 1] < lg[vv]) { val[i] = val[i - 1]; idx[i] = idx[i - 1]; i--; }
        val[i] = lg[vv]; idx[i] = vv;
    }
    double z = 0; float p[64];
    for (int i = 0; i < cnt; i++) { p[i] = expf((val[i] - val[0]) / temp); z += p[i]; }
    srng ^= srng << 13; srng ^= srng >> 7; srng ^= srng << 17;
    double r = (srng >> 11) * (1.0 / 9007199254740992.0) * z;
    for (int i = 0; i < cnt; i++) { r -= p[i]; if (r <= 0) return idx[i]; }
    return idx[cnt - 1];
}
static int draft(const int *hist, int n, int K, int *out) {   // prompt lookup
    for (int ng = 3; ng >= 2; ng--) {
        if (n < ng + 1) continue;
        for (int i = n - ng - 1; i >= 0; i--) {
            if (memcmp(hist + i, hist + n - ng, ng * sizeof(int))) continue;
            int cnt = 0; for (int j = i + ng; j < n && cnt < K; j++) out[cnt++] = hist[j];
            if (cnt) return cnt;
        }
    }
    return 0;
}
