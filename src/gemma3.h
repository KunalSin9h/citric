// Gemma 3 1B pieces shared by gemma.c (4-bit) and gemma3t.c (ternary): lane state, (1+w) RMSNorm, RoPE, GeGLU,
// int8 KV cache and attention. Include after the model #defines and common.h.
#pragma once
typedef struct { int8_t *k8, *v8; float *ksc, *vsc; int32_t *ksum; } kvh_t;
static kvh_t *kvh; static int kv_map[MAXB], nkv = 1;
#define KVH(b, l) (&kvh[(size_t)kv_map[b] * NLAYER + (l)])
static float rope_c[2][MAXCTX][HD / 2], rope_s[2][MAXCTX][HD / 2];   // [0] local base 10k, [1] global base 1M

static float x[MAXB][HID], qkv[MAXB][QKV], att[MAXB][QDIM], gu[MAXB][2 * FFN], tmp[MAXB][HID];
static float *logits;   // [MAXB][VOCAB]
#define q(b) (qkv[b])
#define k(b) (qkv[b] + QDIM)
#define v(b) (qkv[b] + QDIM + KVDIM)
#define g(b) (gu[b])
#define u(b) (gu[b] + FFN)
// Gemma RMSNorm: x * rsqrt(mean(x^2) + eps) * (1 + w)
static void rmsnorm(const float *xf, const float *w, float *o, int n) {
    __m512 ss = _mm512_setzero_ps();
    for (int i = 0; i < n; i += 16) { __m512 xv = _mm512_loadu_ps(xf + i); ss = _mm512_fmadd_ps(xv, xv, ss); }
    __m512 r = _mm512_set1_ps(1.0f / sqrtf(_mm512_reduce_add_ps(ss) / n + EPS)), one = _mm512_set1_ps(1.f);
    for (int i = 0; i < n; i += 16) _mm512_storeu_ps(o + i, _mm512_mul_ps(_mm512_add_ps(one, _mm512_loadu_ps(w + i)), _mm512_mul_ps(_mm512_loadu_ps(xf + i), r)));
}
static void rope(float *p, int pos, int glob) {   // rotate-half over one head
    const float *c = rope_c[glob][pos], *s = rope_s[glob][pos];
    for (int i = 0; i < HD / 2; i++) { float x1 = p[i], x2 = p[i + HD / 2]; p[i] = x1 * c[i] - x2 * s[i]; p[i + HD / 2] = x2 * c[i] + x1 * s[i]; }
}
// tanh-GELU(g) * u on rows i0..i1 of every stream: gelu(x) = 0.5 x (1 + tanh(0.79788456 (x + 0.044715 x^3)))
static void gelu_rows(int i0, int i1) {
    const __m512 c0 = _mm512_set1_ps(0.79788456f), c1 = _mm512_set1_ps(0.044715f), one = _mm512_set1_ps(1.f), two = _mm512_set1_ps(2.f), half = _mm512_set1_ps(0.5f);
    for (int b = 0; b < B; b++)
        for (int i = i0; i < i1; i += 16) {
            __m512 gv = _mm512_loadu_ps(g(b) + i);
            __m512 z = _mm512_mul_ps(c0, _mm512_fmadd_ps(_mm512_mul_ps(c1, _mm512_mul_ps(gv, gv)), gv, gv));
            __m512 th = _mm512_sub_ps(one, _mm512_div_ps(two, _mm512_add_ps(one, exp512(_mm512_mul_ps(two, z)))));   // tanh
            _mm512_storeu_ps(g(b) + i, _mm512_mul_ps(_mm512_mul_ps(_mm512_mul_ps(half, gv), _mm512_add_ps(one, th)), _mm512_loadu_ps(u(b) + i)));
        }
}
// ---------- KV cache + attention (int8, head-major token-blocked; one kv head) ----------
static void kv_alloc(void) {
    kvh = xalloc((size_t)nkv * NLAYER * sizeof *kvh);
    for (size_t i = 0; i < (size_t)nkv * NLAYER; i++) {
        kvh[i].k8 = xalloc(MAXCTX * HD); kvh[i].v8 = xalloc(MAXCTX * HD);
        kvh[i].ksc = xalloc(MAXCTX * 4); kvh[i].vsc = xalloc(MAXCTX * 4); kvh[i].ksum = xalloc(MAXCTX * 4);
    }
}
static float quant_head(const float *xf, int8_t *qo) {   // absmax int8 over HD
    float am = 1e-8f; for (int d = 0; d < HD; d++) am = fmaxf(am, fabsf(xf[d]));
    float s = 127.0f / am; for (int d = 0; d < HD; d++) { float t = rintf(xf[d] * s); qo[d] = (int8_t)(t < -128 ? -128 : t > 127 ? 127 : t); }
    return 1.0f / s;
}
static void kv_insert(kvh_t *c, int pos, const float *kf, const float *vf) {
    int8_t k8[HD], v8[HD];
    c->ksc[pos] = quant_head(kf, k8); c->vsc[pos] = quant_head(vf, v8);
    int32_t sum = 0; for (int d = 0; d < HD; d++) sum += k8[d]; c->ksum[pos] = sum;
    int8_t *kb = c->k8 + (size_t)(pos / 16) * HD * 16 + (pos % 16) * 4;
    for (int gg = 0; gg < HD / 4; gg++) memcpy(kb + gg * 64, k8 + 4 * gg, 4);
    int8_t *vb = c->v8 + (size_t)(pos / 4) * HD * 4 + pos % 4;
    for (int d = 0; d < HD; d++) vb[d * 4] = v8[d];
}
// one (stream, layer, q head). Window: local layers see positions [pos-WINDOW+1, pos].
static void attention_head(int b, int l, int h, int pos, float *out) {
    const kvh_t *c = KVH(b, l);
    int8_t q8[HD]; uint8_t qu[HD];
    float qs = quant_head(q(b) + h * HD, q8) / 16.0f;   // query_pre_attn_scalar = 256 -> 1/16
    for (int d = 0; d < HD; d++) qu[d] = q8[d] + 128;
    const int32_t *q4 = (const int32_t *)qu;
    int t0 = IS_GLOBAL(l) || pos < WINDOW ? 0 : pos - WINDOW + 1, tb0 = t0 / 16, nblk = pos / 16 + 1;
    float sc[MAXCTX]; __m512 mxv = _mm512_set1_ps(-1e30f);
    for (int tb = tb0; tb < nblk; tb++) {
        __m512i acc = _mm512_setzero_si512(); const int8_t *kb = c->k8 + (size_t)tb * HD * 16;
        for (int gg = 0; gg < HD / 4; gg++) acc = _mm512_dpbusd_epi32(acc, _mm512_set1_epi32(q4[gg]), _mm512_loadu_si512(kb + gg * 64));
        __m512i comp = _mm512_slli_epi32(_mm512_loadu_si512(c->ksum + tb * 16), 7);
        __m512 sv = _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_sub_epi32(acc, comp)), _mm512_mul_ps(_mm512_loadu_ps(c->ksc + tb * 16), _mm512_set1_ps(qs)));
        int lo = t0 > tb * 16 ? t0 - tb * 16 : 0, hi = pos - tb * 16 < 15 ? pos - tb * 16 : 15;
        __mmask16 valid = (__mmask16)(((1u << (hi + 1)) - 1) & ~((1u << lo) - 1));
        sv = _mm512_mask_mov_ps(_mm512_set1_ps(-1e30f), valid, sv);
        mxv = _mm512_max_ps(mxv, sv); _mm512_storeu_ps(sc + tb * 16, sv);
    }
    __m512 mx = _mm512_set1_ps(_mm512_reduce_max_ps(mxv)), sum = _mm512_setzero_ps(), pm = _mm512_setzero_ps();
    for (int tb = tb0; tb < nblk; tb++) {
        __m512 e = exp512(_mm512_sub_ps(_mm512_loadu_ps(sc + tb * 16), mx)); sum = _mm512_add_ps(sum, e);
        __m512 pp = _mm512_mul_ps(e, _mm512_loadu_ps(c->vsc + tb * 16)); pm = _mm512_max_ps(pm, pp); _mm512_storeu_ps(sc + tb * 16, pp);
    }
    float pmax = _mm512_reduce_max_ps(pm), inv = pmax / (255.f * _mm512_reduce_add_ps(sum));
    uint8_t pu[MAXCTX]; __m512 r255 = _mm512_set1_ps(255.f / pmax);
    for (int tb = tb0; tb < nblk; tb++)
        _mm_storeu_si128((__m128i *)(pu + tb * 16), _mm512_cvtusepi32_epi8(_mm512_cvtps_epi32(_mm512_mul_ps(_mm512_loadu_ps(sc + tb * 16), r255))));
    __m512i acc[HD / 16]; for (int j = 0; j < HD / 16; j++) acc[j] = _mm512_setzero_si512();
    for (int vb = tb0 * 4; vb <= pos / 4; vb++) {
        __m512i pb = _mm512_set1_epi32(*(const int32_t *)(pu + 4 * vb)); const int8_t *vr = c->v8 + (size_t)vb * HD * 4;
        for (int j = 0; j < HD / 16; j++) acc[j] = _mm512_dpbusd_epi32(acc[j], pb, _mm512_loadu_si512(vr + j * 64));
    }
    for (int j = 0; j < HD / 16; j++) _mm512_storeu_ps(out + h * HD + j * 16, _mm512_mul_ps(_mm512_cvtepi32_ps(acc[j]), _mm512_set1_ps(inv)));
}
static inline int is_eos(int t) { return t == 1 || t == 106; }   // <eos>, <end_of_turn>
