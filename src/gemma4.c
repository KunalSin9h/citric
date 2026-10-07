// Citric engine for google/gemma-4-E4B-it (Gemma 4 text, effective 4B, instruction tuned). Text path only.
// 42 layers, hidden 2560, ffn 10240 GeGLU, 8 q heads, 2 kv heads. Sliding layers (5 of 6): head_dim 256,
// window 512, RoPE base 10k. Global layers: head_dim 512, RoPE base 1M on the first quarter of the dims
// ("proportional"), full causal. Attention scaling 1.0, QK-norm, scale-free V-norm. RMSNorm multiplies by w.
// Layers 24..41 compute no K/V: they attend over layer 22's (sliding) or 23's (global) cache.
// Per-layer embeddings (PLE): a 262144 x (42*256) bf16 table read as a lookup, combined with a projection
// of the token embedding, gated into the residual after each MLP. Residual scaled by a per-layer scalar.
// Final logits soft-capped: 30 * tanh(l / 30). lm_head tied to the (quantized) embedding matrix.
// Weights 4-bit, per-row per-32-block fp16 scales, weights as the unsigned vpdpbusd operand. Activations int8.
// KV int8, head-major token-blocked. Lanes = (token, position, kv stream): prefill, batching, speculation.
// usage: ./gemma4 models/gemma-4-E4B-it N_GEN tok...   |   ./gemma4 models/gemma-4-E4B-it -   (stdin: "ngen id id ..." per line)

#define HID 2560
#define FFN 10240
#define NH 8
#define NKV 2
#define NLAYER 42
#define KV_SHARED_FROM 24      // layers >= this reuse KV of layer 22 (sliding) / 23 (global)
#define VOCAB 262144
#define PLE 256                // per-layer input width
#define MAXCTX 1024
#define WINDOW 512
#define EPS 1e-6f
#define SOFTCAP 30.0f
#ifndef MAXB
#define MAXB 32
#endif
#define IS_GLOBAL(l) (((l) + 1) % 6 == 0)
#define HDIM(l) (IS_GLOBAL(l) ? 512 : 256)
#define QD(l) (NH * HDIM(l))
#define KVD(l) (NKV * HDIM(l))
#define KV_SRC(l) ((l) < KV_SHARED_FROM ? (l) : IS_GLOBAL(l) ? 23 : 22)
#define QKV_MAX (NH * 512 + 2 * NKV * 512)   // 6144

#include "common.h"

typedef struct { int n, k, w4; uint8_t *w; float *scale; uint16_t *sc; } qmat_t;   // w4=0: int8 w+128 [n/16][k/4][16][4], scale per row. w4=1: nibbles w+8 [n/16][k/8][16][4], fp16 scale per (row, 32-block)
typedef struct { qmat_t qkv, o, gu, down, pgate, pproj; float *ln_in, *ln_post_attn, *ln_pre_ffn, *ln_post_ffn, *ln_ple, *qn, *kn; float scalar; int nqkv; } layer_t;

static layer_t L[NLAYER];
static qmat_t emb4, pleproj;             // tied embeddings as lm_head; per_layer_model_projection [42*256, HID]
static const uint16_t *emb_bf16, *ple_bf16;   // mmap: [VOCAB][HID], [VOCAB][42*256]
static float *ln_final, *ln_pleproj;
typedef struct { int hd; int8_t *k8, *v8; float *ksc, *vsc; int32_t *ksum; } kvh_t;
static kvh_t *kvh; static int kv_map[MAXB], nkv = 1;
#define KVH(b, l, h) (&kvh[((size_t)kv_map[b] * KV_SHARED_FROM + KV_SRC(l)) * NKV + (h)])
static float rope_c[2][MAXCTX][256], rope_s[2][MAXCTX][256];   // [0] sliding hd 256 (128 angles), [1] global hd 512 (256 angles, 64 rotated)

static float x[MAXB][HID], qkv[MAXB][QKV_MAX], att[MAXB][NH * 512], gu[MAXB][2 * FFN], tmp[MAXB][HID];
static float plep[MAXB][NLAYER * PLE], ple[MAXB][NLAYER][PLE], pg[MAXB][PLE];
static float *logits;   // [MAXB][VOCAB]
static double prof[8]; static const char *pname[6] = {"matvec", "quant", "attention", "norm/rope", "gelu/ple", "lm_head"};
#define P(i, stmt) do { double t_ = now_ns(); stmt; if (!id) prof[i] += now_ns() - t_; } while (0)

// ---------- safetensors (bf16, multimodal checkpoint; we read model.language_model.* only) ----------
static char *hdr; static uint8_t *data; static int model_fd; static size_t model_size;
static const uint16_t *tensor(const char *name) {
    char key[256]; snprintf(key, sizeof key, "\"%s\"", name);
    char *p = strstr(hdr, key); if (!p) { fprintf(stderr, "missing tensor %s\n", name); exit(1); }
    p = strstr(p, "\"data_offsets\""); p = strchr(p, '[');
    return (const uint16_t *)(data + strtoul(p + 1, NULL, 10));
}
static void load(const char *dir) {
    char path[512]; snprintf(path, sizeof path, "%s/model.safetensors", dir);
    int fd = open(path, O_RDONLY); if (fd < 0) { perror(path); exit(1); }
    struct stat st; fstat(fd, &st); model_fd = fd; model_size = st.st_size;
    uint8_t *m = mmap(NULL, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    uint64_t n = *(uint64_t *)m; hdr = malloc(n + 1); memcpy(hdr, m + 8, n); hdr[n] = 0; data = m + 8 + n;
}
// after a tensor is quantized its bf16 pages are dead weight: unmap them and drop them from the page cache so
// the 16 GB file never competes with the quantized weights for RAM (that competition swapped us earlier).
static void drop_pages(const void *p, size_t bytes) {
    uintptr_t a = ((uintptr_t)p) & ~4095ul, e = ((uintptr_t)p + bytes + 4095) & ~4095ul;
    madvise((void *)a, e - a, MADV_DONTNEED);
    off_t off = (const uint8_t *)a - (data - (8 + strlen(hdr)));   // file offset of the page
    posix_fadvise(model_fd, off, e - a, POSIX_FADV_DONTNEED);
}
static float *vecf(const char *name, int n) { const uint16_t *t = tensor(name); float *f = xalloc(n * 4); for (int i = 0; i < n; i++) f[i] = bf(t[i]); return f; }

// 4-bit: w [n/16][k/8][16][4] nibbles = w+8 (u8 operand); low nibble k=8g+i, high nibble k=8g+4+i.
// scale fp16 per (row, 32-k block): [n/16][k/32][16]. Block-32 weight scales + block-32 activation scales.
static void qmat_alloc4(qmat_t *d, int n, int k) { d->n = n; d->k = k; d->w4 = 1; d->w = xalloc((size_t)n * k / 2); d->sc = xalloc((size_t)n * (k / 32) * 2); }
static void quant44(qmat_t *d, int r0, const uint16_t *w, int n, int k) {
    const int G8 = k / 8, NB = k / 32;
    #pragma omp parallel for schedule(static)
    for (int r = 0; r < n; r++) {
        const uint16_t *row = w + (size_t)r * k; int rr = r0 + r;
        for (int blk = 0; blk < NB; blk++) {
            float am = 1e-8f; for (int i = 0; i < 32; i++) am = fmaxf(am, fabsf(bf(row[blk * 32 + i])));
            uint16_t sh = _cvtss_sh(am / 7.0f, 0); d->sc[((size_t)(rr / 16) * NB + blk) * 16 + rr % 16] = sh; float scl = _cvtsh_ss(sh);
            for (int gg = blk * 4; gg < blk * 4 + 4; gg++) {
                uint8_t *dst = d->w + (((size_t)(rr / 16) * G8 + gg) * 16 + rr % 16) * 4;
                for (int i = 0; i < 4; i++) {
                    float a = rintf(bf(row[8 * gg + i]) / scl), c = rintf(bf(row[8 * gg + 4 + i]) / scl);
                    int wa = a < -8 ? -8 : a > 7 ? 7 : (int)a, wc = c < -8 ? -8 : c > 7 ? 7 : (int)c;
                    dst[i] = (uint8_t)((wa + 8) | (wc + 8) << 4);
                }
            }
        }
    }
}
static void qmat_alloc8(qmat_t *d, int n, int k) { d->n = n; d->k = k; d->w4 = 0; d->w = xalloc((size_t)n * k); d->scale = xalloc(n * 4); }
// quantize rows r0.. of d from a bf16 [n][k] tensor: int8 (absmax/127 per row), stored as w+128 so the weight is
// the unsigned vpdpbusd operand and the activation (signed, block-scaled) is the signed one. Blocked layout
// [n/16][k/4][16][4]: one 64-byte load serves 16 rows at once.
static void quant48(qmat_t *d, int r0, const uint16_t *w, int n, int k) {
    const int G4 = k / 4;
    #pragma omp parallel for schedule(static)
    for (int r = 0; r < n; r++) {
        const uint16_t *row = w + (size_t)r * k; int rr = r0 + r;
        float am = 1e-8f; for (int i = 0; i < k; i++) am = fmaxf(am, fabsf(bf(row[i])));
        float scl = am / 127.0f; d->scale[rr] = scl;
        for (int gg = 0; gg < G4; gg++) {
            uint8_t *dst = d->w + (((size_t)(rr / 16) * G4 + gg) * 16 + rr % 16) * 4;
            for (int i = 0; i < 4; i++) {
                float v = rintf(bf(row[4 * gg + i]) / scl);
                int q = v < -127 ? -127 : v > 127 ? 127 : (int)v;
                dst[i] = (uint8_t)(q + 128);
            }
        }
    }
}

static int w4mask = 215;   // default: 4-bit everywhere except down_proj (8) and PLE proj (32), which lose ~5 ppl in 4-bit. bit per matrix kind: 1 qkv, 2 o, 4 gate|up, 8 down, 16 ple gate, 32 ple proj, 64 ple projection (global), 128 embeddings/lm_head
static void qmat_alloc(qmat_t *d, int n, int k, int kind) { if (w4mask & kind) qmat_alloc4(d, n, k); else qmat_alloc8(d, n, k); }
static void quant4(qmat_t *d, int r0, const uint16_t *w, int n, int k) { if (d->w4) quant44(d, r0, w, n, k); else quant48(d, r0, w, n, k); }

// ---------- kernels ----------
static inline __m512 tanh512(__m512 z) {   // 1 - 2 / (1 + exp(2z))
    const __m512 one = _mm512_set1_ps(1.f), two = _mm512_set1_ps(2.f);
    return _mm512_sub_ps(one, _mm512_div_ps(two, _mm512_add_ps(one, exp512(_mm512_mul_ps(two, z)))));
}

// 16-row block nb of out[b] = W @ x[b] for BB streams. Activations are int8 with one scale per ABLK elements
// (Gemma's residual stream has outlier dims; one scale per token would crush everything else to a few levels).
// Per block: dpbusd(w+128 as u8, x as s8) = sum w x + 128 sum x, so subtract 128*sum(x_block) (from prep),
// scale by the activation block scale, accumulate in f32; at the end multiply by the per-row weight scale.
#define ABLK 32
// per 32-k block: 4 loads of 64 B (16 rows x 8 k nibbles), 2 dpbusd each (low/high nibbles) per stream;
// acc = sum (w+8) x = sum w x + 8 sum x -> subtract xc (= 8*sum x_block from prep), times wscale*ascale, accumulate.
static inline __attribute__((always_inline)) void mv4_block4(const qmat_t *m, const int8_t *const *xq, const float *const *as, const float *const *xc, float *const *out, int b0, int nb, const int BB) {
    const int G8 = m->k / 8, NB = m->k / 32; const __m512i m4 = _mm512_set1_epi8(0x0F);
    __m512 facc[BB]; for (int b = 0; b < BB; b++) facc[b] = _mm512_setzero_ps();
    const uint8_t *wb = m->w + (size_t)nb * G8 * 64;
    for (int blk = 0; blk < NB; blk++) {
        __m512i acc[BB]; for (int b = 0; b < BB; b++) acc[b] = _mm512_setzero_si512();
        for (int gg = blk * 4; gg < blk * 4 + 4; gg++) {
            __m512i p = _mm512_loadu_si512(wb + gg * 64);
            __m512i wlo = _mm512_and_si512(p, m4), whi = _mm512_and_si512(_mm512_srli_epi16(p, 4), m4);
            for (int b = 0; b < BB; b++) {
                acc[b] = _mm512_dpbusd_epi32(acc[b], wlo, _mm512_set1_epi32(*(const int32_t *)(xq[b0 + b] + 8 * gg)));
                acc[b] = _mm512_dpbusd_epi32(acc[b], whi, _mm512_set1_epi32(*(const int32_t *)(xq[b0 + b] + 8 * gg + 4)));
            }
        }
        __m512 ws = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)(m->sc + ((size_t)nb * NB + blk) * 16)));
        for (int b = 0; b < BB; b++)
            facc[b] = _mm512_fmadd_ps(_mm512_sub_ps(_mm512_cvtepi32_ps(acc[b]), _mm512_set1_ps(8.0f * xc[b0 + b][blk])), _mm512_mul_ps(ws, _mm512_set1_ps(as[b0 + b][blk])), facc[b]);
    }
    for (int b = 0; b < BB; b++) _mm512_storeu_ps(out[b0 + b] + nb * 16, facc[b]);
}
static inline __attribute__((always_inline)) void mv4_block8(const qmat_t *m, const int8_t *const *xq, const float *const *as, const float *const *xc, float *const *out, int b0, int nb, const int BB) {
    __m512 facc[BB]; for (int b = 0; b < BB; b++) facc[b] = _mm512_setzero_ps();
    const uint8_t *wb = m->w + (size_t)nb * 16 * m->k;
    for (int blk = 0; blk < m->k / ABLK; blk++) {
        __m512i acc[BB]; for (int b = 0; b < BB; b++) acc[b] = _mm512_setzero_si512();
        for (int gg = blk * (ABLK / 4); gg < (blk + 1) * (ABLK / 4); gg++) {
            __m512i w = _mm512_loadu_si512(wb + gg * 64);
            for (int b = 0; b < BB; b++) acc[b] = _mm512_dpbusd_epi32(acc[b], w, _mm512_set1_epi32(*(const int32_t *)(xq[b0 + b] + 4 * gg)));
        }
        for (int b = 0; b < BB; b++)
            facc[b] = _mm512_fmadd_ps(_mm512_sub_ps(_mm512_cvtepi32_ps(acc[b]), _mm512_set1_ps(128.0f * xc[b0 + b][blk])), _mm512_set1_ps(as[b0 + b][blk]), facc[b]);
    }
    __m512 sc = _mm512_loadu_ps(m->scale + nb * 16);
    for (int b = 0; b < BB; b++) _mm512_storeu_ps(out[b0 + b] + nb * 16, _mm512_mul_ps(facc[b], sc));
}

static void mv4(const qmat_t *m, const int8_t *const *xq, const float *const *as, const float *const *xc, float *const *out, int nstreams, int nb0, int nb1) {
    for (int nb = nb0; nb < nb1; nb++)
        for (int b0 = 0; b0 < nstreams; b0 += 8)
            switch (nstreams - b0 >= 8 ? 8 : nstreams - b0) {
#define D(BB) (m->w4 ? mv4_block4(m, xq, as, xc, out, b0, nb, BB) : mv4_block8(m, xq, as, xc, out, b0, nb, BB))
            case 1: D(1); break; case 2: D(2); break; case 3: D(3); break; case 4: D(4); break;
            case 5: D(5); break; case 6: D(6); break; case 7: D(7); break; default: D(8); break;
#undef D
            }
}

typedef struct { int8_t q[FFN]; float as[FFN / ABLK]; float xc[FFN / ABLK]; } act_t;   // int8 per ABLK block: scale, 128*sum
static act_t act[MAXB];
static void prep(act_t *a, const float *xf, int n) {
    for (int blk = 0; blk < n / ABLK; blk++) {
        const float *xb = xf + blk * ABLK; __m512 am = _mm512_set1_ps(1e-8f);
        for (int i = 0; i < ABLK; i += 16) am = _mm512_max_ps(am, _mm512_abs_ps(_mm512_loadu_ps(xb + i)));
        float sc = 127.0f / _mm512_reduce_max_ps(am); __m512 sv = _mm512_set1_ps(sc); __m512i sum = _mm512_setzero_si512();
        for (int i = 0; i < ABLK; i += 16) {
            __m512i q32 = _mm512_cvtps_epi32(_mm512_mul_ps(_mm512_loadu_ps(xb + i), sv)); sum = _mm512_add_epi32(sum, q32);
            _mm_storeu_si128((__m128i *)(a->q + blk * ABLK + i), _mm512_cvtsepi32_epi8(q32));
        }
        a->as[blk] = 1.0f / sc; a->xc[blk] = (float)_mm512_reduce_add_epi32(sum);   // raw block sum; kernels scale by 128 (int8) or 8 (4-bit)
    }
}
// Gemma 4 RMSNorm: x * rsqrt(mean(x^2) + eps) * w   (w = NULL: no scale)
static void rmsnorm(const float *xf, const float *w, float *o, int n) {
    __m512 ss = _mm512_setzero_ps();
    for (int i = 0; i < n; i += 16) { __m512 xv = _mm512_loadu_ps(xf + i); ss = _mm512_fmadd_ps(xv, xv, ss); }
    __m512 r = _mm512_set1_ps(1.0f / sqrtf(_mm512_reduce_add_ps(ss) / n + EPS));
    for (int i = 0; i < n; i += 16) { __m512 v = _mm512_mul_ps(_mm512_loadu_ps(xf + i), r); _mm512_storeu_ps(o + i, w ? _mm512_mul_ps(_mm512_loadu_ps(w + i), v) : v); }
}
static void linear(const qmat_t *m, float *out, int stride, int id) {   // this thread's row shard, all B lanes
    int b0, b1; shard(m->n / 16, id, &b0, &b1);
    const int8_t *xq[MAXB] = {0}; const float *as[MAXB] = {0}, *xc[MAXB] = {0}; float *o[MAXB] = {0};
    for (int b = 0; b < B; b++) { xq[b] = act[b].q; as[b] = act[b].as; xc[b] = act[b].xc; o[b] = out + (size_t)b * stride; }
    mv4(m, xq, as, xc, o, B, b0, b1);
}
static void rope(float *p, int pos, int glob) {   // rotate-half over one head (hd = 256 or 512)
    int half = glob ? 256 : 128; const float *c = rope_c[glob][pos], *s = rope_s[glob][pos];
    for (int i = 0; i < half; i++) { float x1 = p[i], x2 = p[i + half]; p[i] = x1 * c[i] - x2 * s[i]; p[i + half] = x2 * c[i] + x1 * s[i]; }
}
// g = gelu_tanh(g) * u, n multiple of 16
static void gelu_mul(float *gv_, const float *uv_, int n) {
    const __m512 c0 = _mm512_set1_ps(0.79788456f), c1 = _mm512_set1_ps(0.044715f), one = _mm512_set1_ps(1.f), half = _mm512_set1_ps(0.5f);
    for (int i = 0; i < n; i += 16) {
        __m512 gv = _mm512_loadu_ps(gv_ + i);
        __m512 z = _mm512_mul_ps(c0, _mm512_fmadd_ps(_mm512_mul_ps(c1, _mm512_mul_ps(gv, gv)), gv, gv));
        _mm512_storeu_ps(gv_ + i, _mm512_mul_ps(_mm512_mul_ps(_mm512_mul_ps(half, gv), _mm512_add_ps(one, tanh512(z))), _mm512_loadu_ps(uv_ + i)));
    }
}

// ---------- KV cache + attention (int8, head-major token-blocked; hd 256 or 512) ----------
static void kv_alloc(void) {
    kvh = xalloc((size_t)nkv * KV_SHARED_FROM * NKV * sizeof *kvh);
    for (int s = 0; s < nkv; s++) for (int l = 0; l < KV_SHARED_FROM; l++) for (int h = 0; h < NKV; h++) {
        kvh_t *c = &kvh[((size_t)s * KV_SHARED_FROM + l) * NKV + h]; c->hd = HDIM(l);
        c->k8 = xalloc((size_t)MAXCTX * c->hd); c->v8 = xalloc((size_t)MAXCTX * c->hd);
        c->ksc = xalloc(MAXCTX * 4); c->vsc = xalloc(MAXCTX * 4); c->ksum = xalloc(MAXCTX * 4);
    }
}
static float quant_vec(const float *xf, int8_t *qo, int n) {   // absmax int8
    float am = 1e-8f; for (int d = 0; d < n; d++) am = fmaxf(am, fabsf(xf[d]));
    float s = 127.0f / am; for (int d = 0; d < n; d++) { float t = rintf(xf[d] * s); qo[d] = (int8_t)(t < -128 ? -128 : t > 127 ? 127 : t); }
    return 1.0f / s;
}
static void kv_insert(kvh_t *c, int pos, const float *kf, const float *vf) {
    int hd = c->hd; int8_t k8[512], v8[512];
    c->ksc[pos] = quant_vec(kf, k8, hd); c->vsc[pos] = quant_vec(vf, v8, hd);
    int32_t sum = 0; for (int d = 0; d < hd; d++) sum += k8[d]; c->ksum[pos] = sum;
    int8_t *kb = c->k8 + (size_t)(pos / 16) * hd * 16 + (pos % 16) * 4;
    for (int gg = 0; gg < hd / 4; gg++) memcpy(kb + gg * 64, k8 + 4 * gg, 4);
    int8_t *vb = c->v8 + (size_t)(pos / 4) * hd * 4 + pos % 4;
    for (int d = 0; d < hd; d++) vb[d * 4] = v8[d];
}
// one (lane, layer, q head). Sliding layers see [pos-WINDOW+1, pos]; global layers everything. scaling = 1.0
static void attention_head(int b, int l, int h, int pos, float *out) {
    const kvh_t *c = KVH(b, l, h / (NH / NKV)); const int hd = c->hd;
    int8_t q8[512]; uint8_t qu[512];
    float qs = quant_vec(qkv[b] + h * hd, q8, hd);
    for (int d = 0; d < hd; d++) qu[d] = q8[d] + 128;
    const int32_t *q4 = (const int32_t *)qu;
    int t0 = IS_GLOBAL(l) || pos < WINDOW ? 0 : pos - WINDOW + 1, tb0 = t0 / 16, nblk = pos / 16 + 1;
    float sc[MAXCTX]; __m512 mxv = _mm512_set1_ps(-1e30f);
    for (int tb = tb0; tb < nblk; tb++) {
        __m512i acc = _mm512_setzero_si512(); const int8_t *kb = c->k8 + (size_t)tb * hd * 16;
        for (int gg = 0; gg < hd / 4; gg++) acc = _mm512_dpbusd_epi32(acc, _mm512_set1_epi32(q4[gg]), _mm512_loadu_si512(kb + gg * 64));
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
    for (int d0 = 0; d0 < hd; d0 += 256) {   // P.V in 256-dim chunks (16 accumulators)
        __m512i acc[16]; for (int j = 0; j < 16; j++) acc[j] = _mm512_setzero_si512();
        for (int vb = tb0 * 4; vb <= pos / 4; vb++) {
            __m512i pb = _mm512_set1_epi32(*(const int32_t *)(pu + 4 * vb)); const int8_t *vr = c->v8 + (size_t)vb * hd * 4 + d0 * 4;
            for (int j = 0; j < 16; j++) acc[j] = _mm512_dpbusd_epi32(acc[j], pb, _mm512_loadu_si512(vr + j * 64));
        }
        for (int j = 0; j < 16; j++) _mm512_storeu_ps(out + h * hd + d0 + j * 16, _mm512_mul_ps(_mm512_cvtepi32_ps(acc[j]), _mm512_set1_ps(inv)));
    }
}

// ---------- forward: one step for all B lanes ----------
static void forward_mt(int id) {
    static __thread float h[FFN];
    int b0, b1;
    // embeddings (scaled by sqrt(HID)) and the token-identity half of the per-layer inputs
    for (int b = id; b < B; b += T) {
        const uint16_t *e = emb_bf16 + (size_t)cur_tok[b] * HID; for (int i = 0; i < HID; i++) x[b][i] = bf(e[i]) * 50.596443f;
        P(1, prep(&act[b], x[b], HID));
    }
    sbar_sync(&bar);
    P(0, linear(&pleproj, &plep[0][0], NLAYER * PLE, id));   // context half: per_layer_model_projection(x)
    sbar_sync(&bar);
    for (int b = id; b < B; b += T) {
        const uint16_t *pe = ple_bf16 + (size_t)cur_tok[b] * NLAYER * PLE;
        for (int l = 0; l < NLAYER; l++) {
            float t[PLE]; for (int i = 0; i < PLE; i++) t[i] = plep[b][l * PLE + i] * 0.019764235f;   // HID^-0.5
            rmsnorm(t, ln_pleproj, t, PLE);
            for (int i = 0; i < PLE; i++) ple[b][l][i] = (t[i] + bf(pe[l * PLE + i]) * 16.0f) * 0.70710678f;   // (proj + emb*sqrt(256)) / sqrt(2)
        }
    }
    sbar_sync(&bar);
    static int chk, chklane; if (!id && !chk) { chk = getenv("CHECK") != NULL; chklane = getenv("CHECKLANE") ? atoi(getenv("CHECKLANE")) : 0; if (chklane >= B) chklane = B - 1; }
    int cl = chklane;
    if (!id && chk) { double nn = 0; for (int i = 0; i < HID; i++) nn += (double)x[cl][i] * x[cl][i];
        fprintf(stderr, "  embed x[%d] pos %d: %.5f %.5f %.5f %.5f |x|=%.4f\n", cl, cur_pos[cl], x[cl][0], x[cl][1], x[cl][2], x[cl][3], sqrt(nn));
        fprintf(stderr, "  ple[%d][0] (layer0 PLE):  %.5f %.5f %.5f %.5f\n", cl, ple[cl][0][0], ple[cl][0][1], ple[cl][0][2], ple[cl][0][3]); }
    for (int l = 0; l < NLAYER; l++) {
        layer_t *ly = &L[l]; const int glob = IS_GLOBAL(l), hd = HDIM(l), qd = QD(l), shared = l >= KV_SHARED_FROM;
        for (int b = id; b < B; b += T) { P(3, rmsnorm(x[b], ly->ln_in, h, HID)); P(1, prep(&act[b], h, HID)); }
        sbar_sync(&bar);
        P(0, linear(&ly->qkv, &qkv[0][0], QKV_MAX, id));
        sbar_sync(&bar);
        shard(B * NH, id, &b0, &b1);
        for (int i = b0; i < b1; i++) {   // QK-norm + RoPE on q; k/v norm + RoPE + insert by the units with hh % (NH/NKV) == 0
            int b = i / NH, hh = i % NH, pos = cur_pos[b]; float *qh = qkv[b] + hh * hd;
            P(3, rmsnorm(qh, ly->qn, qh, hd); rope(qh, pos, glob));
            if (!shared && hh % (NH / NKV) == 0) {
                int kvh_ = hh / (NH / NKV); float *kh = qkv[b] + qd + kvh_ * hd, *vh = qkv[b] + qd + NKV * hd + kvh_ * hd;
                P(3, rmsnorm(kh, ly->kn, kh, hd); rope(kh, pos, glob); rmsnorm(vh, NULL, vh, hd));
                kv_insert(KVH(b, l, kvh_), pos, kh, vh);
            }
        }
        sbar_sync(&bar);
        for (int i = b0; i < b1; i++) { int b = i / NH, hh = i % NH; P(2, attention_head(b, l, hh, cur_pos[b], att[b])); }
        sbar_sync(&bar);
        for (int b = id; b < B; b += T) P(1, prep(&act[b], att[b], qd));
        sbar_sync(&bar);
        P(0, linear(&ly->o, &tmp[0][0], HID, id));
        sbar_sync(&bar);
        for (int b = id; b < B; b += T) {   // x += post_attn_norm(o); h = pre_ffn_norm(x)
            P(3, rmsnorm(tmp[b], ly->ln_post_attn, h, HID)); for (int i = 0; i < HID; i++) x[b][i] += h[i];
            P(3, rmsnorm(x[b], ly->ln_pre_ffn, h, HID)); P(1, prep(&act[b], h, HID));
        }
        sbar_sync(&bar);
        P(0, linear(&ly->gu, &gu[0][0], 2 * FFN, id));
        sbar_sync(&bar);
        shard(FFN / 16, id, &b0, &b1);
        P(4, for (int b = 0; b < B; b++) gelu_mul(gu[b] + b0 * 16, gu[b] + FFN + b0 * 16, (b1 - b0) * 16));
        sbar_sync(&bar);
        for (int b = id; b < B; b += T) P(1, prep(&act[b], gu[b], FFN));
        sbar_sync(&bar);
        P(0, linear(&ly->down, &tmp[0][0], HID, id));
        sbar_sync(&bar);
        for (int b = id; b < B; b += T) {   // x += post_ffn_norm(down); quantize x for the PLE gate
            P(3, rmsnorm(tmp[b], ly->ln_post_ffn, h, HID)); for (int i = 0; i < HID; i++) x[b][i] += h[i];
            P(1, prep(&act[b], x[b], HID));
        }
        sbar_sync(&bar);
        P(0, linear(&ly->pgate, &pg[0][0], PLE, id));
        sbar_sync(&bar);
        for (int b = id; b < B; b += T) { P(4, gelu_mul(pg[b], ple[b][l], PLE)); P(1, prep(&act[b], pg[b], PLE)); }
        sbar_sync(&bar);
        P(0, linear(&ly->pproj, &tmp[0][0], HID, id));
        sbar_sync(&bar);
        for (int b = id; b < B; b += T) {   // x = (x + ple_norm(proj)) * layer_scalar
            P(3, rmsnorm(tmp[b], ly->ln_ple, h, HID)); for (int i = 0; i < HID; i++) x[b][i] = (x[b][i] + h[i]) * ly->scalar;
        }
        sbar_sync(&bar);
        if (!id && chk && l < 3) { double nn = 0; for (int i = 0; i < HID; i++) nn += (double)x[cl][i] * x[cl][i];
            fprintf(stderr, "  h[%d] pos %d: %.5f %.5f %.5f %.5f |x|=%.4f\n", l + 1, cur_pos[cl], x[cl][0], x[cl][1], x[cl][2], x[cl][3], sqrt(nn)); }
    }
    for (int b = id; b < B; b += T) { P(3, rmsnorm(x[b], ln_final, h, HID)); P(1, prep(&act[b], h, HID)); }
    sbar_sync(&bar);
    shard(VOCAB / 16, id, &b0, &b1);
    { const int8_t *xq[MAXB] = {0}; const float *as[MAXB] = {0}, *xc[MAXB] = {0}; float *o[MAXB] = {0};
      for (int b = 0; b < B; b++) { xq[b] = act[b].q; as[b] = act[b].as; xc[b] = act[b].xc; o[b] = logits + (size_t)b * VOCAB; }
      P(5, mv4(&emb4, xq, as, xc, o, B, b0, b1)); }
    for (int b = 0; b < B; b++) {   // softcap this shard, then local argmax
        float *lg = logits + (size_t)b * VOCAB; const __m512 cap = _mm512_set1_ps(SOFTCAP), icap = _mm512_set1_ps(1.0f / SOFTCAP);
        for (int vv = b0 * 16; vv < b1 * 16; vv += 16) _mm512_storeu_ps(lg + vv, _mm512_mul_ps(cap, tanh512(_mm512_mul_ps(_mm512_loadu_ps(lg + vv), icap))));
        int best = b0 * 16; for (int vv = b0 * 16 + 1; vv < b1 * 16; vv++) if (lg[vv] > lg[best]) best = vv; best_t[b][id] = best;
    }
    sbar_sync(&bar);
    if (!id) for (int b = 0; b < B; b++) { const float *lg = logits + (size_t)b * VOCAB; for (int i = 1; i < T; i++) if (lg[best_t[b][i]] > lg[best_t[b][0]]) best_t[b][0] = best_t[b][i]; }
}

// ---------- driver: threads, sampling, speculation, prefill (same protocol as the other engines) ----------
static void step(const int *tok, const int *pos, int *next) {
    for (int b = 0; b < B; b++) { cur_tok[b] = tok[b]; cur_pos[b] = pos[b]; }
    sbar_sync(&bar); forward_mt(0); sbar_sync(&bar);
    for (int b = 0; b < B; b++) next[b] = best_t[b][0];
}
static inline int is_eos(int t) { return t == 1 || t == 106; }   // <eos>, <turn|>

static void generate(const int *prompt, int np, int ngen) {
    int tok[MAXB], posv[MAXB], next[MAXB]; double t0;
    int K = getenv("SPEC") ? atoi(getenv("SPEC")) : 0; if (K > MAXB - 1) K = MAXB - 1;
    static int hist[MAXCTX]; int n = 0;
    for (int i = 0; i < np; i++) hist[n++] = prompt[i];
    const int nstreams = B;
    t0 = now_ns(); double nll = 0;
    for (int i0 = 0; i0 < np; i0 += MAXB / nstreams) {   // batched prefill
        int cnt = np - i0 < MAXB / nstreams ? np - i0 : MAXB / nstreams; B = cnt * nstreams;
        for (int b = 0; b < B; b++) { kv_map[b] = b / cnt; tok[b] = hist[i0 + b % cnt]; posv[b] = i0 + b % cnt; }
        step(tok, posv, next);
        if (getenv("PPL")) for (int j = 0; j < cnt && i0 + j + 1 < np; j++) {
            const float *lg = logits + (size_t)j * VOCAB; float mx = -1e30f; for (int vv = 0; vv < VOCAB; vv++) if (lg[vv] > mx) mx = lg[vv];
            double z = 0; for (int vv = 0; vv < VOCAB; vv++) z += exp(lg[vv] - mx);
            nll -= lg[hist[i0 + j + 1]] - mx - log(z);
        }
        for (int b = 0; b < nstreams; b++) { int last = (b + 1) * cnt - 1;   // last lane of each stream holds the next-token logits
            next[b] = next[last]; if (last != b) memcpy(logits + (size_t)b * VOCAB, logits + (size_t)last * VOCAB, VOCAB * 4); }
    }
    double tp = now_ns() - t0;
    if (getenv("PPL")) fprintf(stderr, "perplexity over %d tokens: %.2f\n", np - 1, exp(nll / (np - 1)));
    { const float *lg = logits; fprintf(stderr, "top5:"); static float tmpl[VOCAB]; memcpy(tmpl, lg, sizeof tmpl);
      for (int i = 0; i < 5; i++) { int bb = 0; for (int vv = 1; vv < VOCAB; vv++) if (tmpl[vv] > tmpl[bb]) bb = vv; fprintf(stderr, " %d(%.3f)", bb, tmpl[bb]); tmpl[bb] = -1e30f; } fprintf(stderr, "\n"); }
    t0 = now_ns();
    int gen = 0, same = 1, steps = 0, drafted = 0, accepted = 0;
    B = nstreams; for (int b = 0; b < B; b++) kv_map[b] = b;
    if (K && nstreams == 1 && temp <= 0 && rep <= 1.0f) {
        int cur = next[0];
        while (gen < ngen && n < MAXCTX - K - 1) {
            printf("%d\n", cur); fflush(stdout); hist[n++] = cur; gen++; if (is_eos(cur)) break;
            int d[MAXB], nd = draft(hist, n, K, d); drafted += nd;
            B = 1 + nd; tok[0] = cur; posv[0] = n - 1;
            for (int j = 0; j < nd; j++) { tok[1 + j] = d[j]; posv[1 + j] = n + j; kv_map[1 + j] = 0; }
            step(tok, posv, next); steps++;
            int acc = 0; while (acc < nd && next[acc] == d[acc]) acc++;
            accepted += acc;
            for (int j = 0; j < acc && gen < ngen; j++) { printf("%d\n", d[j]); fflush(stdout); hist[n++] = d[j]; gen++; }
            cur = next[acc]; B = 1;
        }
    } else {
        for (int b = 0; b < B; b++) tok[b] = sample(logits + (size_t)b * VOCAB, next[b], hist, n);
        for (; gen < ngen && n < MAXCTX; gen++) {
            printf("%d\n", tok[0]); fflush(stdout);
            for (int b = 1; b < B; b++) if (tok[b] != tok[0]) same = 0;
            if (is_eos(tok[0])) { gen++; break; }
            hist[n] = tok[0];
            for (int b = 0; b < B; b++) posv[b] = n;
            step(tok, posv, next); steps++; n++;
            for (int b = 0; b < B; b++) tok[b] = sample(logits + (size_t)b * VOCAB, next[b], hist, n);
        }
    }
    double tg = now_ns() - t0;
    if (getenv("PROF")) for (int i = 0; i < 6; i++) fprintf(stderr, "  %-10s %7.2f ms/step (thread 0)\n", pname[i], prof[i] / 1e6 / (steps + (np + MAXB - 1) / MAXB));
    if (nstreams > 1) fprintf(stderr, "all %d streams identical: %s\n", nstreams, same ? "yes" : "NO");
    if (K && nstreams == 1) fprintf(stderr, "speculation K=%d: %d steps, %d drafted, %d accepted (%.0f%%), %.2f tokens/step\n", K, steps, drafted, accepted, drafted ? 100.0 * accepted / drafted : 0, (double)gen / (steps ? steps : 1));
    fprintf(stderr, "prefill %d tok in %.1f ms (%.0f tok/s), decode %d tok x %d streams in %.1f ms: %.1f tok/s per stream, %.0f tok/s aggregate\n",
            np, tp / 1e6, np / (tp / 1e9), gen, nstreams, tg / 1e6, gen / (tg / 1e9), (double)gen * nstreams / (tg / 1e9));
}

int main(int argc, char **argv) {
    if (argc < 4 && !(argc == 3 && !strcmp(argv[2], "-"))) { fprintf(stderr, "usage: %s model_dir n_gen tok...  |  %s model_dir -\n", argv[0], argv[0]); return 1; }
    int nc = find_cores(cpus, 256);
    T = getenv("THREADS") ? atoi(getenv("THREADS")) : nc; if (T > nc) for (int i = 0; i < T && i < 256; i++) cpus[i] = i;
    B = getenv("BATCH") ? atoi(getenv("BATCH")) : 1; if (B > MAXB) B = MAXB;
    if (getenv("TEMP")) temp = atof(getenv("TEMP"));
    if (getenv("REP")) rep = atof(getenv("REP"));
    if (getenv("TOPK")) topk = atoi(getenv("TOPK"));
    if (getenv("W4MASK")) w4mask = atoi(getenv("W4MASK"));
    if (getenv("SEED")) srng = strtoull(getenv("SEED"), NULL, 10) * 2654435761ull + 1;
    nkv = B; kv_alloc(); logits = xalloc((size_t)MAXB * VOCAB * 4);
    double t0 = now_ns();
    load(argv[1]);
    char nm[200];
#define TN(t) (snprintf(nm, sizeof nm, "model.language_model.layers.%d." t ".weight", l), tensor(nm))
#define VN(t, n) (snprintf(nm, sizeof nm, "model.language_model.layers.%d." t ".weight", l), vecf(nm, n))
    for (int l = 0; l < NLAYER; l++) {
        layer_t *ly = &L[l]; int hd = HDIM(l), qd = QD(l), kvd = KVD(l), shared = l >= KV_SHARED_FROM;
        ly->nqkv = shared ? qd : qd + 2 * kvd;
        qmat_alloc(&ly->qkv, ly->nqkv, HID, 1); quant4(&ly->qkv, 0, TN("self_attn.q_proj"), qd, HID);
        if (!shared) { quant4(&ly->qkv, qd, TN("self_attn.k_proj"), kvd, HID); quant4(&ly->qkv, qd + kvd, TN("self_attn.v_proj"), kvd, HID); ly->kn = VN("self_attn.k_norm", hd); }
        qmat_alloc(&ly->o, HID, qd, 2); quant4(&ly->o, 0, TN("self_attn.o_proj"), HID, qd);
        qmat_alloc(&ly->gu, 2 * FFN, HID, 4); quant4(&ly->gu, 0, TN("mlp.gate_proj"), FFN, HID); quant4(&ly->gu, FFN, TN("mlp.up_proj"), FFN, HID);
        qmat_alloc(&ly->down, HID, FFN, 8); quant4(&ly->down, 0, TN("mlp.down_proj"), HID, FFN);
        qmat_alloc(&ly->pgate, PLE, HID, 16); quant4(&ly->pgate, 0, TN("per_layer_input_gate"), PLE, HID);
        qmat_alloc(&ly->pproj, HID, PLE, 32); quant4(&ly->pproj, 0, TN("per_layer_projection"), HID, PLE);
        { const uint16_t *a = TN("input_layernorm"), *z = TN("self_attn.v_proj");   // alphabetically first and last tensor of this layer: the layer is one contiguous file range
          drop_pages(a, (size_t)((const uint8_t *)z - (const uint8_t *)a) + (size_t)kvd * HID * 2); }
        ly->ln_in = VN("input_layernorm", HID); ly->ln_post_attn = VN("post_attention_layernorm", HID);
        ly->ln_pre_ffn = VN("pre_feedforward_layernorm", HID); ly->ln_post_ffn = VN("post_feedforward_layernorm", HID);
        ly->ln_ple = VN("post_per_layer_input_norm", HID); ly->qn = VN("self_attn.q_norm", hd);
        snprintf(nm, sizeof nm, "model.language_model.layers.%d.layer_scalar", l); ly->scalar = bf(*tensor(nm));
    }
    emb_bf16 = tensor("model.language_model.embed_tokens.weight"); ple_bf16 = tensor("model.language_model.embed_tokens_per_layer.weight");
    ln_final = vecf("model.language_model.norm.weight", HID); ln_pleproj = vecf("model.language_model.per_layer_projection_norm.weight", PLE);
    qmat_alloc(&pleproj, NLAYER * PLE, HID, 64); quant4(&pleproj, 0, tensor("model.language_model.per_layer_model_projection.weight"), NLAYER * PLE, HID);
    qmat_alloc(&emb4, VOCAB, HID, 128); quant4(&emb4, 0, emb_bf16, VOCAB, HID);
    drop_pages(tensor("model.language_model.per_layer_model_projection.weight"), (size_t)NLAYER * PLE * HID * 2);
    posix_fadvise(model_fd, 0, 0, POSIX_FADV_DONTNEED);   // and whatever else of the 16 GB file is still cached (vision/audio towers, read-ahead)
    for (int p = 0; p < MAXCTX; p++) {
        for (int i = 0; i < 128; i++) { float ang = p * powf(10000.0f, -2.0f * i / 256); rope_c[0][p][i] = cosf(ang); rope_s[0][p][i] = sinf(ang); }
        for (int i = 0; i < 256; i++) { float ang = i < 64 ? p * powf(1000000.0f, -2.0f * i / 512) : 0.f; rope_c[1][p][i] = cosf(ang); rope_s[1][p][i] = sinf(ang); }
    }
    { FILE *f = fopen("/proc/self/status", "r"); char ln[256]; long rss = 0; while (f && fgets(ln, sizeof ln, f)) if (sscanf(ln, "VmRSS: %ld", &rss) == 1) break; if (f) fclose(f);
      fprintf(stderr, "loaded gemma-4-E4B-it in %.1f s, %d threads, batch %d, w4mask %d, RSS %.1f GB\n", (now_ns() - t0) / 1e9, T, B, w4mask, rss / 1048576.0); }

    sbar_init(&bar, T); pthread_t th[256]; pin(cpus[0]);
    for (long i = 1; i < T; i++) pthread_create(&th[i], NULL, worker, (void *)i);
    if (argc == 3) {
        char line[65536]; static int ids[MAXCTX]; fprintf(stderr, "ready\n");
        while (fgets(line, sizeof line, stdin)) {
            char *p = line; int ngen = (int)strtol(p, &p, 10), np = 0;
            while (np < MAXCTX - 1) { char *e; long vv = strtol(p, &e, 10); if (e == p) break; ids[np++] = (int)vv; p = e; }
            if (np) generate(ids, np, ngen);
            printf("END\n"); fflush(stdout);
        }
    } else {
        static int ids[MAXCTX]; int np = argc - 3; if (np > MAXCTX - 1) np = MAXCTX - 1;
        for (int i = 0; i < np; i++) ids[i] = atoi(argv[3 + i]);
        generate(ids, np, atoi(argv[2]));
    }
    quit = 1; sbar_sync(&bar); for (int i = 1; i < T; i++) pthread_join(th[i], NULL);
    return 0;
}
