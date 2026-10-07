// BitNet b1.58 729M (1bitLLM/bitnet_b1_58-large) end to end on the ternary LUT kernel.
// Llama arch + inner_attn_ln (before o_proj) + ffn_layernorm (before down_proj).
// Weights: per-tensor ternary, scale = mean|w|, packed 5 trits per byte (1.6 bits/weight, 20% fewer
// bytes than 2-bit). Activations: per-token absmax int8. The activation vector is permuted once per
// token so each decoded trit plane is directly a vpdpbusd operand: no transpose, no LUT.
// lm_head: tied embeddings in 4-bit (LM8=1 for int8).
// Threads: every matvec, attention head and lm_head row is sharded across T pinned
// cores with a spin barrier (4 per layer). Small ops (norm, quant, LUT build) are
// redundant per thread: cheaper than a barrier.
// KV cache: int8, head-major, token-blocked (K [t/16][d/4][16][4], V [t/4][d][4]) so attention is
// pure vpdpbusd. Next layer's KV for this thread's heads is prefetched into L2 during the MLP matvec.
// Batch: BATCH=B runs B lockstep streams (same prompt) and every weight byte is read once
// per step for all B. In the DRAM-bound regime that is ~B x aggregate throughput.
// usage: THREADS=6 BATCH=4 ./bitnet models/bitnet-large N_GEN tok tok ...  -> stream 0 ids on stdout, timing on stderr.

#define HID 1536
#define FFN 4096
#define NH 16
#define HD 96
#define NLAYER 24
#define VOCAB 32002
#define MAXCTX 256
#ifndef MAXB
#define MAXB 32
#endif
#define EPS 1e-5f
#define VPAD ((VOCAB + 15) / 16 * 16)

#include "common.h"
#include "ternary.h"

// kp = k padded to a multiple of 20. Layout [n/16][kp/20][16 rows][4 bytes]; byte (quad qd, pos bp)
// of row r holds trits for k = (qd*4+bp)*5 + i, i=0..4, as sum trit_i * 3^i, trit = w+1.
static int16_t perm_h[KPAD(HID)], perm_f[KPAD(FFN)];   // xperm[j] = x[perm[j]], -1 = zero pad
typedef struct { lmat_t qkv, o, gu, down; const float *ln_in, *ln_post, *ln_inner, *ln_ffn; } layer_t;   // qkv: 3*HID rows fused, gu: gate;up fused

static layer_t L[NLAYER];
static const float *embed, *ln_final;
static int8_t *e8; static float *escale; static int32_t *esum; static int lm8;   // int8 lm_head: e8 [v/16][HID/4][16][4], per-row scale
static uint8_t *e4; static uint16_t *e4s;   // 4-bit lm_head: e4 [v/16][HID/8][16][8 nibbles] holding w+8; e4s fp16 scale per (row, 32-k block): [v/16][HID/32][16]
#define NBLK (HID / 32)
typedef struct { int8_t *k8, *v8; float *ksc, *vsc; int32_t *ksum; } kvh_t;   // one (stream, layer, head)
static kvh_t *kvh;                                                           // [B][NLAYER][NH]
static int kv_map[MAXB], nkv = 1;                                           // lane b uses KV stream kv_map[b]
#define KVH(b, l, h) (&kvh[((size_t)kv_map[b] * NLAYER + (l)) * NH + (h)])
static int prefetch_on = 1, nlayers_run = NLAYER;
static float rope_c[MAXCTX][HD / 2], rope_s[MAXCTX][HD / 2];

// shared activations (written in shards, read whole after a barrier)
static float x[MAXB][HID], qkv[MAXB][3 * HID], att[MAXB][HID], gu[MAXB][2 * FFN], logits[MAXB][VPAD];
#define q(b) (qkv[b])
#define k(b) (qkv[b] + HID)
#define v(b) (qkv[b] + 2 * HID)
#define g(b) (gu[b])
#define u(b) (gu[b] + FFN)   // VPAD: lm_head writes whole 16-row blocks
static double prof[8]; static const char *pname[6] = {"matvec", "quant+perm", "attention", "rope", "norm/silu", "lm_head"};
#define P(i, stmt) do { double t_ = now_ns(); stmt; if (!id) prof[i] += now_ns() - t_; } while (0)

// ---------- safetensors ----------
static char *hdr; static uint8_t *data;
static const float *tensor(const char *name) {
    char key[256]; snprintf(key, sizeof key, "\"%s\"", name);
    char *p = strstr(hdr, key); if (!p) { fprintf(stderr, "missing tensor %s\n", name); exit(1); }
    p = strstr(p, "\"data_offsets\""); p = strchr(p, '[');
    return (const float *)(data + strtoul(p + 1, NULL, 10));
}
static void load(const char *dir) {
    char path[512]; snprintf(path, sizeof path, "%s/model.safetensors", dir);
    int fd = open(path, O_RDONLY); if (fd < 0) { perror(path); exit(1); }
    struct stat st; fstat(fd, &st);
    uint8_t *m = mmap(NULL, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    uint64_t n = *(uint64_t *)m;
    hdr = malloc(n + 1); memcpy(hdr, m + 8, n); hdr[n] = 0;
    data = m + 8 + n;
}

// ternarize: s = 1/mean|w|, t = clamp(rint(w*s), -1, 1). Pack into LUT layout.
static void lmat_alloc(lmat_t *d, int n, int k) {
    d->n = n; d->k = k; d->kp = KPAD(k);
    d->w = xalloc((size_t)n * (d->kp / 20) * 4); d->rs = xalloc(n * 4); d->scale = xalloc(n / 16 * 4);
}
// ternarize one tensor (n rows x k) into rows r0.. of d. r0 multiple of 16. Scale = mean|w| per tensor.
static void ternarize(lmat_t *d, int r0, const float *w, int n, int k) {
    double sum = 0; for (long i = 0; i < (long)n * k; i++) sum += fabsf(w[i]);
    float mean = sum / ((double)n * k), s = 1.0f / fmaxf(mean, 1e-5f);
    for (int nb = r0 / 16; nb < (r0 + n) / 16; nb++) d->scale[nb] = 1.0f / s;
    int Q = d->kp / 20;
    for (int r = 0; r < n; r++) {
        int32_t rs = 0, rr = r0 + r;
        for (int qd = 0; qd < Q; qd++)
            for (int bp = 0; bp < 4; bp++) {
                int v = 0, p = 1;
                for (int i = 0; i < 4 + 1; i++, p *= 3) {
                    int kk = (qd * 4 + bp) * 5 + i, t = 0;
                    if (kk < k) { float f = rintf(w[(long)r * k + kk] * s); t = f > 0 ? 1 : f < 0 ? -1 : 0; }
                    rs += t; v += (t + 1) * p;
                }
                d->w[(((size_t)(rr / 16) * Q + qd) * 16 + rr % 16) * 4 + bp] = (uint8_t)v;
            }
        d->rs[rr] = rs;
    }
}

// per-token absmax int8. returns dequant factor (absmax/127).
static float quant(const float *xf, int n, int8_t *qo, uint8_t *qu) {
    __m512 am = _mm512_set1_ps(1e-5f);
    for (int i = 0; i < n; i += 16) am = _mm512_max_ps(am, _mm512_abs_ps(_mm512_loadu_ps(xf + i)));
    float amax = _mm512_reduce_max_ps(am), sc = 127.0f / amax;
    __m512 sv = _mm512_set1_ps(sc); __m128i x80 = _mm_set1_epi8((char)0x80);
    for (int i = 0; i < n; i += 16) {
        __m128i qv = _mm512_cvtsepi32_epi8(_mm512_cvtps_epi32(_mm512_mul_ps(_mm512_loadu_ps(xf + i), sv)));   // rint + saturate
        _mm_storeu_si128((__m128i *)(qo + i), qv);
        if (qu) _mm_storeu_si128((__m128i *)(qu + i), _mm_xor_si128(qv, x80));
    }
    return 1.0f / sc;
}

static void rmsnorm(const float *xf, const float *w, float *o, int n) {
    __m512 ss = _mm512_setzero_ps();
    for (int i = 0; i < n; i += 16) { __m512 xv = _mm512_loadu_ps(xf + i); ss = _mm512_fmadd_ps(xv, xv, ss); }
    __m512 r = _mm512_set1_ps(1.0f / sqrtf(_mm512_reduce_add_ps(ss) / n + EPS));
    for (int i = 0; i < n; i += 16) _mm512_storeu_ps(o + i, _mm512_mul_ps(_mm512_loadu_ps(w + i), _mm512_mul_ps(_mm512_loadu_ps(xf + i), r)));
}

typedef struct { int8_t q[FFN]; uint8_t qu[FFN]; uint8_t xp[KPAD(FFN)]; float dq; int32_t y[MAXB][2 * FFN]; } act_t;   // y sized for the fused gate|up output
static act_t act[MAXB];   // shared: stream b's quantized input + tables; y[id] is thread id's output scratch
static float sparse_h = 0, sparse_f = 0;
static void prep(act_t *a, const float *xf, int n) {
    a->dq = quant(xf, n, a->q, a->qu);
    float frac = n == HID ? sparse_h : sparse_f;
    if (frac > 0) {   // histogram of |q| (int8, 128 bins) -> threshold at the frac quantile, zero below it
        if (getenv("SPARSE_GRP")) {   // structured: whole groups of GS consecutive k (default 4), ranked by sum |q|
            const int GS = atoi(getenv("SPARSE_GRP")) > 1 ? atoi(getenv("SPARSE_GRP")) : 4;
            int hist[4096] = {0}, G4 = n / GS; for (int gg = 0; gg < G4; gg++) { int sm = 0; for (int i = 0; i < GS; i++) sm += abs(a->q[GS * gg + i]); hist[sm > 4095 ? 4095 : sm]++; }
            int cnt = 0, thr = 0; while (thr < 4095 && cnt + hist[thr] <= frac * G4) cnt += hist[thr++];
            for (int gg = 0; gg < G4; gg++) { int sm = 0; for (int i = 0; i < GS; i++) sm += abs(a->q[GS * gg + i]); if (sm < thr) for (int i = 0; i < GS; i++) { a->q[GS * gg + i] = 0; a->qu[GS * gg + i] = 128; } }
        } else {
            int hist[128] = {0}; for (int i = 0; i < n; i++) hist[abs(a->q[i])]++;
            int cnt = 0, thr = 0; while (thr < 127 && cnt + hist[thr] <= frac * n) cnt += hist[thr++];
            for (int i = 0; i < n; i++) if (abs(a->q[i]) < thr) { a->q[i] = 0; a->qu[i] = 128; }
        }
    }
    const int16_t *p = n == HID ? perm_h : perm_f;
    for (int j = 0; j < KPAD(n); j++) a->xp[j] = p[j] < 0 ? 128 : a->qu[p[j]];
}

// this thread's row shard of out[b] = m @ x[b] for all B streams (or += if accumulate)
static void linear(act_t *a, const lmat_t *m, float *out, int stride, int id, int accumulate) {
    int b0, b1; shard(m->n / 16, id, &b0, &b1);
    const uint8_t *xp[MAXB] = {0}; int32_t *y[MAXB] = {0};
    for (int b = 0; b < B; b++) { xp[b] = a[b].xp; y[b] = a[b].y[id]; }
    mv_b(m, xp, y, B, b0, b1);
    for (int b = 0; b < B; b++) {
        float *o = out + (size_t)b * stride;
        for (int nb = b0; nb < b1; nb++) {
            float f = a[b].dq * m->scale[nb];
            if (accumulate) for (int i = nb * 16; i < nb * 16 + 16; i++) o[i] += y[b][i] * f;
            else            for (int i = nb * 16; i < nb * 16 + 16; i++) o[i] = y[b][i] * f;
        }
    }
}

static void rope_head(float *vv, int h, int pos) {   // rotate-half
    float *p = vv + h * HD;
    for (int i = 0; i < HD / 2; i++) {
        float c = rope_c[pos][i], s = rope_s[pos][i], x1 = p[i], x2 = p[i + HD / 2];
        p[i] = x1 * c - x2 * s; p[i + HD / 2] = x2 * c + x1 * s;
    }
}

static void kv_alloc(void) {
    kvh = xalloc((size_t)nkv * NLAYER * NH * sizeof *kvh);
    for (size_t i = 0; i < (size_t)nkv * NLAYER * NH; i++) {
        kvh[i].k8 = xalloc(MAXCTX * HD); kvh[i].v8 = xalloc(MAXCTX * HD);
        kvh[i].ksc = xalloc(MAXCTX * 4); kvh[i].vsc = xalloc(MAXCTX * 4); kvh[i].ksum = xalloc(MAXCTX * 4);
    }
}

// quantize this token's k, v head slices into the blocked int8 cache
static void kv_insert(kvh_t *c, int pos, const float *kf, const float *vf) {
    int8_t k8[HD], v8[HD];
    c->ksc[pos] = quant(kf, HD, k8, NULL); c->vsc[pos] = quant(vf, HD, v8, NULL);
    int32_t sum = 0; for (int d = 0; d < HD; d++) sum += k8[d]; c->ksum[pos] = sum;
    int8_t *kb = c->k8 + (size_t)(pos / 16) * HD * 16 + (pos % 16) * 4;
    for (int gg = 0; gg < HD / 4; gg++) memcpy(kb + gg * 64, k8 + 4 * gg, 4);
    int8_t *vb = c->v8 + (size_t)(pos / 4) * HD * 4 + pos % 4;
    for (int d = 0; d < HD; d++) vb[d * 4] = v8[d];
}

// prefetch half (0 or 1) of layer l's KV for this thread's heads into L2
static void prefetch_kv(int l, int id, int half) {
    if (!prefetch_on || l >= NLAYER) return;
    int u0, u1; shard(B * NH, id, &u0, &u1);
    for (int i = u0; i < u1; i++) {
        int b = i / NH, hh = i % NH, pos = cur_pos[b]; kvh_t *c = KVH(b, l, hh);
        size_t kbytes = (size_t)(pos / 16 + 1) * HD * 16, vbytes = (size_t)(pos / 4 + 1) * HD * 4;
        for (size_t o = half ? kbytes / 2 : 0; o < (half ? kbytes : kbytes / 2); o += 64) _mm_prefetch((const char *)c->k8 + o, _MM_HINT_T1);
        for (size_t o = half ? vbytes / 2 : 0; o < (half ? vbytes : vbytes / 2); o += 64) _mm_prefetch((const char *)c->v8 + o, _MM_HINT_T1);
    }
}

// one (stream, layer, head): scores and P.V entirely in int8 VNNI, softmax in f32
static void attention_head(int b, int l, int h, int pos, float *out) {
    const kvh_t *c = KVH(b, l, h);
    int8_t q8[HD]; uint8_t qu[HD];
    float qs = quant(q(b) + h * HD, HD, q8, qu) / sqrtf((float)HD);
    const int32_t *q4 = (const int32_t *)qu;
    float sc[MAXCTX]; __m512 mxv = _mm512_set1_ps(-1e30f);
    int nblk = pos / 16 + 1;
    for (int tb = 0; tb < nblk; tb++) {
        __m512i acc = _mm512_setzero_si512(); const int8_t *kb = c->k8 + (size_t)tb * HD * 16;
        for (int gg = 0; gg < HD / 4; gg++) acc = _mm512_dpbusd_epi32(acc, _mm512_set1_epi32(q4[gg]), _mm512_loadu_si512(kb + gg * 64));
        __m512i comp = _mm512_slli_epi32(_mm512_loadu_si512(c->ksum + tb * 16), 7);
        __m512 sv = _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_sub_epi32(acc, comp)), _mm512_mul_ps(_mm512_loadu_ps(c->ksc + tb * 16), _mm512_set1_ps(qs)));
        __mmask16 valid = pos - tb * 16 >= 15 ? 0xFFFF : (__mmask16)((1u << (pos - tb * 16 + 1)) - 1);
        sv = _mm512_mask_mov_ps(_mm512_set1_ps(-1e30f), valid, sv);
        mxv = _mm512_max_ps(mxv, sv); _mm512_storeu_ps(sc + tb * 16, sv);
    }
    __m512 mx = _mm512_set1_ps(_mm512_reduce_max_ps(mxv)), sum = _mm512_setzero_ps(), pm = _mm512_setzero_ps();
    for (int tb = 0; tb < nblk; tb++) {   // e = exp(s - max); p' = e * vsc  (V scale folded into the probability)
        __m512 e = exp512(_mm512_sub_ps(_mm512_loadu_ps(sc + tb * 16), mx));
        sum = _mm512_add_ps(sum, e);
        __m512 pp = _mm512_mul_ps(e, _mm512_loadu_ps(c->vsc + tb * 16));
        pm = _mm512_max_ps(pm, pp); _mm512_storeu_ps(sc + tb * 16, pp);
    }
    float pmax = _mm512_reduce_max_ps(pm), inv = pmax / (255.f * _mm512_reduce_add_ps(sum));
    uint8_t pu[MAXCTX]; __m512 r255 = _mm512_set1_ps(255.f / pmax);
    for (int tb = 0; tb < nblk; tb++)
        _mm_storeu_si128((__m128i *)(pu + tb * 16), _mm512_cvtusepi32_epi8(_mm512_cvtps_epi32(_mm512_mul_ps(_mm512_loadu_ps(sc + tb * 16), r255))));
    __m512i acc[HD / 16]; for (int j = 0; j < HD / 16; j++) acc[j] = _mm512_setzero_si512();
    for (int vb = 0; vb <= pos / 4; vb++) {
        __m512i pb = _mm512_set1_epi32(*(const int32_t *)(pu + 4 * vb)); const int8_t *vr = c->v8 + (size_t)vb * HD * 4;
        for (int j = 0; j < HD / 16; j++) acc[j] = _mm512_dpbusd_epi32(acc[j], pb, _mm512_loadu_si512(vr + j * 64));
    }
    for (int j = 0; j < HD / 16; j++) _mm512_storeu_ps(out + h * HD + j * 16, _mm512_mul_ps(_mm512_cvtepi32_ps(acc[j]), _mm512_set1_ps(inv)));
}

// int8 lm_head, 16-row blocks nb0..nb1, all B streams, no horizontal reductions:
// e8 blocked [v/16][HID/4][16][4], acc[b] holds 16 logits. logits = (acc - 128*esum) * escale * dq
static inline __attribute__((always_inline)) void lm_block(const uint8_t *const *xu, const int8_t *const *xq, const float *dq, const float (*xbs)[NBLK], int b0, int nb, const int BB) {
    if (lm8) {
        __m512i acc[BB]; for (int b = 0; b < BB; b++) acc[b] = _mm512_setzero_si512();
        const int8_t *wb = e8 + (size_t)nb * HID * 16;
        for (int gg = 0; gg < HID / 4; gg++) {
            __m512i w = _mm512_loadu_si512(wb + gg * 64);
            for (int b = 0; b < BB; b++) acc[b] = _mm512_dpbusd_epi32(acc[b], _mm512_set1_epi32(*(const int32_t *)(xu[b0 + b] + 4 * gg)), w);
        }
        __m512i comp = _mm512_slli_epi32(_mm512_loadu_si512(esum + nb * 16), 7); __m512 es = _mm512_loadu_ps(escale + nb * 16);
        for (int b = 0; b < BB; b++)
            _mm512_storeu_ps(logits[b0 + b] + nb * 16, _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_sub_epi32(acc[b], comp)), _mm512_mul_ps(es, _mm512_set1_ps(dq[b0 + b]))));
        return;
    }
    // 4-bit: dpbusd(w+8 as u8, x as s8) = sum w x + 8 sum x. Per 32-k block: subtract 8*sum(x_block), scale by fp16 block scale.
    __m512 facc[BB]; for (int b = 0; b < BB; b++) facc[b] = _mm512_setzero_ps();
    const uint8_t *wb = e4 + (size_t)nb * HID * 8; const __m512i m4 = _mm512_set1_epi8(0x0F);
    for (int blk = 0; blk < NBLK; blk++) {
        __m512i acc[BB]; for (int b = 0; b < BB; b++) acc[b] = _mm512_setzero_si512();
        for (int gg = blk * 4; gg < blk * 4 + 4; gg++) {   // 4 x (16 rows x 8 k)
            __m512i p = _mm512_loadu_si512(wb + gg * 64);
            __m512i wlo = _mm512_and_si512(p, m4), whi = _mm512_and_si512(_mm512_srli_epi16(p, 4), m4);
            for (int b = 0; b < BB; b++) {
                acc[b] = _mm512_dpbusd_epi32(acc[b], wlo, _mm512_set1_epi32(*(const int32_t *)(xq[b0 + b] + 8 * gg)));
                acc[b] = _mm512_dpbusd_epi32(acc[b], whi, _mm512_set1_epi32(*(const int32_t *)(xq[b0 + b] + 8 * gg + 4)));
            }
        }
        __m512 sc = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)(e4s + ((size_t)nb * NBLK + blk) * 16)));
        for (int b = 0; b < BB; b++)
            facc[b] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(_mm512_sub_epi32(acc[b], _mm512_set1_epi32(8 * (int)xbs[b0 + b][blk]))), sc, facc[b]);
    }
    for (int b = 0; b < BB; b++) _mm512_storeu_ps(logits[b0 + b] + nb * 16, _mm512_mul_ps(facc[b], _mm512_set1_ps(dq[b0 + b])));
}
static void lm_head(const act_t *a, int nb0, int nb1, int id) {
    const uint8_t *xu[MAXB] = {0}; const int8_t *xq[MAXB] = {0}; float dq[MAXB] = {0}; static __thread float xbs[MAXB][NBLK];
    for (int b = 0; b < B; b++) { dq[b] = a[b].dq; xu[b] = a[b].qu; xq[b] = a[b].q;
        for (int blk = 0; blk < NBLK; blk++) { int sm = 0; for (int i = 0; i < 32; i++) sm += a[b].q[blk * 32 + i]; xbs[b][blk] = sm; } }
    for (int nb = nb0; nb < nb1; nb++)
        for (int b0 = 0; b0 < B; b0 += 8)
            switch (B - b0 >= 8 ? 8 : B - b0) {
            case 1: lm_block(xu, xq, dq, xbs, b0, nb, 1); break;
            case 2: lm_block(xu, xq, dq, xbs, b0, nb, 2); break;
            case 3: lm_block(xu, xq, dq, xbs, b0, nb, 3); break;
            case 4: lm_block(xu, xq, dq, xbs, b0, nb, 4); break;
            case 5: lm_block(xu, xq, dq, xbs, b0, nb, 5); break;
            case 6: lm_block(xu, xq, dq, xbs, b0, nb, 6); break;
            case 7: lm_block(xu, xq, dq, xbs, b0, nb, 7); break;
            default: lm_block(xu, xq, dq, xbs, b0, nb, 8); break;
            }
    int v0 = nb0 * 16, v1 = nb1 * 16 < VOCAB ? nb1 * 16 : VOCAB;
    for (int b = 0; b < B; b++) { int best = v0; for (int vv = v0 + 1; vv < v1; vv++) if (logits[b][vv] > logits[b][best]) best = vv; best_t[b][id] = best; }
}

static void silu_rows(int i0, int i1) {   // g = silu(g) * u on rows i0..i1 of every stream
    for (int b = 0; b < B; b++)
        for (int i = i0; i < i1; i += 16) {
            __m512 gv = _mm512_loadu_ps(g(b) + i), e = exp512(_mm512_sub_ps(_mm512_setzero_ps(), gv));
            _mm512_storeu_ps(g(b) + i, _mm512_mul_ps(_mm512_div_ps(gv, _mm512_add_ps(_mm512_set1_ps(1.f), e)), _mm512_loadu_ps(u(b) + i)));
        }
}

// one step for all B streams, run by every thread with its own id. Results in best_t[b][0] after the final barrier.
static void forward_mt(int id) {
    static __thread float h[FFN], a2[FFN];
    int b0, b1;
    if (!id) for (int b = 0; b < B; b++) memcpy(x[b], embed + (size_t)cur_tok[b] * HID, sizeof x[b]);
    sbar_sync(&bar);
    for (int l = 0; l < nlayers_run; l++) {
        layer_t *ly = &L[l];
        for (int b = id; b < B; b += T) { P(4, rmsnorm(x[b], ly->ln_in, h, HID)); P(1, prep(&act[b], h, HID)); }
        sbar_sync(&bar);
        P(0, linear(act, &ly->qkv, &qkv[0][0], 3 * HID, id, 0));
        sbar_sync(&bar);
        shard(B * NH, id, &b0, &b1);
        for (int i = b0; i < b1; i++) {
            int b = i / NH, hh = i % NH, pos = cur_pos[b];
            P(3, rope_head(q(b), hh, pos); rope_head(k(b), hh, pos));
            kv_insert(KVH(b, l, hh), pos, k(b) + hh * HD, v(b) + hh * HD);
        }
        sbar_sync(&bar);   // lanes sharing a KV cache (prefill, speculation) must see each other's K/V
        for (int i = b0; i < b1; i++) { int b = i / NH, hh = i % NH; P(2, attention_head(b, l, hh, cur_pos[b], att[b])); }
        sbar_sync(&bar);
        for (int b = id; b < B; b += T) { P(4, rmsnorm(att[b], ly->ln_inner, h, HID)); P(1, prep(&act[b], h, HID)); }
        sbar_sync(&bar);
        P(0, linear(act, &ly->o, &x[0][0], HID, id, 1));
        sbar_sync(&bar);
        for (int b = id; b < B; b += T) { P(4, rmsnorm(x[b], ly->ln_post, h, HID)); P(1, prep(&act[b], h, HID)); }
        sbar_sync(&bar);
        prefetch_kv(l + 1, id, 0);
        P(0, linear(act, &ly->gu, &gu[0][0], 2 * FFN, id, 0));
        sbar_sync(&bar);   // fused gate|up: this thread's silu rows need up rows other threads produced
        shard(FFN / 16, id, &b0, &b1);
        P(4, silu_rows(b0 * 16, b1 * 16));
        sbar_sync(&bar);
        for (int b = id; b < B; b += T) { P(4, rmsnorm(g(b), ly->ln_ffn, a2, FFN)); P(1, prep(&act[b], a2, FFN)); }
        sbar_sync(&bar);
        prefetch_kv(l + 1, id, 1);
        P(0, linear(act, &ly->down, &x[0][0], HID, id, 1));
        sbar_sync(&bar);
    }
    for (int b = id; b < B; b += T) { P(4, rmsnorm(x[b], ln_final, h, HID)); P(1, prep(&act[b], h, HID)); }
    sbar_sync(&bar);
    shard(VPAD / 16, id, &b0, &b1);
    P(5, lm_head(act, b0, b1, id));
    sbar_sync(&bar);
    if (!id) for (int b = 0; b < B; b++) for (int i = 1; i < T; i++) if (logits[b][best_t[b][i]] > logits[b][best_t[b][0]]) best_t[b][0] = best_t[b][i];
}

static void step(const int *tok, const int *pos, int *next) {   // B lanes, each with its own token/position
    for (int b = 0; b < B; b++) { cur_tok[b] = tok[b]; cur_pos[b] = pos[b]; }
    sbar_sync(&bar); forward_mt(0); sbar_sync(&bar);
    for (int b = 0; b < B; b++) next[b] = best_t[b][0];
}

// retrieval drafting (REST-style, training-free): 3-gram -> position in a token corpus, open addressing.
static int32_t *corpus; static long ncorpus; static uint32_t *ctab; static const uint32_t CTAB = 1u << 22;
static inline uint32_t h3(int a, int b, int c) { uint64_t k = ((uint64_t)a << 42) ^ ((uint64_t)b << 21) ^ (uint64_t)c; k *= 0x9E3779B97F4A7C15ull; return (uint32_t)(k >> 40) & (CTAB - 1); }
static void corpus_load(const char *path) {
    FILE *f = fopen(path, "rb"); if (!f) { perror(path); return; }
    fseek(f, 0, SEEK_END); ncorpus = ftell(f) / 4; fseek(f, 0, SEEK_SET);
    corpus = xalloc(ncorpus * 4); if (fread(corpus, 4, ncorpus, f) != (size_t)ncorpus) { perror("read"); exit(1); } fclose(f);
    ctab = xalloc((size_t)CTAB * 4); memset(ctab, 0, (size_t)CTAB * 4);   // 0 = empty; stores pos+1 of the latest occurrence
    for (long i = 0; i + 3 < ncorpus; i++) {
        uint32_t hh = h3(corpus[i], corpus[i + 1], corpus[i + 2]);
        for (int p = 0; p < 8; p++, hh = (hh + 1) & (CTAB - 1)) {
            uint32_t e = ctab[hh];
            if (!e || (corpus[e - 1] == corpus[i] && corpus[e] == corpus[i + 1] && corpus[e + 1] == corpus[i + 2])) { ctab[hh] = i + 1; break; }
        }
    }
    fprintf(stderr, "corpus: %ld tokens indexed\n", ncorpus);
}
static int draft_corpus(const int *hist, int n, int K, int *out) {
    if (!corpus || n < 3) return 0;
    uint32_t hh = h3(hist[n - 3], hist[n - 2], hist[n - 1]);
    for (int p = 0; p < 8; p++, hh = (hh + 1) & (CTAB - 1)) {
        uint32_t e = ctab[hh]; if (!e) return 0;
        long i = e - 1;
        if (corpus[i] == hist[n - 3] && corpus[i + 1] == hist[n - 2] && corpus[i + 2] == hist[n - 1]) {
            int cnt = 0; for (long j = i + 3; j < ncorpus && cnt < K; j++) out[cnt++] = corpus[j];
            return cnt;
        }
    }
    return 0;
}

static int draft_any(const int *hist, int n, int K, int *out) {   // prompt lookup, then the corpus table
    int c = draft(hist, n, K, out); return c ? c : draft_corpus(hist, n, K, out);
}

// prefill `prompt`, generate up to ngen tokens, print ids (stream 0) to stdout, stats to stderr.
static void generate(const int *prompt, int np, int ngen) {
    int tok[MAXB], posv[MAXB], next[MAXB]; double t0;
    int K = getenv("SPEC") ? atoi(getenv("SPEC")) : 0; if (K > MAXB - 1) K = MAXB - 1;
    static int hist[MAXCTX]; int n = 0;
    for (int i = 0; i < np; i++) hist[n++] = prompt[i];
    const int nstreams = B;
    // prefill: prompt tokens as lanes, up to MAXB per step, every stream gets the same prompt
    t0 = now_ns(); double nll = 0;
    for (int i0 = 0; i0 < np; i0 += MAXB / nstreams) {
        int cnt = np - i0 < MAXB / nstreams ? np - i0 : MAXB / nstreams; B = cnt * nstreams;
        for (int b = 0; b < B; b++) { kv_map[b] = b / cnt; tok[b] = hist[i0 + b % cnt]; posv[b] = i0 + b % cnt; }
        step(tok, posv, next);
        if (getenv("PPL")) for (int j = 0; j < cnt && i0 + j + 1 < np; j++) {   // log p(next token | prefix), stream 0
            float mx = -1e30f; for (int vv = 0; vv < VOCAB; vv++) if (logits[j][vv] > mx) mx = logits[j][vv];
            double z = 0; for (int vv = 0; vv < VOCAB; vv++) z += exp(logits[j][vv] - mx);
            nll -= logits[j][hist[i0 + j + 1]] - mx - log(z);
        }
        if (getenv("DUMP")) for (int j = 0; j < cnt; j++) printf("%d\n", next[j]);
        for (int b = 0; b < nstreams; b++) { int last = (b + 1) * cnt - 1;   // last lane of each stream holds the next-token logits
            next[b] = next[last]; if (last != b) memcpy(logits[b], logits[last], sizeof logits[b]); }   // last lane of each stream
    }
    double tp = now_ns() - t0;
    if (getenv("PPL")) fprintf(stderr, "perplexity over %d tokens: %.2f\n", np - 1, exp(nll / (np - 1)));
    fprintf(stderr, "top5:"); { static float tmp[VOCAB]; memcpy(tmp, logits[np - 1 < MAXB / nstreams ? np - 1 : (np - 1) % (MAXB / nstreams)], sizeof tmp);
        for (int i = 0; i < 5; i++) { int b = 0; for (int vv = 1; vv < VOCAB; vv++) if (tmp[vv] > tmp[b]) b = vv; fprintf(stderr, " %d(%.3f)", b, tmp[b]); tmp[b] = -1e30f; } }
    fprintf(stderr, "\n");

    t0 = now_ns();
    int gen = 0, same = 1, steps = 0, drafted = 0, accepted = 0;
    B = nstreams; for (int b = 0; b < B; b++) kv_map[b] = b;
    if (K && nstreams == 1 && temp <= 0 && rep <= 1.0f) {   // single stream, speculative (exact greedy only): lanes = [next, draft...] at consecutive positions, shared KV
        int cur = next[0];
        while (gen < ngen && n < MAXCTX - K - 1) {
            printf("%d\n", cur); fflush(stdout); hist[n++] = cur; gen++; if (cur == 2) break;
            int d[MAXB], nd = draft_any(hist, n, K, d); drafted += nd;
            B = 1 + nd; tok[0] = cur; posv[0] = n - 1;
            for (int j = 0; j < nd; j++) { tok[1 + j] = d[j]; posv[1 + j] = n + j; kv_map[1 + j] = 0; }
            step(tok, posv, next); steps++;
            int acc = 0; while (acc < nd && next[acc] == d[acc]) acc++;   // lane j predicted draft j+1?
            accepted += acc;
            for (int j = 0; j < acc && gen < ngen; j++) { printf("%d\n", d[j]); fflush(stdout); hist[n++] = d[j]; gen++; }
            cur = next[acc];   // first token the model disagreed on (or the one after a full accept)
            B = 1;
        }
    } else {
        for (int b = 0; b < B; b++) tok[b] = sample(logits[b], next[b], hist, n);
        for (; gen < ngen && n < MAXCTX; gen++) {
            printf("%d\n", tok[0]); fflush(stdout);
            for (int b = 1; b < B; b++) if (tok[b] != tok[0]) same = 0;
            if (tok[0] == 2) break;
            for (int b = 0; b < B; b++) posv[b] = n;
            step(tok, posv, next); steps++; n++;
            for (int b = 0; b < B; b++) tok[b] = sample(logits[b], next[b], hist, n);
        }
    }
    double tg = now_ns() - t0;
    if (getenv("PROF")) for (int i = 0; i < 6; i++) fprintf(stderr, "  %-10s %7.2f ms/step (thread 0)\n", pname[i], prof[i] / 1e6 / (steps + (np + MAXB - 1) / MAXB));
    if (nstreams > 1) fprintf(stderr, "all %d streams identical: %s\n", nstreams, same ? "yes" : "NO");
    if (K && nstreams == 1) fprintf(stderr, "speculation K=%d: %d steps, %d drafted, %d accepted (%.0f%%), %.2f tokens/step\n", K, steps, drafted, accepted, drafted ? 100.0 * accepted / drafted : 0, (double)gen / steps);
    fprintf(stderr, "prefill %d tok in %.1f ms (%.0f tok/s), decode %d tok x %d streams in %.1f ms: %.1f tok/s per stream, %.0f tok/s aggregate\n",
            np, tp / 1e6, np / (tp / 1e9), gen, nstreams, tg / 1e6, gen / (tg / 1e9), (double)gen * nstreams / (tg / 1e9));
}

int main(int argc, char **argv) {
    if (argc < 4 && !(argc == 3 && !strcmp(argv[2], "-"))) { fprintf(stderr, "usage: %s model_dir n_gen tok...  |  %s model_dir -  (stdin: \"ngen id id ...\" per line)\n", argv[0], argv[0]); return 1; }
    int nc = find_cores(cpus, 256);
    T = getenv("THREADS") ? atoi(getenv("THREADS")) : nc;
    if (T > nc) for (int i = 0; i < T && i < 256; i++) cpus[i] = i;   // SMT: one thread per logical cpu
    B = getenv("BATCH") ? atoi(getenv("BATCH")) : 1; if (B > MAXB) B = MAXB;
    nkv = B; kv_alloc(); prefetch_on = !getenv("NOPREFETCH");
    double t0 = now_ns();
    sparse_h = getenv("SPARSE") ? atof(getenv("SPARSE")) : 0; sparse_f = getenv("SPARSE_FFN") ? atof(getenv("SPARSE_FFN")) : 0;
    if (getenv("EXIT")) nlayers_run = atoi(getenv("EXIT"));
    if (getenv("TEMP")) temp = atof(getenv("TEMP"));
    if (getenv("REP")) rep = atof(getenv("REP"));
    if (getenv("TOPK")) topk = atoi(getenv("TOPK"));
    if (getenv("SEED")) srng = strtoull(getenv("SEED"), NULL, 10) * 2654435761ull + 1;
    if (getenv("CORPUS")) corpus_load(getenv("CORPUS"));
    load(argv[1]); make_tables(); make_perm(perm_h, HID); make_perm(perm_f, FFN);
    char nm[128];
    for (int l = 0; l < NLAYER; l++) {
#define TN(tname) (snprintf(nm, sizeof nm, "model.layers.%d." tname ".weight", l), tensor(nm))
        lmat_alloc(&L[l].qkv, 3 * HID, HID); ternarize(&L[l].qkv, 0, TN("self_attn.q_proj"), HID, HID); ternarize(&L[l].qkv, HID, TN("self_attn.k_proj"), HID, HID); ternarize(&L[l].qkv, 2 * HID, TN("self_attn.v_proj"), HID, HID);
        lmat_alloc(&L[l].o, HID, HID); ternarize(&L[l].o, 0, TN("self_attn.o_proj"), HID, HID);
        lmat_alloc(&L[l].gu, 2 * FFN, HID); ternarize(&L[l].gu, 0, TN("mlp.gate_proj"), FFN, HID); ternarize(&L[l].gu, FFN, TN("mlp.up_proj"), FFN, HID);
        lmat_alloc(&L[l].down, HID, FFN); ternarize(&L[l].down, 0, TN("mlp.down_proj"), HID, FFN);
#define LN(field, tname) snprintf(nm, sizeof nm, "model.layers.%d." tname ".weight", l); L[l].field = tensor(nm);
        LN(ln_in, "input_layernorm") LN(ln_post, "post_attention_layernorm") LN(ln_inner, "self_attn.inner_attn_ln") LN(ln_ffn, "mlp.ffn_layernorm")
    }
    embed = tensor("model.embed_tokens.weight"); ln_final = tensor("model.norm.weight");
    lm8 = getenv("LM8") != NULL;
    escale = xalloc(VPAD * 4); esum = xalloc(VPAD * 4); memset(escale, 0, VPAD * 4); memset(esum, 0, VPAD * 4);
    if (lm8) { e8 = xalloc((size_t)VPAD * HID); memset(e8, 0, (size_t)VPAD * HID); }
    else      { e4 = xalloc((size_t)VPAD * HID / 2); memset(e4, 0, (size_t)VPAD * HID / 2); e4s = xalloc((size_t)VPAD * NBLK * 2); memset(e4s, 0, (size_t)VPAD * NBLK * 2); }
    for (int vv = 0; vv < VOCAB; vv++) {
        const float *row = embed + (size_t)vv * HID; int32_t sm = 0;
        if (lm8) {
            int8_t r8[HID]; escale[vv] = quant(row, HID, r8, NULL);
            for (int i = 0; i < HID; i++) sm += r8[i];
            for (int gg = 0; gg < HID / 4; gg++) memcpy(e8 + (((size_t)(vv / 16) * (HID / 4) + gg) * 16 + vv % 16) * 4, r8 + 4 * gg, 4);
        } else {
            for (int blk = 0; blk < NBLK; blk++) {
                float am = 1e-5f; for (int i = 0; i < 32; i++) am = fmaxf(am, fabsf(row[blk * 32 + i]));
                float scl = am / 7.0f; e4s[((size_t)(vv / 16) * NBLK + blk) * 16 + vv % 16] = _cvtss_sh(scl, 0);
                for (int gg = blk * 4; gg < blk * 4 + 4; gg++) {
                    uint8_t *dst = e4 + (((size_t)(vv / 16) * (HID / 8) + gg) * 16 + vv % 16) * 4;
                    for (int i = 0; i < 4; i++) {
                        float a = rintf(row[8 * gg + i] / scl), c = rintf(row[8 * gg + 4 + i] / scl);
                        int wa = a < -8 ? -8 : a > 7 ? 7 : (int)a, wc = c < -8 ? -8 : c > 7 ? 7 : (int)c;
                        dst[i] = (uint8_t)((wa + 8) | (wc + 8) << 4);
                    }
                }
            }
        }
        esum[vv] = sm;
    }
    for (int p = 0; p < MAXCTX; p++) for (int i = 0; i < HD / 2; i++) {
        float ang = p * powf(10000.0f, -2.0f * i / HD); rope_c[p][i] = cosf(ang); rope_s[p][i] = sinf(ang);
    }
    fprintf(stderr, "loaded in %.1f s, %d threads, batch %d\n", (now_ns() - t0) / 1e9, T, B);
    if (getenv("CHECK2")) {   // one full layer, scalar f32 reference vs engine residual (pos 0, token 1)
        static float xr[HID], hh[FFN], qr[3 * HID], ar[HID], o2[HID], gr[2 * FFN], ac[FFN], dn[HID];
        memcpy(xr, embed + HID, sizeof xr);
        // reference linear: same int8 quantization, scalar trit dot, per-row scale
        #define REFLIN(m, in, out) do { int8_t qq[FFN]; float dq_ = quant(in, (m)->k, qq, NULL); int Q_ = (m)->kp / 20; \
            for (int r = 0; r < (m)->n; r++) { long acc = 0; for (int qd = 0; qd < Q_; qd++) for (int bp = 0; bp < 4; bp++) { \
                int vv = (m)->w[(((size_t)(r / 16) * Q_ + qd) * 16 + r % 16) * 4 + bp]; \
                for (int i = 0; i < 5; i++, vv /= 3) { int kk = (qd * 4 + bp) * 5 + i; if (kk < (m)->k) acc += (vv % 3 - 1) * qq[kk]; } } \
                out[r] = acc * dq_ * (m)->scale[r / 16]; } } while (0)
        layer_t *ly = &L[0];
        rmsnorm(xr, ly->ln_in, hh, HID); REFLIN(&ly->qkv, hh, qr);
        memcpy(ar, qr + 2 * HID, sizeof ar);                      // pos 0: attention output = v
        rmsnorm(ar, ly->ln_inner, hh, HID); REFLIN(&ly->o, hh, o2); for (int i = 0; i < HID; i++) xr[i] += o2[i];
        rmsnorm(xr, ly->ln_post, hh, HID); REFLIN(&ly->gu, hh, gr);
        for (int i = 0; i < FFN; i++) ac[i] = gr[i] / (1.0f + expf(-gr[i])) * gr[FFN + i];
        rmsnorm(ac, ly->ln_ffn, hh, FFN); REFLIN(&ly->down, hh, dn); for (int i = 0; i < HID; i++) xr[i] += dn[i];
        // engine: one layer
        nlayers_run = 1; sbar_init(&bar, T); pthread_t th2[256]; pin(cpus[0]);
        for (long i = 1; i < T; i++) pthread_create(&th2[i], NULL, worker, (void *)i);
        int tk[1] = {1}, ps[1] = {0}, nx[1]; B = 1; kv_map[0] = 0; step(tk, ps, nx);
        double md = 0; int mi = 0; for (int i = 0; i < HID; i++) if (fabsf(xr[i] - x[0][i]) > md) { md = fabsf(xr[i] - x[0][i]); mi = i; }
        fprintf(stderr, "  layer0 residual: max |ref - engine| = %g at %d (ref %g engine %g); x[0..2] ref %g %g %g engine %g %g %g\n", md, mi, xr[mi], x[0][mi], xr[0], xr[1], xr[2], x[0][0], x[0][1], x[0][2]);
        // also q/k/v vs engine's qkv buffer, and gate/up
        md = 0; for (int i = 0; i < 3 * HID; i++) if (fabsf(qr[i] - qkv[0][i]) > md) md = fabsf(qr[i] - qkv[0][i]);
        fprintf(stderr, "  qkv max diff %g (engine qkv is post-rope; pos 0 rope = identity)\n", md);
        md = 0; for (int i = 0; i < FFN; i++) if (fabsf(ac[i] - gu[0][i]) > md) md = fabsf(ac[i] - gu[0][i]);
        fprintf(stderr, "  silu(g)*u max diff %g\n", md);
        quit = 1; sbar_sync(&bar); for (int i = 1; i < T; i++) pthread_join(th2[i], NULL);
        return 0;
    }
    if (getenv("CHECK")) {   // fused matvec vs scalar reference on a few rows of each matrix of layer 0
        static float xf[FFN]; for (int i = 0; i < FFN; i++) xf[i] = sinf(i * 0.37f) * 3;
        const lmat_t *ms[4] = {&L[0].qkv, &L[0].o, &L[0].gu, &L[0].down}; const char *mn[4] = {"qkv", "o", "gu", "down"};
        for (int mi = 0; mi < 4; mi++) {
            const lmat_t *m = ms[mi]; prep(&act[0], xf, m->k);
            const uint8_t *xp[1] = {act[0].xp}; int32_t *y[1] = {act[0].y[0]};
            B = 1; mv_b(m, xp, y, 1, 0, m->n / 16);
            int rows[4] = {0, m->n / 3, m->n / 2, m->n - 1}, bad = 0, Q = m->kp / 20;
            for (int ri = 0; ri < 4; ri++) {
                int r = rows[ri]; long acc = 0;
                for (int qd = 0; qd < Q; qd++) for (int bp = 0; bp < 4; bp++) {
                    int vv = m->w[(((size_t)(r / 16) * Q + qd) * 16 + r % 16) * 4 + bp];
                    for (int i = 0; i < 5; i++, vv /= 3) { int kk = (qd * 4 + bp) * 5 + i; if (kk < m->k) acc += (vv % 3 - 1) * act[0].q[kk]; }
                }
                if (acc != y[0][r]) { bad++; fprintf(stderr, "  %s row %d: ref %ld got %d\n", mn[mi], r, acc, y[0][r]); }
            }
            fprintf(stderr, "  %-4s n=%d k=%d: %s, scale[0]=%g scale[last]=%g\n", mn[mi], m->n, m->k, bad ? "MISMATCH" : "ok", m->scale[0], m->scale[m->n / 16 - 1]);
        }
        return 0;
    }
    if (getenv("KBENCH")) {   // hot gate matrix (4096x1536), 1 thread, compute-bound: ns and GMAC/s per batch size
        static float xf[FFN]; for (int i = 0; i < FFN; i++) xf[i] = sinf(i * 0.37f);
        for (int b = 0; b < MAXB; b++) prep(&act[b], xf, HID);
        const uint8_t *xp[MAXB]; int32_t *y[MAXB]; for (int b = 0; b < MAXB; b++) { xp[b] = act[b].xp; y[b] = act[b].y[0]; }
        int bs[] = {1, 4, 8, 16, 32};
        for (int i = 0; i < 5; i++) { double t; BENCH(t, 20, mv_b(&L[0].gu, xp, y, bs[i], 0, FFN / 16));
            fprintf(stderr, "  1.6-bit SLAB=%d B=%2d: %7.0f us  %5.0f GMAC/s\n", SLAB, bs[i], t / 1e3, (double)FFN * HID * bs[i] / t); }
        return 0;
    }

    sbar_init(&bar, T);
    pthread_t th[256]; pin(cpus[0]);
    for (long i = 1; i < T; i++) pthread_create(&th[i], NULL, worker, (void *)i);

    if (argc == 3 && !strcmp(argv[2], "-")) {   // REPL: lines of "ngen id id id ...", answer ids then END
        char line[65536]; static int ids[MAXCTX];
        fprintf(stderr, "ready\n");
        while (fgets(line, sizeof line, stdin)) {
            char *p = line; int ngen = (int)strtol(p, &p, 10), np = 0;
            while (np < MAXCTX - 1) { char *e; long v = strtol(p, &e, 10); if (e == p) break; ids[np++] = (int)v; p = e; }
            if (np) generate(ids, np, ngen);
            printf("END\n"); fflush(stdout);
        }
    } else {
        static int ids[MAXCTX]; int np = argc - 3; if (np > MAXCTX - 1) np = MAXCTX - 1;
        for (int i = 0; i < np; i++) ids[i] = atoi(argv[3 + i]);
        generate(ids, np, atoi(argv[2]));
    }
    quit = 1; sbar_sync(&bar);
    for (int i = 1; i < T; i++) pthread_join(th[i], NULL);
    return 0;
}
