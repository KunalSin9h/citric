// Citric engine for kunalsin9h/gemma-3-1b-it-ternary: Gemma 3 1B-it whose decoder linears were trained ternary
// (QAT + distillation, train/ternary.py). Same forward as gemma.c; the 4 decoder matrices per layer run the 1.6-bit
// kernel from bitnet.c (5 trits per byte, permuted int8 activations, vpdpbusd), with the int8 per-token activation
// quantization the model was trained with. The tied embedding / lm_head stays 4-bit block-32 (it was not ternarized).
// usage: ./gemma3t models/gemma-3-1b-it-ternary N_GEN tok...   |   ./gemma3t models/gemma-3-1b-it-ternary -

#define HID 1152
#define FFN 6912
#define NH 4
#define HD 256
#define QDIM (NH * HD)       // 1024
#define KVDIM HD             // 1 kv head
#define QKV (QDIM + 2 * KVDIM)
#define NLAYER 26
#define VOCAB 262144
#define MAXCTX 1024
#define WINDOW 512
#define EPS 1e-6f
#ifndef MAXB
#define MAXB 32
#endif
#define IS_GLOBAL(l) (((l) + 1) % 6 == 0)

#include "common.h"
#include "q4.h"
#include "ternary.h"
#include "gemma3.h"

// ternary matrix, bitnet.c layout: [n/16][kp/20][16 rows][4 bytes], byte = sum (trit_i + 1) * 3^i over 5 k-values
static int16_t perm_h[KPAD(HID)], perm_q[KPAD(QDIM)], perm_f[KPAD(FFN)];   // xperm[j] = x[perm[j]], -1 = zero pad
typedef struct { lmat_t qkv, o, gu, down; float *ln_in, *ln_post_attn, *ln_pre_ffn, *ln_post_ffn, *qn, *kn; } layer_t;

static layer_t L[NLAYER];
static qmat_t emb4;                 // tied embeddings as the lm_head
static const uint16_t *emb_bf16;    // embedding lookup rows (mmap)
static float *ln_final;
static double prof[12]; static const char *pname[11] = {"matvec", "quant", "attention", "rope/norm", "gelu", "lm_head (exact)", "barrier wait", "outside fwd", "head: sketch", "", "head: rescore"};
#define P(i, stmt) do { double t_ = now_ns(); stmt; if (!id) prof[i] += now_ns() - t_; } while (0)

// ---------- safetensors (bf16) ----------
static char *hdr; static uint8_t *data;
static const void *tensor_raw(const char *name) {
    char key[256]; snprintf(key, sizeof key, "\"%s\"", name);
    char *p = strstr(hdr, key); if (!p) { fprintf(stderr, "missing tensor %s\n", name); exit(1); }
    p = strstr(p, "\"data_offsets\""); p = strchr(p, '[');
    return (const void *)(data + strtoul(p + 1, NULL, 10));
}
static const uint16_t *tensor(const char *name) { return (const uint16_t *)tensor_raw(name); }
static int packed;   // model-trit5.safetensors: trits packed 5 per byte (as published on Hugging Face); else the int8 model-ternary.safetensors
static void load(const char *dir) {
    char path[512]; snprintf(path, sizeof path, "%s/model-trit5.safetensors", dir);
    int fd = open(path, O_RDONLY); packed = fd >= 0;
    if (!packed) { snprintf(path, sizeof path, "%s/model-ternary.safetensors", dir); fd = open(path, O_RDONLY); }
    if (fd < 0) { perror(path); exit(1); }
    struct stat st; fstat(fd, &st);
    uint8_t *m = mmap(NULL, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    uint64_t n = *(uint64_t *)m; hdr = malloc(n + 1); memcpy(hdr, m + 8, n); hdr[n] = 0; data = m + 8 + n;
}
// int8 trits of an [n][k] ternary matrix, unpacked from 5-per-byte when the file is packed (buffer reused per call)
static const int8_t *trits(const char *name, int n, int k) {
    if (!packed) return (const int8_t *)tensor_raw(name);
    static int8_t *buf; static size_t cap; const uint8_t *p = tensor_raw(name); const int kb = (k + 4) / 5;
    if ((size_t)n * k > cap) { free(buf); cap = (size_t)n * k; buf = xalloc(cap); }
    for (int r = 0; r < n; r++)
        for (int j = 0; j < k; j++) buf[(size_t)r * k + j] = p[(size_t)r * kb + j / 5] / (int[]){1, 3, 9, 27, 81}[j % 5] % 3 - 1;
    return buf;
}
static float *vecf(const char *name, int n) { const uint16_t *t = tensor(name); float *f = xalloc(n * 4); for (int i = 0; i < n; i++) f[i] = bf(t[i]); return f; }

static void lmat_alloc(lmat_t *d, int n, int k) { d->n = n; d->k = k; d->kp = KPAD(k); d->w = xalloc((size_t)n * (d->kp / 20) * 4); d->rs = xalloc(n * 4); d->scale = xalloc(n / 16 * 4); }
// pack an exported int8 {-1,0,1} [n][k] matrix with scale s into rows r0.. of d
static void pack_ternary(lmat_t *d, int r0, const int8_t *q, float s, int n, int k) {
    const int Q = d->kp / 20;
    for (int nb = r0 / 16; nb < (r0 + n) / 16; nb++) d->scale[nb] = s;
    for (int r = 0; r < n; r++) {
        int rr = r0 + r; int32_t rs = 0;
        for (int qd = 0; qd < Q; qd++)
            for (int bp = 0; bp < 4; bp++) {
                int v = 0, p = 1;
                for (int i = 0; i < 5; i++, p *= 3) { int kk = (qd * 4 + bp) * 5 + i, t = kk < k ? q[(size_t)r * k + kk] : 0; rs += t; v += (t + 1) * p; }
                d->w[(((size_t)(rr / 16) * Q + qd) * 16 + rr % 16) * 4 + bp] = (uint8_t)v;
            }
        d->rs[rr] = rs;
    }
}

// per-token absmax int8 + per-32-block sums. returns dequant factor.
typedef struct { int8_t q[FFN]; float xbs[FFN / 32]; float dq; uint8_t xp[KPAD(FFN)]; int32_t y[MAXB][2 * FFN]; } act_t;   // xp: permuted q+128 for the ternary kernel; y[thread]: its int32 output
static act_t act[MAXB];
static int st_kind = -1; static long st_groups[4], st_zero[4];   // STATS=1: groups of 4 consecutive int8 activations that are all zero
static void prep(act_t *a, const float *xf, int n) {
    __m512 am = _mm512_set1_ps(1e-8f);
    for (int i = 0; i < n; i += 16) am = _mm512_max_ps(am, _mm512_abs_ps(_mm512_loadu_ps(xf + i)));
    float sc = 127.0f / _mm512_reduce_max_ps(am); __m512 sv = _mm512_set1_ps(sc);
    for (int i = 0; i < n; i += 16) _mm_storeu_si128((__m128i *)(a->q + i), _mm512_cvtsepi32_epi8(_mm512_cvtps_epi32(_mm512_mul_ps(_mm512_loadu_ps(xf + i), sv))));
    for (int blk = 0; blk < n / 32; blk++) { int s = 0; for (int i = 0; i < 32; i++) s += a->q[blk * 32 + i]; a->xbs[blk] = s; }
    a->dq = 1.0f / sc;
    if (st_kind >= 0) for (int g = 0; g < n / 4; g++) { st_groups[st_kind]++; st_zero[st_kind] += !(a->q[4*g] | a->q[4*g+1] | a->q[4*g+2] | a->q[4*g+3]); }
    const int16_t *p = n == HID ? perm_h : n == QDIM ? perm_q : perm_f;
    for (int j = 0; j < KPAD(n); j++) a->xp[j] = p[j] < 0 ? 128 : (uint8_t)(a->q[p[j]] + 128);
}
// this thread's row shard of out[b] = W @ x[b] for all lanes, ternary kernel; dequant = act scale * matrix scale
// out[b] = W @ x[b] for all lanes, ternary kernel; dequant = act scale * matrix scale. Threads take CH row blocks at a
// time from *ctr (zeroed once per step) instead of a fixed shard: under shared DRAM the shards finish unevenly.
#ifndef CH
#define CH 2
#endif
static int mm_ctr[NLAYER * 4][16] __attribute__((aligned(64))), hd_ctr[16] __attribute__((aligned(64)));   // one cache line each: no false sharing
static void tlinear_rows(const lmat_t *m, float *out, int stride, int id, int b0, int b1) {
    const uint8_t *xp[MAXB] = {0}; int32_t *y[MAXB] = {0};
    for (int b = 0; b < B; b++) { xp[b] = act[b].xp; y[b] = act[b].y[id]; }
    mv_b(m, xp, y, B, b0, b1);
    for (int b = 0; b < B; b++) {
        float *o = out + (size_t)b * stride;
        for (int nb = b0; nb < b1; nb++) { float f = act[b].dq * m->scale[nb]; for (int i = nb * 16; i < nb * 16 + 16; i++) o[i] = y[b][i] * f; }
    }
}
static void tlinear(const lmat_t *m, float *out, int stride, int id, int *ctr) {
    const int nb = m->n / 16;
    if (B > 4) { int b0, b1; shard(nb, id, &b0, &b1); tlinear_rows(m, out, stride, id, b0, b1); return; }   // batches: compute-bound, static shards balance
    for (int b0; (b0 = __atomic_fetch_add(ctr, CH, __ATOMIC_RELAXED)) < nb;) tlinear_rows(m, out, stride, id, b0, b0 + CH < nb ? b0 + CH : nb);
}

// ---------- sketch output head ----------
// 1-bit sign copy of the tied embedding, [VOCAB/8][HW words][8 rows] u64, plus mean|e| per row. A query is the
// final hidden state's signs and 4 magnitude buckets (top 1/8, next 1/8, next 1/4, rest of |h|), each with its mean
// |h| as weight. Approximate logit_v = escale_v * sum_b w_b * (n_b - 2 * popcount((sign_e_v xor sign_h) & mask_b)).
// Hybrid: the TOPD strongest dims of h are scored exactly from a column-major int8 table; the sketch covers the rest.
// The top ~SKETCH_K rows are rescored exactly from a row-major int8 table; the rest get -inf.
// Measured: picks the same word as an exact int8 head over all 262k rows in 227/227 positions (K=512); 2.6 ms vs 6.4 ms.
#define HW (HID / 64)   // 18 words of 64 dims
#ifndef NBK
#define NBK 8
#endif
static uint64_t *sk; static float *sk_scale;
#ifndef TOPD
#define TOPD 32   // hidden dims scored exactly (32 vs 64: same argmax in 758/758 checked steps, 4% faster)
#endif
static int8_t *embR, *embT; static float *es;   // int8 table row-major [V][HID] and column-major [HID][V], scale per row
static int sketch_on, sketch_check, sketch_k = 1024;
typedef struct { uint64_t sgn[HW], msk[NBK][HW]; float w[NBK]; int n[NBK]; int top[TOPD]; float htop[TOPD]; float h[HID]; } skq_t;
static skq_t skq[MAXB];
static float *skscore;   // [MAXB][VOCAB]
static float skthr[MAXB];
// Output head lanes. Only lanes whose logits are needed (need[b]; a prefill chunk needs just its last token) run the
// 262k-row head, compacted to head slots 0..HB-1: slot j reads lane hl[j]'s hidden state, writes lane hl[j]'s logits.
static int need[MAXB], HB, hl[MAXB];
// Optional per-lane top-TK candidates (sorted, best first), merged from each thread's vocab shard while it is hot in cache.
#define TK 64
static int topk_on; static float *tkv; static int *tki;   // per thread: [T][MAXB][TK]; merged result in thread 0's rows
static long ck_steps, ck_hit, ck_same8; static double ck_mass;

static void sketch_build(void) {
    sk = xalloc((size_t)VOCAB * HW * 8); sk_scale = xalloc((size_t)VOCAB * 4); skscore = xalloc((size_t)MAXB * VOCAB * 4);
    for (int v = 0; v < VOCAB; v++) {
        const uint16_t *e = emb_bf16 + (size_t)v * HID; double a = 0;
        for (int w = 0; w < HW; w++) {
            uint64_t bits = 0;
            for (int i = 0; i < 64; i++) { uint16_t u = e[w * 64 + i]; if (u & 0x8000) bits |= 1ull << i; a += fabsf(bf(u)); }
            sk[((size_t)(v / 8) * HW + w) * 8 + v % 8] = bits;
        }
        sk_scale[v] = (float)(a / HID);
    }
    embR = xalloc((size_t)VOCAB * HID); embT = xalloc((size_t)VOCAB * HID); es = xalloc((size_t)VOCAB * 4);
    for (int v = 0; v < VOCAB; v++) {
        const uint16_t *e = emb_bf16 + (size_t)v * HID; float am = 1e-8f;
        for (int k = 0; k < HID; k++) am = fmaxf(am, fabsf(bf(e[k])));
        es[v] = am / 127.0f;
        for (int k = 0; k < HID; k++) { int8_t q8 = (int8_t)rintf(bf(e[k]) / es[v]); embR[(size_t)v * HID + k] = q8; embT[(size_t)k * VOCAB + v] = q8; }
    }
}
static int cmpf_desc(const void *a, const void *b) { float x = *(const float *)a, y = *(const float *)b; return (x < y) - (x > y); }
static void sketch_query(skq_t *q, const float *h) {
    static __thread float mag[HID], srt[HID];
    for (int i = 0; i < HID; i++) { mag[i] = fabsf(h[i]); srt[i] = mag[i]; }
    qsort(srt, HID, sizeof(float), cmpf_desc);
#if NBK == 8
    const int cut[NBK - 1] = { HID / 32, HID / 16, HID / 8, 3 * HID / 16, HID / 4, 3 * HID / 8, HID / 2 };   // finer buckets where |h| is largest
#else
    const int cut[NBK - 1] = { HID / 8, HID / 4, HID / 2 };
#endif
    float t[NBK - 1]; for (int j = 0; j < NBK - 1; j++) t[j] = srt[cut[j]];   // bucket lower bounds
    double sum[NBK] = {0}; memset(q, 0, sizeof *q); memcpy(q->h, h, sizeof q->h);
    const float ttop = srt[TOPD - 1]; int nt = 0;
    for (int i = 0; i < HID; i++) {
        if (mag[i] >= ttop && nt < TOPD) { q->top[nt] = i; q->htop[nt++] = h[i]; continue; }   // exact part, not in the sketch
        int bk = NBK - 1; for (int j = 0; j < NBK - 1; j++) if (mag[i] > t[j]) { bk = j; break; }
        q->msk[bk][i / 64] |= 1ull << (i % 64); q->n[bk]++; sum[bk] += mag[i];
        if (h[i] < 0) q->sgn[i / 64] |= 1ull << (i % 64);
    }
    for (int b = 0; b < NBK; b++) q->w[b] = q->n[b] ? (float)(sum[b] / q->n[b]) : 0.f;
}
// scores for rows of 8-row groups g0..g1, all lanes: exact over the TOPD strongest dims + sketch over the rest
static void sketch_scores(int g0, int g1) {
    for (int g = g0; g < g1; g++) {
        const uint64_t *base = sk + (size_t)g * HW * 8;
        for (int b = 0; b < HB; b++) {
            const skq_t *q = &skq[b]; __m512i dis[NBK];
            for (int k = 0; k < NBK; k++) dis[k] = _mm512_setzero_si512();
            for (int w = 0; w < HW; w++) {
                __m512i x = _mm512_xor_si512(_mm512_loadu_si512(base + w * 8), _mm512_set1_epi64((long long)q->sgn[w]));
                for (int k = 0; k < NBK; k++) dis[k] = _mm512_add_epi64(dis[k], _mm512_popcnt_epi64(_mm512_and_si512(x, _mm512_set1_epi64((long long)q->msk[k][w]))));
            }
            __m512d acc = _mm512_setzero_pd();
            for (int k = 0; k < NBK; k++)
                acc = _mm512_fmadd_pd(_mm512_sub_pd(_mm512_set1_pd(q->n[k]), _mm512_mul_pd(_mm512_set1_pd(2.0), _mm512_cvtepi64_pd(dis[k]))), _mm512_set1_pd(q->w[k]), acc);
            __m256 sc = _mm256_mul_ps(_mm512_cvtpd_ps(acc), _mm256_loadu_ps(sk_scale + (size_t)g * 8));
            _mm256_storeu_ps(skscore + (size_t)b * VOCAB + (size_t)g * 8, sc);
        }
    }
}
static void sketch_exact(int g0, int g1) {
    // exact part, 64 rows at a time, column-major: one full 64-byte cache line per top dim
    for (int r = g0 * 8; r < g1 * 8; r += 64)
        for (int b = 0; b < HB; b++) {
            const skq_t *q = &skq[b]; __m512 e0 = _mm512_setzero_ps(), e1 = e0, e2 = e0, e3 = e0;
            for (int j = 0; j < TOPD; j++) {
                const int8_t *col = embT + (size_t)q->top[j] * VOCAB + r; __m512 hv = _mm512_set1_ps(q->htop[j]);
                e0 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i *)(col)))), hv, e0);
                e1 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i *)(col + 16)))), hv, e1);
                e2 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i *)(col + 32)))), hv, e2);
                e3 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i *)(col + 48)))), hv, e3);
            }
            float *sc = skscore + (size_t)b * VOCAB + r; __m512 ex[4] = { e0, e1, e2, e3 };
            for (int c = 0; c < 4; c++) _mm512_storeu_ps(sc + 16 * c, _mm512_fmadd_ps(ex[c], _mm512_loadu_ps(es + r + 16 * c), _mm512_loadu_ps(sc + 16 * c)));
        }
}
// exact logit of one row from the row-major int8 table and the float hidden state
static inline float exact_row(const skq_t *q, int v) {
    const int8_t *e = embR + (size_t)v * HID; __m512 acc = _mm512_setzero_ps();
    for (int k = 0; k < HID; k += 16) acc = _mm512_fmadd_ps(_mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i *)(e + k)))), _mm512_loadu_ps(q->h + k), acc);
    return _mm512_reduce_add_ps(acc) * es[v];
}
// threshold keeping ~sketch_k rows: the (sketch_k/64)-th largest of every 64th score
static float sketch_threshold(const float *sc) {
    const int M = sketch_k / 64 + 1; float top[256]; int n = 0;
    for (int v = 0; v < VOCAB; v += 64) {
        float x = sc[v];
        if (n < M) { int i = n++; while (i > 0 && top[i - 1] < x) { top[i] = top[i - 1]; i--; } top[i] = x; }
        else if (x > top[M - 1]) { int i = M - 1; while (i > 0 && top[i - 1] < x) { top[i] = top[i - 1]; i--; } top[i] = x; }
    }
    return top[n - 1];
}

// ---------- forward: one step for all B lanes, every thread runs it with its id ----------
static int stats;
static void forward_mt(int id) {
    static __thread float h[FFN];
    int b0, b1;
    if (!id) { memset(mm_ctr, 0, sizeof mm_ctr); hd_ctr[0] = 0; }   // safe: every thread is past the previous step's last barrier
    if (!id) for (int b = 0; b < B; b++) { const uint16_t *e = emb_bf16 + (size_t)cur_tok[b] * HID; for (int i = 0; i < HID; i++) x[b][i] = bf(e[i]) * 33.941125f; }  // sqrt(1152)
    P(6, sbar_sync(&bar));
    static int chk; if (!id && !chk) chk = getenv("CHECK") != NULL;
    if (!id && chk) { double nn = 0; for (int i = 0; i < HID; i++) nn += x[0][i] * x[0][i]; fprintf(stderr, "  h[0] pos %d: %.4f %.4f %.4f %.4f |x|=%.3f\n", cur_pos[0], x[0][0], x[0][1], x[0][2], x[0][3], sqrt(nn)); }
    for (int l = 0; l < NLAYER; l++) {
        layer_t *ly = &L[l]; int glob = IS_GLOBAL(l);
        if (stats && !id) st_kind = 0;
        for (int b = id; b < B; b += T) { P(3, rmsnorm(x[b], ly->ln_in, h, HID)); P(1, prep(&act[b], h, HID)); }
        if (!id) st_kind = -1;
        P(6, sbar_sync(&bar));
        P(0, tlinear(&ly->qkv, &qkv[0][0], QKV, id, mm_ctr[l * 4 + 0]));
        P(6, sbar_sync(&bar));
        shard(B * NH, id, &b0, &b1);
        for (int i = b0; i < b1; i++) {   // QK-norm + RoPE; the kv head is handled by the lane's head-0 unit
            int b = i / NH, hh = i % NH, pos = cur_pos[b];
            P(3, rmsnorm(q(b) + hh * HD, ly->qn, q(b) + hh * HD, HD); rope(q(b) + hh * HD, pos, glob));
            if (hh == 0) { P(3, rmsnorm(k(b), ly->kn, k(b), HD); rope(k(b), pos, glob)); kv_insert(KVH(b, l), pos, k(b), v(b)); }
        }
        P(6, sbar_sync(&bar));
        for (int i = b0; i < b1; i++) { int b = i / NH, hh = i % NH; P(2, attention_head(b, l, hh, cur_pos[b], att[b])); }
        P(6, sbar_sync(&bar));
        if (stats && !id) st_kind = 1;
        for (int b = id; b < B; b += T) P(1, prep(&act[b], att[b], QDIM));   // quantize attention output for o_proj
        if (!id) st_kind = -1;
        P(6, sbar_sync(&bar));
        P(0, tlinear(&ly->o, &tmp[0][0], HID, id, mm_ctr[l * 4 + 1]));
        P(6, sbar_sync(&bar));
        for (int b = id; b < B; b += T) {   // x += post_attn_norm(o); h = pre_ffn_norm(x); quantize
            P(3, rmsnorm(tmp[b], ly->ln_post_attn, h, HID)); for (int i = 0; i < HID; i++) x[b][i] += h[i];
            if (stats && !id) st_kind = 2;
            P(3, rmsnorm(x[b], ly->ln_pre_ffn, h, HID)); P(1, prep(&act[b], h, HID));
            if (!id) st_kind = -1;
        }
        P(6, sbar_sync(&bar));
        P(0, tlinear(&ly->gu, &gu[0][0], 2 * FFN, id, mm_ctr[l * 4 + 2]));
        P(6, sbar_sync(&bar));
        shard(FFN / 16, id, &b0, &b1); P(4, gelu_rows(b0 * 16, b1 * 16));
        P(6, sbar_sync(&bar));
        if (stats && !id) st_kind = 3;
        for (int b = id; b < B; b += T) P(1, prep(&act[b], g(b), FFN));
        if (!id) st_kind = -1;
        P(6, sbar_sync(&bar));
        P(0, tlinear(&ly->down, &tmp[0][0], HID, id, mm_ctr[l * 4 + 3]));
        P(6, sbar_sync(&bar));
        for (int b = id; b < B; b += T) { P(3, rmsnorm(tmp[b], ly->ln_post_ffn, h, HID)); for (int i = 0; i < HID; i++) x[b][i] += h[i]; }
        P(6, sbar_sync(&bar));
        if (!id && chk && (l < 2 || l == NLAYER - 1)) { double nn = 0; for (int i = 0; i < HID; i++) nn += x[0][i] * x[0][i]; fprintf(stderr, "  h[%d] pos %d: %.4f %.4f %.4f %.4f |x|=%.3f\n", l + 1, cur_pos[0], x[0][0], x[0][1], x[0][2], x[0][3], sqrt(nn)); }
    }
    // sketch head pays off for <= 4 head lanes (decode); bigger heads (batches) amortize the exact head's weight reads
    if (!id) { HB = 0; for (int b = 0; b < B; b++) if (need[b]) hl[HB++] = b; }
    P(6, sbar_sync(&bar));
    const int use_sk = sketch_on && (HB <= 4 || sketch_check);
    for (int j = id; j < HB; j += T) { P(3, rmsnorm(x[hl[j]], ln_final, h, HID)); P(1, prep(&act[j], h, HID)); if (use_sk) sketch_query(&skq[j], h); }
    P(6, sbar_sync(&bar));
    const int8_t *xq[MAXB] = {0}; const float *xbs[MAXB] = {0}; float dq[MAXB] = {0}; float *o[MAXB] = {0};
    for (int j = 0; j < HB; j++) { xq[j] = act[j].q; xbs[j] = act[j].xbs; dq[j] = act[j].dq; o[j] = logits + (size_t)hl[j] * VOCAB; }
    if (use_sk) {
        P(8, for (int u0; (u0 = __atomic_fetch_add(hd_ctr, 16, __ATOMIC_RELAXED)) < VOCAB / 64;) {   // 16 units of 64 rows
            sketch_scores(u0 * 8, (u0 + 16) * 8); sketch_exact(u0 * 8, (u0 + 16) * 8); });
        P(6, sbar_sync(&bar));
        if (!id) for (int j = 0; j < HB; j++) skthr[j] = sketch_threshold(skscore + (size_t)j * VOCAB);
        P(6, sbar_sync(&bar));
    }
    shard(VOCAB / 16, id, &b0, &b1);
    if (use_sk && !sketch_check) {
        P(10, for (int j = 0; j < HB; j++) { const float *sc = skscore + (size_t)j * VOCAB; float *lg = o[j];
            for (int v = b0 * 16; v < b1 * 16; v++) lg[v] = sc[v] >= skthr[j] ? exact_row(&skq[j], v) : -1e30f; });
    } else P(5, mv4(&emb4, xq, xbs, dq, o, HB, b0, b1));
    for (int j = 0; j < HB; j++) {
        const float *lg = o[j];
        if (topk_on) {   // this shard's top TK (sorted insert; almost every value fails the first compare)
            float *tv = tkv + ((size_t)id * MAXB + j) * TK; int *ti = tki + ((size_t)id * MAXB + j) * TK, n = 0;
            for (int vv = b0 * 16; vv < b1 * 16; vv++) {
                float x = lg[vv]; if (n == TK && x <= tv[TK - 1]) continue;
                int i = n < TK ? n++ : TK - 1;
                while (i > 0 && tv[i - 1] < x) { tv[i] = tv[i - 1]; ti[i] = ti[i - 1]; i--; }
                tv[i] = x; ti[i] = vv;
            }
            for (; n < TK; n++) { tv[n] = -1e30f; ti[n] = b0 * 16; }
            best_t[j][id] = ti[0];
        } else { int best = b0 * 16; for (int vv = b0 * 16 + 1; vv < b1 * 16; vv++) if (lg[vv] > lg[best]) best = vv; best_t[j][id] = best; }
    }
    P(6, sbar_sync(&bar));
    if (topk_on) for (int j = id; j < HB; j += T) {   // merge the T sorted shard lists into thread 0's row
        float mv[TK]; int mi[TK], at[256] = {0};
        for (int k = 0; k < TK; k++) {
            int bi = 0; for (int t = 1; t < T; t++) if (tkv[((size_t)t * MAXB + j) * TK + at[t]] > tkv[((size_t)bi * MAXB + j) * TK + at[bi]]) bi = t;
            mv[k] = tkv[((size_t)bi * MAXB + j) * TK + at[bi]]; mi[k] = tki[((size_t)bi * MAXB + j) * TK + at[bi]]; at[bi]++;
        }
        memcpy(tkv + (size_t)j * TK, mv, sizeof mv); memcpy(tki + (size_t)j * TK, mi, sizeof mi);
    }
    if (!id) for (int j = 0; j < HB; j++) { const float *lg = o[j]; for (int i = 1; i < T; i++) if (lg[best_t[j][i]] > lg[best_t[j][0]]) best_t[j][0] = best_t[j][i]; }
    if (!id) for (int j = HB - 1; j >= 0; j--) best_t[hl[j]][0] = best_t[j][0];   // back to lane order (hl[j] >= j)
    if (topk_on) P(6, sbar_sync(&bar));
    if (!id && sketch_check) for (int b = 0; b < B; b++) {   // exact logits are in place: is the true argmax a candidate, and how much softmax mass do candidates hold?
        const float *lg = logits + (size_t)b * VOCAB, *sc = skscore + (size_t)b * VOCAB; float mx = lg[best_t[b][0]]; double z = 0, zin = 0; long ncand = 0;
        for (int v = 0; v < VOCAB; v++) { double e = exp(lg[v] - mx); z += e; if (sc[v] >= skthr[b]) { zin += e; ncand++; } }
        ck_steps++; ck_hit += sc[best_t[b][0]] >= skthr[b]; ck_mass += zin / z; (void)ncand;
        { int a8 = 0, c8 = -1; float l8 = -1e30f, lc = -1e30f;   // exact int8 head over ALL rows vs over the candidates only
          for (int v = 0; v < VOCAB; v++) { float l = exact_row(&skq[b], v); if (l > l8) { l8 = l; a8 = v; } if (sc[v] >= skthr[b] && l > lc) { lc = l; c8 = v; } }
          ck_same8 += a8 == c8; }
        if (getenv("SKDBG") && ck_steps <= 4) { int bv = best_t[b][0], bs = -1; float bl = -1e30f;
            for (int v = 0; v < VOCAB; v++) if (sc[v] >= skthr[b]) { float l = exact_row(&skq[b], v); if (l > bl) { bl = l; bs = v; } }
            fprintf(stderr, "  dbg step %ld: 4-bit argmax %d logit %.3f | int8 exact of it %.3f | best candidate by int8 %d logit %.3f | ncand %ld\n",
                    ck_steps, bv, lg[bv], exact_row(&skq[b], bv), bs, bl, ncand); }
    }
}

// ---------- driver: threads, sampling, speculation, prefill ----------
static void step_need(const int *tok, const int *pos, int *next);
static void step(const int *tok, const int *pos, int *next) {
    for (int b = 0; b < B; b++) { cur_tok[b] = tok[b]; cur_pos[b] = pos[b]; need[b] = 1; }
    step_need(tok, pos, next);
}
static void step_need(const int *tok, const int *pos, int *next) {
    for (int b = 0; b < B; b++) { cur_tok[b] = tok[b]; cur_pos[b] = pos[b]; }
    static double t_out; double t_ = now_ns(); if (t_out) prof[7] += t_ - t_out;
    sbar_sync(&bar); forward_mt(0); sbar_sync(&bar);
    for (int b = 0; b < B; b++) next[b] = best_t[b][0];
    t_out = now_ns();
}

static void generate(const int *prompt, int np, int ngen) {
    int tok[MAXB], posv[MAXB], next[MAXB]; double t0;
    int K = getenv("SPEC") ? atoi(getenv("SPEC")) : 0; if (K > MAXB - 1) K = MAXB - 1;
    static int hist[MAXCTX]; int n = 0;
    for (int i = 0; i < np; i++) hist[n++] = prompt[i];
    const int nstreams = B;
    t0 = now_ns(); double nll = 0; const int ppl = getenv("PPL") != NULL;
    for (int i0 = 0; i0 < np; i0 += MAXB / nstreams) {   // batched prefill
        int cnt = np - i0 < MAXB / nstreams ? np - i0 : MAXB / nstreams; B = cnt * nstreams;
        for (int b = 0; b < B; b++) { kv_map[b] = b / cnt; tok[b] = hist[i0 + b % cnt]; posv[b] = i0 + b % cnt; need[b] = ppl || b % cnt == cnt - 1; }
        step_need(tok, posv, next);   // prefill: only each stream's last lane needs the output head (all of them for PPL)
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
    { const float *lg = logits + (size_t)((np - 1) % (MAXB / nstreams)) * VOCAB; fprintf(stderr, "top5:"); static float tmpl[VOCAB]; memcpy(tmpl, lg, sizeof tmpl);
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
    if (getenv("PROF")) for (int i = 0; i < 11; i++) if (*pname[i]) fprintf(stderr, "  %-10s %7.2f ms/step (thread 0)\n", pname[i], prof[i] / 1e6 / (steps + (np + MAXB - 1) / MAXB));
    if (nstreams > 1) fprintf(stderr, "all %d streams identical: %s\n", nstreams, same ? "yes" : "NO");
    if (K && nstreams == 1) fprintf(stderr, "speculation K=%d: %d steps, %d drafted, %d accepted (%.0f%%), %.2f tokens/step\n", K, steps, drafted, accepted, drafted ? 100.0 * accepted / drafted : 0, (double)gen / (steps ? steps : 1));
    fprintf(stderr, "prefill %d tok in %.1f ms (%.0f tok/s), decode %d tok x %d streams in %.1f ms: %.1f tok/s per stream, %.0f tok/s aggregate\n",
            np, tp / 1e6, np / (tp / 1e9), gen, nstreams, tg / 1e6, gen / (tg / 1e9), (double)gen * nstreams / (tg / 1e9));
}

// load the model, allocate nseq KV streams, start the worker threads (the caller becomes thread 0)
static pthread_t th[256];
static void init(const char *dir, int nseq) {
    int nc = find_cores(cpus, 256);
    T = getenv("THREADS") ? atoi(getenv("THREADS")) : nc; if (T > nc) for (int i = 0; i < T && i < 256; i++) cpus[i] = i;
    nkv = nseq; kv_alloc(); logits = xalloc((size_t)MAXB * VOCAB * 4);
    double t0 = now_ns();
    stats = getenv("STATS") != NULL;
    load(dir); make_tables(); make_perm(perm_h, HID); make_perm(perm_q, QDIM); make_perm(perm_f, FFN);
    char nm[160];
#define TN(t) (snprintf(nm, sizeof nm, "model.layers.%d." t ".weight", l), tensor(nm))
#define VN(t, n) (snprintf(nm, sizeof nm, "model.layers.%d." t ".weight", l), vecf(nm, n))
    for (int l = 0; l < NLAYER; l++) {
#define TQ(t, n, k) (snprintf(nm, sizeof nm, "model.layers.%d." t ".weight", l), trits(nm, n, k))
#define TS(t) (snprintf(nm, sizeof nm, "model.layers.%d." t ".scale", l), *(const float *)tensor_raw(nm))
        lmat_alloc(&L[l].qkv, QKV, HID); pack_ternary(&L[l].qkv, 0, TQ("self_attn.q_proj", QDIM, HID), TS("self_attn.q_proj"), QDIM, HID);
        pack_ternary(&L[l].qkv, QDIM, TQ("self_attn.k_proj", KVDIM, HID), TS("self_attn.k_proj"), KVDIM, HID);
        pack_ternary(&L[l].qkv, QDIM + KVDIM, TQ("self_attn.v_proj", KVDIM, HID), TS("self_attn.v_proj"), KVDIM, HID);
        lmat_alloc(&L[l].o, HID, QDIM); pack_ternary(&L[l].o, 0, TQ("self_attn.o_proj", HID, QDIM), TS("self_attn.o_proj"), HID, QDIM);
        lmat_alloc(&L[l].gu, 2 * FFN, HID); pack_ternary(&L[l].gu, 0, TQ("mlp.gate_proj", FFN, HID), TS("mlp.gate_proj"), FFN, HID);
        pack_ternary(&L[l].gu, FFN, TQ("mlp.up_proj", FFN, HID), TS("mlp.up_proj"), FFN, HID);
        lmat_alloc(&L[l].down, HID, FFN); pack_ternary(&L[l].down, 0, TQ("mlp.down_proj", HID, FFN), TS("mlp.down_proj"), HID, FFN);
        L[l].ln_in = VN("input_layernorm", HID); L[l].ln_post_attn = VN("post_attention_layernorm", HID);
        L[l].ln_pre_ffn = VN("pre_feedforward_layernorm", HID); L[l].ln_post_ffn = VN("post_feedforward_layernorm", HID);
        L[l].qn = VN("self_attn.q_norm", HD); L[l].kn = VN("self_attn.k_norm", HD);
    }
    emb_bf16 = tensor("model.embed_tokens.weight"); ln_final = vecf("model.norm.weight", HID);
    qmat_alloc(&emb4, VOCAB, HID); quant4(&emb4, 0, emb_bf16, VOCAB, HID);
    // default on; EXACT=1 forces the exact 4-bit head; perplexity needs full logits
    sketch_on = (!getenv("EXACT") && !getenv("PPL")) || getenv("SKETCHCHECK") != NULL; sketch_check = getenv("SKETCHCHECK") != NULL;
    if (getenv("SKETCH_K")) sketch_k = atoi(getenv("SKETCH_K"));
    if (sketch_on) sketch_build();
    for (int gl = 0; gl < 2; gl++) { float base = gl ? 1000000.0f : 10000.0f;
        for (int p = 0; p < MAXCTX; p++) for (int i = 0; i < HD / 2; i++) { float ang = p * powf(base, -2.0f * i / HD); rope_c[gl][p][i] = cosf(ang); rope_s[gl][p][i] = sinf(ang); } }
    fprintf(stderr, "loaded gemma-3-1b-it-ternary in %.1f s, %d threads, %d kv streams\n", (now_ns() - t0) / 1e9, T, nkv);
    sbar_init(&bar, T); pin(cpus[0]);
    for (long i = 1; i < T; i++) pthread_create(&th[i], NULL, worker, (void *)i);
}

#ifdef CITRIC_LIB
// ---------- library API (the HTTP server links this file with -DCITRIC_LIB). Call everything from one thread. ----------
int citric_init(const char *dir, int max_seqs) {
    init(dir, max_seqs); topk_on = 1; tkv = xalloc((size_t)T * MAXB * TK * 4); tki = xalloc((size_t)T * MAXB * TK * 4);
    return VOCAB;
}
int citric_max_lanes(void) { return MAXB; }
int citric_max_ctx(void) { return MAXCTX; }
// One forward step over n <= MAXB lanes. Lane i feeds token tok[i] at position pos[i] of sequence seq[i] (a KV
// stream < max_seqs); lanes of one sequence may cover consecutive positions (chunked prefill). Afterwards lane i's
// next-token logits are at citric_logits() + i * VOCAB and its argmax in next[i]. With <= 4 lanes the sketch head
// runs: logits outside its ~1k candidates are -1e30.
// want[i] = 0: lane i's logits are not needed (a prefill token before the prompt's last): its head is skipped,
// next[i] and its logits row are undefined. After the call, citric_topk(i, &v, &id) gives lane i's TK best
// (logit, token) pairs, best first, for every lane with want[i] = 1.
void citric_step(int n, const int *tok, const int *pos, const int *seq, const int *want, int *next) {
    B = n; for (int b = 0; b < n; b++) { kv_map[b] = seq[b]; need[b] = want[b]; }
    step_need(tok, pos, next);
}
const float *citric_logits(void) { return logits; }
// Copy the KV cache of positions 0..n-1 from sequence slot src to dst (prefix reuse: a shared system prompt or an
// earlier turn of the same chat). Whole 16-position blocks are copied; dst positions >= n are rewritten before use.
void citric_kv_copy(int src, int dst, int n) {
    size_t n16 = (size_t)(n + 15) / 16 * 16;
    for (int l = 0; l < NLAYER; l++) {
        kvh_t *a = &kvh[(size_t)src * NLAYER + l], *b = &kvh[(size_t)dst * NLAYER + l];
        memcpy(b->k8, a->k8, n16 * HD); memcpy(b->v8, a->v8, n16 * HD);
        memcpy(b->ksc, a->ksc, n16 * 4); memcpy(b->vsc, a->vsc, n16 * 4); memcpy(b->ksum, a->ksum, n16 * 4);
    }
}
int citric_topk(int lane, const float **v, const int **ids) {   // lane -> its head slot
    int j = 0; while (j < HB && hl[j] != lane) j++;
    *v = tkv + (size_t)j * TK; *ids = tki + (size_t)j * TK; return TK;
}
#else
int main(int argc, char **argv) {
    if (argc < 4 && !(argc == 3 && !strcmp(argv[2], "-"))) { fprintf(stderr, "usage: %s model_dir n_gen tok...  |  %s model_dir -\n", argv[0], argv[0]); return 1; }
    B = getenv("BATCH") ? atoi(getenv("BATCH")) : 1; if (B > MAXB) B = MAXB;
    if (getenv("TEMP")) temp = atof(getenv("TEMP"));
    if (getenv("REP")) rep = atof(getenv("REP"));
    if (getenv("TOPK")) topk = atoi(getenv("TOPK"));
    if (getenv("SEED")) srng = strtoull(getenv("SEED"), NULL, 10) * 2654435761ull + 1;
    init(argv[1], B);
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
    if (sketch_check && ck_steps) fprintf(stderr, "sketch K=%d: 4-bit argmax among candidates %ld/%ld (%.1f%%), softmax mass captured %.4f; sketch picks the full int8 head's argmax %ld/%ld (%.1f%%)\n", sketch_k, ck_hit, ck_steps, 100.0 * ck_hit / ck_steps, ck_mass / ck_steps, ck_same8, ck_steps, 100.0 * ck_same8 / ck_steps);
    if (stats) { const char *nm4[4] = {"qkv input", "o input", "gate|up input", "down input"};
        for (int i = 0; i < 4; i++) fprintf(stderr, "  %-14s %5.1f%% of groups of 4 are all zero (%ld groups)\n", nm4[i], 100.0 * st_zero[i] / (st_groups[i] ? st_groups[i] : 1), st_groups[i]); }
    quit = 1; sbar_sync(&bar); for (int i = 1; i < T; i++) pthread_join(th[i], NULL);
    return 0;
}
#endif
