// Citric engine for google/gemma-3-1b-it (Gemma 3 text, 1B, instruction tuned).
// Architecture: 26 layers, hidden 1152, ffn 6912 GeGLU (tanh-GELU), 4 q heads x 256 / 1 kv head (GQA),
// QK-norm, sandwich RMSNorms with (1+w) scale, embeddings scaled by sqrt(hidden), tied lm_head over a
// 262144 vocabulary. Sliding window 512 on 5 of every 6 layers (RoPE base 10k), global layers base 1M.
// Weights: 4-bit, per-row per-32-block fp16 scales, weights as the unsigned vpdpbusd operand so the only
// compensation is 8 * sum(x_block). Activations int8 per token. KV int8, head-major token-blocked.
// Lanes = (token, position, kv stream): prefill, batching and speculation are one code path.
// usage: ./gemma models/gemma-3-1b-it N_GEN tok tok ...   |   ./gemma models/gemma-3-1b-it -   (stdin: "ngen id id ..." per line)

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
#include "gemma3.h"

typedef struct { qmat_t qkv, o, gu, down; float *ln_in, *ln_post_attn, *ln_pre_ffn, *ln_post_ffn, *qn, *kn; } layer_t;

static layer_t L[NLAYER];
static qmat_t emb4;                 // tied embeddings as the lm_head
static const uint16_t *emb_bf16;    // embedding lookup rows (mmap)
static float *ln_final;
static double prof[8]; static const char *pname[6] = {"matvec", "quant", "attention", "rope/norm", "gelu", "lm_head"};
#define P(i, stmt) do { double t_ = now_ns(); stmt; if (!id) prof[i] += now_ns() - t_; } while (0)

// ---------- safetensors (bf16) ----------
static char *hdr; static uint8_t *data;
static const uint16_t *tensor(const char *name) {
    char key[256]; snprintf(key, sizeof key, "\"%s\"", name);
    char *p = strstr(hdr, key); if (!p) { fprintf(stderr, "missing tensor %s\n", name); exit(1); }
    p = strstr(p, "\"data_offsets\""); p = strchr(p, '[');
    return (const uint16_t *)(data + strtoul(p + 1, NULL, 10));
}
static void load(const char *dir) {
    char path[512]; snprintf(path, sizeof path, "%s/model.safetensors", dir);
    int fd = open(path, O_RDONLY); if (fd < 0) { perror(path); exit(1); }
    struct stat st; fstat(fd, &st);
    uint8_t *m = mmap(NULL, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    uint64_t n = *(uint64_t *)m; hdr = malloc(n + 1); memcpy(hdr, m + 8, n); hdr[n] = 0; data = m + 8 + n;
}
static float *vecf(const char *name, int n) { const uint16_t *t = tensor(name); float *f = xalloc(n * 4); for (int i = 0; i < n; i++) f[i] = bf(t[i]); return f; }

// per-token absmax int8 + per-32-block sums. returns dequant factor.
typedef struct { int8_t q[FFN]; float xbs[FFN / 32]; float dq; } act_t;
static act_t act[MAXB];
static void prep(act_t *a, const float *xf, int n) {
    __m512 am = _mm512_set1_ps(1e-8f);
    for (int i = 0; i < n; i += 16) am = _mm512_max_ps(am, _mm512_abs_ps(_mm512_loadu_ps(xf + i)));
    float sc = 127.0f / _mm512_reduce_max_ps(am); __m512 sv = _mm512_set1_ps(sc);
    for (int i = 0; i < n; i += 16) _mm_storeu_si128((__m128i *)(a->q + i), _mm512_cvtsepi32_epi8(_mm512_cvtps_epi32(_mm512_mul_ps(_mm512_loadu_ps(xf + i), sv))));
    for (int blk = 0; blk < n / 32; blk++) { int s = 0; for (int i = 0; i < 32; i++) s += a->q[blk * 32 + i]; a->xbs[blk] = s; }
    a->dq = 1.0f / sc;
}
static void linear(const qmat_t *m, float *out, int stride, int id) {
    int b0, b1; shard(m->n / 16, id, &b0, &b1);
    const int8_t *xq[MAXB] = {0}; const float *xbs[MAXB] = {0}; float dq[MAXB] = {0}; float *o[MAXB] = {0};
    for (int b = 0; b < B; b++) { xq[b] = act[b].q; xbs[b] = act[b].xbs; dq[b] = act[b].dq; o[b] = out + (size_t)b * stride; }
    mv4(m, xq, xbs, dq, o, B, b0, b1);
}

// ---------- forward: one step for all B lanes, every thread runs it with its id ----------
static void forward_mt(int id) {
    static __thread float h[FFN];
    int b0, b1;
    if (!id) for (int b = 0; b < B; b++) { const uint16_t *e = emb_bf16 + (size_t)cur_tok[b] * HID; for (int i = 0; i < HID; i++) x[b][i] = bf(e[i]) * 33.941125f; }  // sqrt(1152)
    sbar_sync(&bar);
    static int chk; if (!id && !chk) chk = getenv("CHECK") != NULL;
    if (!id && chk) { double nn = 0; for (int i = 0; i < HID; i++) nn += x[0][i] * x[0][i]; fprintf(stderr, "  h[0] pos %d: %.4f %.4f %.4f %.4f |x|=%.3f\n", cur_pos[0], x[0][0], x[0][1], x[0][2], x[0][3], sqrt(nn)); }
    for (int l = 0; l < NLAYER; l++) {
        layer_t *ly = &L[l]; int glob = IS_GLOBAL(l);
        for (int b = id; b < B; b += T) { P(3, rmsnorm(x[b], ly->ln_in, h, HID)); P(1, prep(&act[b], h, HID)); }
        sbar_sync(&bar);
        P(0, linear(&ly->qkv, &qkv[0][0], QKV, id));
        sbar_sync(&bar);
        shard(B * NH, id, &b0, &b1);
        for (int i = b0; i < b1; i++) {   // QK-norm + RoPE; the kv head is handled by the lane's head-0 unit
            int b = i / NH, hh = i % NH, pos = cur_pos[b];
            P(3, rmsnorm(q(b) + hh * HD, ly->qn, q(b) + hh * HD, HD); rope(q(b) + hh * HD, pos, glob));
            if (hh == 0) { P(3, rmsnorm(k(b), ly->kn, k(b), HD); rope(k(b), pos, glob)); kv_insert(KVH(b, l), pos, k(b), v(b)); }
        }
        sbar_sync(&bar);
        for (int i = b0; i < b1; i++) { int b = i / NH, hh = i % NH; P(2, attention_head(b, l, hh, cur_pos[b], att[b])); }
        sbar_sync(&bar);
        for (int b = id; b < B; b += T) P(1, prep(&act[b], att[b], QDIM));   // quantize attention output for o_proj
        sbar_sync(&bar);
        P(0, linear(&ly->o, &tmp[0][0], HID, id));
        sbar_sync(&bar);
        for (int b = id; b < B; b += T) {   // x += post_attn_norm(o); h = pre_ffn_norm(x); quantize
            P(3, rmsnorm(tmp[b], ly->ln_post_attn, h, HID)); for (int i = 0; i < HID; i++) x[b][i] += h[i];
            P(3, rmsnorm(x[b], ly->ln_pre_ffn, h, HID)); P(1, prep(&act[b], h, HID));
        }
        sbar_sync(&bar);
        P(0, linear(&ly->gu, &gu[0][0], 2 * FFN, id));
        sbar_sync(&bar);
        shard(FFN / 16, id, &b0, &b1); P(4, gelu_rows(b0 * 16, b1 * 16));
        sbar_sync(&bar);
        for (int b = id; b < B; b += T) P(1, prep(&act[b], g(b), FFN));
        sbar_sync(&bar);
        P(0, linear(&ly->down, &tmp[0][0], HID, id));
        sbar_sync(&bar);
        for (int b = id; b < B; b += T) { P(3, rmsnorm(tmp[b], ly->ln_post_ffn, h, HID)); for (int i = 0; i < HID; i++) x[b][i] += h[i]; }
        sbar_sync(&bar);
        if (!id && chk && (l < 2 || l == NLAYER - 1)) { double nn = 0; for (int i = 0; i < HID; i++) nn += x[0][i] * x[0][i]; fprintf(stderr, "  h[%d] pos %d: %.4f %.4f %.4f %.4f |x|=%.3f\n", l + 1, cur_pos[0], x[0][0], x[0][1], x[0][2], x[0][3], sqrt(nn)); }
    }
    for (int b = id; b < B; b += T) { P(3, rmsnorm(x[b], ln_final, h, HID)); P(1, prep(&act[b], h, HID)); }
    sbar_sync(&bar);
    shard(VOCAB / 16, id, &b0, &b1);
    { const int8_t *xq[MAXB] = {0}; const float *xbs[MAXB] = {0}; float dq[MAXB] = {0}; float *o[MAXB] = {0};
      for (int b = 0; b < B; b++) { xq[b] = act[b].q; xbs[b] = act[b].xbs; dq[b] = act[b].dq; o[b] = logits + (size_t)b * VOCAB; }
      P(5, mv4(&emb4, xq, xbs, dq, o, B, b0, b1)); }
    for (int b = 0; b < B; b++) { const float *lg = logits + (size_t)b * VOCAB; int best = b0 * 16; for (int vv = b0 * 16 + 1; vv < b1 * 16; vv++) if (lg[vv] > lg[best]) best = vv; best_t[b][id] = best; }
    sbar_sync(&bar);
    if (!id) for (int b = 0; b < B; b++) { const float *lg = logits + (size_t)b * VOCAB; for (int i = 1; i < T; i++) if (lg[best_t[b][i]] > lg[best_t[b][0]]) best_t[b][0] = best_t[b][i]; }
}

// ---------- driver: threads, sampling, speculation, prefill ----------
static void step(const int *tok, const int *pos, int *next) {
    for (int b = 0; b < B; b++) { cur_tok[b] = tok[b]; cur_pos[b] = pos[b]; }
    sbar_sync(&bar); forward_mt(0); sbar_sync(&bar);
    for (int b = 0; b < B; b++) next[b] = best_t[b][0];
}

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
    if (getenv("SEED")) srng = strtoull(getenv("SEED"), NULL, 10) * 2654435761ull + 1;
    nkv = B; kv_alloc(); logits = xalloc((size_t)MAXB * VOCAB * 4);
    double t0 = now_ns();
    load(argv[1]);
    char nm[160];
#define TN(t) (snprintf(nm, sizeof nm, "model.layers.%d." t ".weight", l), tensor(nm))
#define VN(t, n) (snprintf(nm, sizeof nm, "model.layers.%d." t ".weight", l), vecf(nm, n))
    for (int l = 0; l < NLAYER; l++) {
        qmat_alloc(&L[l].qkv, QKV, HID); quant4(&L[l].qkv, 0, TN("self_attn.q_proj"), QDIM, HID); quant4(&L[l].qkv, QDIM, TN("self_attn.k_proj"), KVDIM, HID); quant4(&L[l].qkv, QDIM + KVDIM, TN("self_attn.v_proj"), KVDIM, HID);
        qmat_alloc(&L[l].o, HID, QDIM); quant4(&L[l].o, 0, TN("self_attn.o_proj"), HID, QDIM);
        qmat_alloc(&L[l].gu, 2 * FFN, HID); quant4(&L[l].gu, 0, TN("mlp.gate_proj"), FFN, HID); quant4(&L[l].gu, FFN, TN("mlp.up_proj"), FFN, HID);
        qmat_alloc(&L[l].down, HID, FFN); quant4(&L[l].down, 0, TN("mlp.down_proj"), HID, FFN);
        L[l].ln_in = VN("input_layernorm", HID); L[l].ln_post_attn = VN("post_attention_layernorm", HID);
        L[l].ln_pre_ffn = VN("pre_feedforward_layernorm", HID); L[l].ln_post_ffn = VN("post_feedforward_layernorm", HID);
        L[l].qn = VN("self_attn.q_norm", HD); L[l].kn = VN("self_attn.k_norm", HD);
    }
    emb_bf16 = tensor("model.embed_tokens.weight"); ln_final = vecf("model.norm.weight", HID);
    qmat_alloc(&emb4, VOCAB, HID); quant4(&emb4, 0, emb_bf16, VOCAB, HID);
    for (int gl = 0; gl < 2; gl++) { float base = gl ? 1000000.0f : 10000.0f;
        for (int p = 0; p < MAXCTX; p++) for (int i = 0; i < HD / 2; i++) { float ang = p * powf(base, -2.0f * i / HD); rope_c[gl][p][i] = cosf(ang); rope_s[gl][p][i] = sinf(ang); } }
    fprintf(stderr, "loaded gemma-3-1b-it in %.1f s, %d threads, batch %d\n", (now_ns() - t0) / 1e9, T, B);

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
