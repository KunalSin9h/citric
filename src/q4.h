// The 4-bit block-32 kernel (gemma.c, gemma3t.c lm_head): weights+8 as the unsigned vpdpbusd operand, fp16 scale per
// 32-block, int8 activations with per-32-block sums for the +8 compensation.
#pragma once
// 4-bit matrix: w [n/16][k/8][16 rows][4 bytes] nibbles (w+8): low nibble k=8g+i, high nibble k=8g+4+i.
// sc fp16 [n/16][k/32][16 rows].
typedef struct { int n, k; uint8_t *w; uint16_t *sc; } qmat_t;
static void qmat_alloc(qmat_t *d, int n, int k) { d->n = n; d->k = k; d->w = xalloc((size_t)n * k / 2); d->sc = xalloc((size_t)n * (k / 32) * 2); }
// quantize rows r0.. of d from a bf16 [n][k] tensor: 4-bit symmetric per 32-block, fp16 scale
static void quant4(qmat_t *d, int r0, const uint16_t *w, int n, int k) {
    const int G8 = k / 8, NB = k / 32;
    for (int r = 0; r < n; r++) {
        const uint16_t *row = w + (size_t)r * k; int rr = r0 + r;
        for (int blk = 0; blk < NB; blk++) {
            float am = 1e-8f; for (int i = 0; i < 32; i++) am = fmaxf(am, fabsf(bf(row[blk * 32 + i])));
            float scl = am / 7.0f; d->sc[((size_t)(rr / 16) * NB + blk) * 16 + rr % 16] = _cvtss_sh(scl, 0);
            scl = bf(0) + (float)_cvtsh_ss(_cvtss_sh(scl, 0));   // quantize with the stored (fp16-rounded) scale
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
// 16-row block nb of out[b] = W @ x[b] for BB streams. 4-bit weights as the u8 operand, x int8 as the s8 operand:
// per 32-block, acc = sum (w+8) x = sum w x + 8 sum x  ->  subtract 8*xbs, scale by the fp16 block scale.
static inline __attribute__((always_inline)) void mv4_block(const qmat_t *m, const int8_t *const *xq, const float *const *xbs, const float *dq, float *const *out, int b0, int nb, const int BB) {
    const int G8 = m->k / 8, NB = m->k / 32;
    __m512 facc[BB]; for (int b = 0; b < BB; b++) facc[b] = _mm512_setzero_ps();
    const uint8_t *wb = m->w + (size_t)nb * G8 * 64; const __m512i m4 = _mm512_set1_epi8(0x0F);
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
        __m512 sc = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)(m->sc + ((size_t)nb * NB + blk) * 16)));
        for (int b = 0; b < BB; b++)
            facc[b] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(_mm512_sub_epi32(acc[b], _mm512_set1_epi32(8 * (int)xbs[b0 + b][blk]))), sc, facc[b]);
    }
    for (int b = 0; b < BB; b++) _mm512_storeu_ps(out[b0 + b] + nb * 16, _mm512_mul_ps(facc[b], _mm512_set1_ps(dq[b0 + b])));
}
static void mv4(const qmat_t *m, const int8_t *const *xq, const float *const *xbs, const float *dq, float *const *out, int nstreams, int nb0, int nb1) {
    for (int nb = nb0; nb < nb1; nb++)
        for (int b0 = 0; b0 < nstreams; b0 += 8)
            switch (nstreams - b0 >= 8 ? 8 : nstreams - b0) {
            case 1: mv4_block(m, xq, xbs, dq, out, b0, nb, 1); break;
            case 2: mv4_block(m, xq, xbs, dq, out, b0, nb, 2); break;
            case 3: mv4_block(m, xq, xbs, dq, out, b0, nb, 3); break;
            case 4: mv4_block(m, xq, xbs, dq, out, b0, nb, 4); break;
            case 5: mv4_block(m, xq, xbs, dq, out, b0, nb, 5); break;
            case 6: mv4_block(m, xq, xbs, dq, out, b0, nb, 6); break;
            case 7: mv4_block(m, xq, xbs, dq, out, b0, nb, 7); break;
            default: mv4_block(m, xq, xbs, dq, out, b0, nb, 8); break;
            }
}
