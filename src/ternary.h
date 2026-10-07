// The 1.6-bit ternary kernel (bitnet.c, gemma3t.c): 5 trits per byte, decoded with vpermi2b tables into int8
// operands for vpdpbusd against permuted int8 activations (+128). Layout: [n/16][kp/20][16 rows][4 bytes].
#pragma once
typedef struct { int n, k, kp; uint8_t *w; int32_t *rs; float *scale; } lmat_t;   // rs = row trit sums, scale per 16-row block
#define KPAD(k) (((k) + 19) / 20 * 20)
static void make_perm(int16_t *p, int k) { for (int j = 0; j < KPAD(k); j++) { int qd = j / 20, i = j % 20 / 4, bp = j % 4, kk = (qd * 4 + bp) * 5 + i; p[j] = kk < k ? kk : -1; } }
// trit decode tables: T[i][v] = (v / 3^i) % 3 - 1 for v < 243, else 0. Four 64-byte quarters each.
static __m512i trit_tab[5][4];
static void make_tables(void) {
    for (int i = 0; i < 5; i++) {
        int8_t t[256]; int p = 1; for (int j = 0; j < i; j++) p *= 3;
        for (int v = 0; v < 256; v++) t[v] = v < 243 ? (int8_t)((v / p) % 3 - 1) : 0;
        for (int qq = 0; qq < 4; qq++) trit_tab[i][qq] = _mm512_loadu_si512(t + qq * 64);
    }
}
// 64 packed bytes (16 rows x 4 positions) -> int8 operand for plane i: lane r = 4 trits (positions 0..3)
static inline __m512i decode_plane(__m512i v, int i) {
    __m512i lo = _mm512_permutex2var_epi8(trit_tab[i][0], v, trit_tab[i][1]);   // values 0..127 (7-bit index)
    __m512i hi = _mm512_permutex2var_epi8(trit_tab[i][2], v, trit_tab[i][3]);   // values 128..255
    return _mm512_mask_blend_epi8(_mm512_movepi8_mask(v), lo, hi);
}
// One slab (SLAB quads = SLAB*5 operands) of a 16-row block for BB streams. y[b]+nb*16 holds the
// running int32 accumulator across slabs (zeroed before the first slab, compensated after the last).
#ifndef SLAB
#define SLAB 64
#endif
static inline __attribute__((always_inline)) void mv_slab(const uint8_t *const *xp, int32_t *const *y, int nb, int j0, int j1, const int8_t *buf, const int BB) {
    __m512i acc[BB]; for (int b = 0; b < BB; b++) acc[b] = _mm512_loadu_si512(y[b] + nb * 16);
    for (int j = j0; j < j1; j++) {
        __m512i w8 = _mm512_loadu_si512(buf + (j - j0) * 64);
        for (int b = 0; b < BB; b++) acc[b] = _mm512_dpbusd_epi32(acc[b], _mm512_set1_epi32(*(const int32_t *)(xp[b] + 4 * j)), w8);
    }
    for (int b = 0; b < BB; b++) _mm512_storeu_si512(y[b] + nb * 16, acc[b]);
}
// Row blocks nb0..nb1 for nb_streams streams: decode SLAB quads to int8 (5 KB, stays in L1), run every
// stream over the slab, repeat. Each weight byte is read from DRAM once and decoded once per step.
static void mv_b(const lmat_t *m, const uint8_t *const *xp, int32_t *const *y, int nb_streams, int nb0, int nb1) {
    int8_t buf[SLAB * 5 * 64] __attribute__((aligned(64)));
    const int Q = m->kp / 20;
    for (int nb = nb0; nb < nb1; nb++) {
        const uint8_t *wb = m->w + (size_t)nb * Q * 64;
        for (int b = 0; b < nb_streams; b++) _mm512_storeu_si512(y[b] + nb * 16, _mm512_setzero_si512());
        for (int q0 = 0; q0 < Q; q0 += SLAB) {
            int q1 = q0 + SLAB < Q ? q0 + SLAB : Q;
            for (int qd = q0; qd < q1; qd++) {
                __m512i v = _mm512_loadu_si512(wb + qd * 64);
                for (int i = 0; i < 5; i++) _mm512_storeu_si512(buf + ((qd - q0) * 5 + i) * 64, decode_plane(v, i));
            }
            for (int b0 = 0; b0 < nb_streams; b0 += 8)
                switch (nb_streams - b0 >= 8 ? 8 : nb_streams - b0) {
                case 1: mv_slab(xp + b0, y + b0, nb, q0 * 5, q1 * 5, buf, 1); break;
                case 2: mv_slab(xp + b0, y + b0, nb, q0 * 5, q1 * 5, buf, 2); break;
                case 3: mv_slab(xp + b0, y + b0, nb, q0 * 5, q1 * 5, buf, 3); break;
                case 4: mv_slab(xp + b0, y + b0, nb, q0 * 5, q1 * 5, buf, 4); break;
                case 5: mv_slab(xp + b0, y + b0, nb, q0 * 5, q1 * 5, buf, 5); break;
                case 6: mv_slab(xp + b0, y + b0, nb, q0 * 5, q1 * 5, buf, 6); break;
                case 7: mv_slab(xp + b0, y + b0, nb, q0 * 5, q1 * 5, buf, 7); break;
                default: mv_slab(xp + b0, y + b0, nb, q0 * 5, q1 * 5, buf, 8); break;
                }
        }
        __m512i comp = _mm512_slli_epi32(_mm512_loadu_si512(m->rs + nb * 16), 7);
        for (int b = 0; b < nb_streams; b++) _mm512_storeu_si512(y[b] + nb * 16, _mm512_sub_epi32(_mm512_loadu_si512(y[b] + nb * 16), comp));
    }
}
