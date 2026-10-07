#pragma once
#include <stdio.h>
#include <time.h>

static inline double now_ns(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec * 1e9 + t.tv_nsec;
}

// Runs body `iters` times, 5 reps, stores best (min) ns/iter into `out`.
#define BENCH(out, iters, body) do {                                   \
    double best_ = 1e300;                                              \
    for (int rep_ = 0; rep_ < 5; rep_++) {                             \
        double t0_ = now_ns();                                         \
        for (int it_ = 0; it_ < (iters); it_++) {                      \
            body;                                                      \
            __asm__ volatile("" ::: "memory");                         \
        }                                                              \
        double t_ = (now_ns() - t0_) / (iters);                        \
        if (t_ < best_) best_ = t_;                                    \
    }                                                                  \
    (out) = best_;                                                     \
} while (0)
