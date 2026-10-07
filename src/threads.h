// Pinned threads + split-phase spin barrier (arrive now, wait later).
#pragma once
#define _GNU_SOURCE
#include <immintrin.h>
#include <pthread.h>
#include <sched.h>
#include <stdatomic.h>
#include <stdio.h>

typedef struct {
    _Alignas(64) atomic_int count;
    _Alignas(64) atomic_int gen;
    int n;
} sbar_t;

static void sbar_init(sbar_t *b, int n) { atomic_store(&b->count, 0); atomic_store(&b->gen, 0); b->n = n; }
static int sbar_arrive(sbar_t *b) {
    int g = atomic_load(&b->gen);
    if (atomic_fetch_add(&b->count, 1) == b->n - 1) { atomic_store(&b->count, 0); atomic_store(&b->gen, g + 1); }
    return g;
}
static void sbar_wait(sbar_t *b, int g) { while (atomic_load(&b->gen) == g) _mm_pause(); }
static void sbar_sync(sbar_t *b) { sbar_wait(b, sbar_arrive(b)); }

// first logical cpu of each physical core; returns count
static int find_cores(int *cpus, int max) {
    int seen[256] = {0}, n = 0;
    for (int c = 0; c < 256 && n < max; c++) {
        char p[128]; snprintf(p, sizeof p, "/sys/devices/system/cpu/cpu%d/topology/core_id", c);
        FILE *f = fopen(p, "r"); if (!f) break;
        int core; if (fscanf(f, "%d", &core) == 1 && core < 256 && !seen[core]) { seen[core] = 1; cpus[n++] = c; }
        fclose(f);
    }
    return n;
}
static void pin(int cpu) { cpu_set_t cs; CPU_ZERO(&cs); CPU_SET(cpu, &cs); pthread_setaffinity_np(pthread_self(), sizeof cs, &cs); }
