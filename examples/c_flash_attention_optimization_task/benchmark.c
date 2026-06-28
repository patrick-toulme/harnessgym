#include "kernel.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifndef N
#define N 192
#endif

#ifndef D
#define D 64
#endif

#ifndef RUNS
#define RUNS 9
#endif

#ifndef CASE_SEED
#define CASE_SEED 0xC0FFEEu
#endif

#if defined(__x86_64__) || defined(_M_X64)
static inline uint64_t read_cycles(void) {
    unsigned int lo = 0;
    unsigned int hi = 0;
    __asm__ __volatile__("rdtsc" : "=a"(lo), "=d"(hi));
    return ((uint64_t)hi << 32) | lo;
}
#elif defined(__aarch64__)
static inline uint64_t read_cycles(void) {
    uint64_t value = 0;
    __asm__ __volatile__("mrs %0, cntvct_el0" : "=r"(value));
    return value;
}
#else
static inline uint64_t read_cycles(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}
#endif

static uint32_t rng_state = 0xC0FFEEu;

static float next_float(void) {
    rng_state = 1664525u * rng_state + 1013904223u;
    uint32_t bits = (rng_state >> 9) | 0x3f800000u;
    float value = 0.0f;
    memcpy(&value, &bits, sizeof(value));
    return (value - 1.0f) * 2.0f - 1.0f;
}

static void fill_random(float *data, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        data[i] = next_float() * 0.5f;
    }
}

static void make_case(float *q, float *k, float *v, size_t count, uint32_t seed) {
    rng_state = seed;
    fill_random(q, count);
    fill_random(k, count);
    fill_random(v, count);
}

static void reference_flash_attention(
    const float *q,
    const float *k,
    const float *v,
    float *out,
    int n,
    int d,
    float scale
) {
    double *scores = (double *)malloc((size_t)n * sizeof(double));
    for (int i = 0; i < n; ++i) {
        const float *qi = q + (size_t)i * (size_t)d;
        double max_score = -INFINITY;
        for (int j = 0; j < n; ++j) {
            const float *kj = k + (size_t)j * (size_t)d;
            double dot = 0.0;
            for (int x = 0; x < d; ++x) {
                dot += (double)qi[x] * (double)kj[x];
            }
            double score = dot * (double)scale;
            scores[j] = score;
            if (score > max_score) {
                max_score = score;
            }
        }

        double denom = 0.0;
        for (int j = 0; j < n; ++j) {
            scores[j] = exp(scores[j] - max_score);
            denom += scores[j];
        }

        for (int x = 0; x < d; ++x) {
            double acc = 0.0;
            for (int j = 0; j < n; ++j) {
                acc += (scores[j] / denom) * (double)v[(size_t)j * (size_t)d + (size_t)x];
            }
            out[(size_t)i * (size_t)d + (size_t)x] = (float)acc;
        }
    }
    free(scores);
}

static double check_error(const float *got, const float *expected, size_t count) {
    double max_abs = 0.0;
    for (size_t i = 0; i < count; ++i) {
        double err = fabs((double)got[i] - (double)expected[i]);
        if (err > max_abs) {
            max_abs = err;
        }
    }
    return max_abs;
}

static int compare_float(const void *left, const void *right) {
    float a = *(const float *)left;
    float b = *(const float *)right;
    return (a > b) - (a < b);
}

int main(void) {
    const int n = N;
    const int d = D;
    const float scale = 1.0f / sqrtf((float)d);
    const size_t count = (size_t)n * (size_t)d;

    float *q = (float *)aligned_alloc(64, (size_t)RUNS * count * sizeof(float));
    float *k = (float *)aligned_alloc(64, (size_t)RUNS * count * sizeof(float));
    float *v = (float *)aligned_alloc(64, (size_t)RUNS * count * sizeof(float));
    float *out = (float *)aligned_alloc(64, count * sizeof(float));
    float *expected = (float *)aligned_alloc(64, (size_t)RUNS * count * sizeof(float));
    if (!q || !k || !v || !out || !expected) {
        fprintf(stderr, "allocation failed\n");
        return 2;
    }

    for (int run = 0; run < RUNS; ++run) {
        make_case(
            q + (size_t)run * count,
            k + (size_t)run * count,
            v + (size_t)run * count,
            count,
            (uint32_t)CASE_SEED + (uint32_t)run * 1009u
        );
        reference_flash_attention(
            q + (size_t)run * count,
            k + (size_t)run * count,
            v + (size_t)run * count,
            expected + (size_t)run * count,
            n,
            d,
            scale
        );
    }

    float measurements[RUNS];
    double max_abs = 0.0;
    for (int run = 0; run < RUNS; ++run) {
        const float *case_q = q + (size_t)run * count;
        const float *case_k = k + (size_t)run * count;
        const float *case_v = v + (size_t)run * count;
        const float *case_expected = expected + (size_t)run * count;
        memset(out, 0, count * sizeof(float));
        uint64_t start = read_cycles();
        flash_attention_forward(case_q, case_k, case_v, out, n, d, scale);
        uint64_t end = read_cycles();
        measurements[run] = (float)(end - start);
        max_abs = check_error(out, case_expected, count);
        if (max_abs > 2.5e-4) {
            printf("{\"status\":\"failed\",\"run\":%d,\"max_abs\":%.9g,\"tolerance\":2.5e-4}\n", run, max_abs);
            return 1;
        }
    }
    qsort(measurements, RUNS, sizeof(float), compare_float);

    printf("{\"status\":\"passed\",\"n\":%d,\"d\":%d,\"cases\":%d,\"best_cycles\":%.0f,\"median_cycles\":%.0f,\"p90_cycles\":%.0f,\"max_abs\":%.9g,\"target_cycles\":350000}\n",
           n,
           d,
           RUNS,
           measurements[0],
           measurements[RUNS / 2],
           measurements[(RUNS * 9) / 10],
           max_abs);

    free(q);
    free(k);
    free(v);
    free(out);
    free(expected);
    return 0;
}
