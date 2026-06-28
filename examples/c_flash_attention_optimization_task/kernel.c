#include "kernel.h"

#include <math.h>
#include <stdlib.h>

void flash_attention_forward(
    const float *q,
    const float *k,
    const float *v,
    float *out,
    int n,
    int d,
    float scale
) {
    float *scores = (float *)malloc((size_t)n * sizeof(float));
    if (scores == 0) {
        return;
    }

    for (int i = 0; i < n; ++i) {
        const float *qi = q + (size_t)i * (size_t)d;
        float max_score = -INFINITY;

        for (int j = 0; j < n; ++j) {
            const float *kj = k + (size_t)j * (size_t)d;
            float dot = 0.0f;
            for (int x = 0; x < d; ++x) {
                dot += qi[x] * kj[x];
            }
            float score = dot * scale;
            scores[j] = score;
            if (score > max_score) {
                max_score = score;
            }
        }

        float denom = 0.0f;
        for (int j = 0; j < n; ++j) {
            float weight = expf(scores[j] - max_score);
            scores[j] = weight;
            denom += weight;
        }

        for (int x = 0; x < d; ++x) {
            out[(size_t)i * (size_t)d + (size_t)x] = 0.0f;
        }

        float inv_denom = 1.0f / denom;
        for (int j = 0; j < n; ++j) {
            const float *vj = v + (size_t)j * (size_t)d;
            float weight = scores[j] * inv_denom;
            for (int x = 0; x < d; ++x) {
                out[(size_t)i * (size_t)d + (size_t)x] += weight * vj[x];
            }
        }
    }

    free(scores);
}
