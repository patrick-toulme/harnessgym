#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

void moe_forward(
    const float *input,
    const float *w1,
    const float *b1,
    const float *w2,
    const float *b2,
    const int32_t *top1,
    const int32_t *top2,
    const float *gate1,
    const float *gate2,
    float *output,
    int tokens,
    int experts,
    int d_model,
    int d_hidden
);

#ifdef __cplusplus
}
#endif
