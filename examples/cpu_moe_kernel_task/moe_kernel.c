#include "moe_kernel.h"

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#ifndef HF_ROUTE_MODE
#define HF_ROUTE_MODE 0
#endif

#ifndef HF_HIDDEN_TILE
#define HF_HIDDEN_TILE 16
#endif

#ifndef HF_OUTPUT_TILE
#define HF_OUTPUT_TILE 8
#endif

#ifndef HF_INPUT_UNROLL
#define HF_INPUT_UNROLL 1
#endif

#ifndef HF_HIDDEN_UNROLL
#define HF_HIDDEN_UNROLL 1
#endif

#ifndef HF_PREFETCH
#define HF_PREFETCH 0
#endif

#ifndef HF_ZERO_MODE
#define HF_ZERO_MODE 0
#endif

#define HF_MAX_TOKENS 512
#define HF_MAX_EXPERTS 32
#define HF_MAX_HIDDEN 256

static inline float relu(float value) {
    return value > 0.0f ? value : 0.0f;
}

static void zero_output(float *output, int tokens, int d_model) {
    const int total = tokens * d_model;
#if HF_ZERO_MODE == 1
    for (int i = 0; i < total; ++i) {
        output[i] = 0.0f;
    }
#else
    memset(output, 0, (size_t)total * sizeof(float));
#endif
}

static void compute_hidden(
    const float *x,
    const float *expert_w1,
    const float *expert_b1,
    float *hidden,
    int d_model,
    int d_hidden
) {
    int h = 0;
    for (; h + HF_HIDDEN_TILE <= d_hidden; h += HF_HIDDEN_TILE) {
        for (int hh = h; hh < h + HF_HIDDEN_TILE; ++hh) {
            const float *w = expert_w1 + (size_t)hh * d_model;
            float sum = expert_b1[hh];
            int d = 0;
#if HF_INPUT_UNROLL >= 4
            for (; d + 3 < d_model; d += 4) {
                sum += x[d] * w[d];
                sum += x[d + 1] * w[d + 1];
                sum += x[d + 2] * w[d + 2];
                sum += x[d + 3] * w[d + 3];
            }
#endif
            for (; d < d_model; ++d) {
                sum += x[d] * w[d];
            }
            hidden[hh] = relu(sum);
        }
    }
    for (; h < d_hidden; ++h) {
        const float *w = expert_w1 + (size_t)h * d_model;
        float sum = expert_b1[h];
        for (int d = 0; d < d_model; ++d) {
            sum += x[d] * w[d];
        }
        hidden[h] = relu(sum);
    }
}

static void accumulate_output(
    const float *hidden,
    const float *expert_w2,
    const float *expert_b2,
    float gate,
    float *y,
    int d_model,
    int d_hidden
) {
    int d = 0;
    for (; d + HF_OUTPUT_TILE <= d_model; d += HF_OUTPUT_TILE) {
        for (int dd = d; dd < d + HF_OUTPUT_TILE; ++dd) {
            const float *w = expert_w2 + (size_t)dd * d_hidden;
            float sum = expert_b2[dd];
            int h = 0;
#if HF_HIDDEN_UNROLL >= 4
            for (; h + 3 < d_hidden; h += 4) {
                sum += hidden[h] * w[h];
                sum += hidden[h + 1] * w[h + 1];
                sum += hidden[h + 2] * w[h + 2];
                sum += hidden[h + 3] * w[h + 3];
            }
#endif
            for (; h < d_hidden; ++h) {
                sum += hidden[h] * w[h];
            }
            y[dd] += gate * sum;
        }
    }
    for (; d < d_model; ++d) {
        const float *w = expert_w2 + (size_t)d * d_hidden;
        float sum = expert_b2[d];
        for (int h = 0; h < d_hidden; ++h) {
            sum += hidden[h] * w[h];
        }
        y[d] += gate * sum;
    }
}

static void run_expert_for_token(
    const float *input,
    const float *w1,
    const float *b1,
    const float *w2,
    const float *b2,
    float *output,
    int token,
    int expert,
    float gate,
    int experts,
    int d_model,
    int d_hidden
) {
    (void)experts;
    float hidden[HF_MAX_HIDDEN];
    if (d_hidden > HF_MAX_HIDDEN) {
        return;
    }
    const float *x = input + (size_t)token * d_model;
    const float *expert_w1 = w1 + (size_t)expert * d_hidden * d_model;
    const float *expert_b1 = b1 + (size_t)expert * d_hidden;
    const float *expert_w2 = w2 + (size_t)expert * d_model * d_hidden;
    const float *expert_b2 = b2 + (size_t)expert * d_model;
    float *y = output + (size_t)token * d_model;
    compute_hidden(x, expert_w1, expert_b1, hidden, d_model, d_hidden);
    accumulate_output(hidden, expert_w2, expert_b2, gate, y, d_model, d_hidden);
}

static void forward_token_major(
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
) {
    for (int t = 0; t < tokens; ++t) {
#if HF_PREFETCH > 0
        const int pf = t + HF_PREFETCH;
        if (pf < tokens) {
            __builtin_prefetch(input + (size_t)pf * d_model, 0, 1);
        }
#endif
        run_expert_for_token(input, w1, b1, w2, b2, output, t, top1[t], gate1[t], experts, d_model, d_hidden);
        run_expert_for_token(input, w1, b1, w2, b2, output, t, top2[t], gate2[t], experts, d_model, d_hidden);
    }
}

static void forward_expert_scan(
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
) {
    for (int expert = 0; expert < experts; ++expert) {
        for (int t = 0; t < tokens; ++t) {
            if (top1[t] == expert) {
                run_expert_for_token(input, w1, b1, w2, b2, output, t, expert, gate1[t], experts, d_model, d_hidden);
            }
            if (top2[t] == expert) {
                run_expert_for_token(input, w1, b1, w2, b2, output, t, expert, gate2[t], experts, d_model, d_hidden);
            }
        }
    }
}

static void forward_bucketed(
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
) {
    int counts[HF_MAX_EXPERTS];
    int offsets[HF_MAX_EXPERTS + 1];
    int cursor[HF_MAX_EXPERTS];
    int token_ids[HF_MAX_TOKENS * 2];
    float token_gates[HF_MAX_TOKENS * 2];
    if (tokens > HF_MAX_TOKENS || experts > HF_MAX_EXPERTS) {
        forward_expert_scan(input, w1, b1, w2, b2, top1, top2, gate1, gate2, output, tokens, experts, d_model, d_hidden);
        return;
    }

    for (int e = 0; e < experts; ++e) {
        counts[e] = 0;
    }
    for (int t = 0; t < tokens; ++t) {
        counts[top1[t]] += 1;
        counts[top2[t]] += 1;
    }
    offsets[0] = 0;
    for (int e = 0; e < experts; ++e) {
        offsets[e + 1] = offsets[e] + counts[e];
        cursor[e] = offsets[e];
    }
    for (int t = 0; t < tokens; ++t) {
        int p = cursor[top1[t]]++;
        token_ids[p] = t;
        token_gates[p] = gate1[t];
        p = cursor[top2[t]]++;
        token_ids[p] = t;
        token_gates[p] = gate2[t];
    }
    for (int e = 0; e < experts; ++e) {
        for (int i = offsets[e]; i < offsets[e + 1]; ++i) {
            run_expert_for_token(input, w1, b1, w2, b2, output, token_ids[i], e, token_gates[i], experts, d_model, d_hidden);
        }
    }
}

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
) {
    if (tokens <= 0 || experts <= 0 || d_model <= 0 || d_hidden <= 0) {
        return;
    }
    zero_output(output, tokens, d_model);
#if HF_ROUTE_MODE == 2
    forward_bucketed(input, w1, b1, w2, b2, top1, top2, gate1, gate2, output, tokens, experts, d_model, d_hidden);
#elif HF_ROUTE_MODE == 1
    forward_expert_scan(input, w1, b1, w2, b2, top1, top2, gate1, gate2, output, tokens, experts, d_model, d_hidden);
#else
    forward_token_major(input, w1, b1, w2, b2, top1, top2, gate1, gate2, output, tokens, experts, d_model, d_hidden);
#endif
}
