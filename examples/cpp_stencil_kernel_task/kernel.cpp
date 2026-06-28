#include "kernel.h"

#if defined(_MSC_VER)
#define HF_NOINLINE __declspec(noinline)
#else
#define HF_NOINLINE __attribute__((noinline))
#endif

static HF_NOINLINE float clamped_load(const float* src, int rows, int cols, int r, int c) {
    if (r < 0) {
        r = 0;
    } else if (r >= rows) {
        r = rows - 1;
    }
    if (c < 0) {
        c = 0;
    } else if (c >= cols) {
        c = cols - 1;
    }
    return src[r * cols + c];
}

extern "C" void stencil_step(const float* src, float* dst, int rows, int cols, float alpha) {
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            const float center = clamped_load(src, rows, cols, r, c);
            const float north = clamped_load(src, rows, cols, r - 1, c);
            const float south = clamped_load(src, rows, cols, r + 1, c);
            const float west = clamped_load(src, rows, cols, r, c - 1);
            const float east = clamped_load(src, rows, cols, r, c + 1);
            const float filtered = 0.50f * center + 0.125f * (north + south + west + east);
            dst[r * cols + c] = alpha * filtered + (1.0f - alpha) * center;
        }
    }
}
