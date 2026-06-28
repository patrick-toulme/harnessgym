#ifndef FLASH_ATTENTION_KERNEL_H
#define FLASH_ATTENTION_KERNEL_H

void flash_attention_forward(
    const float *q,
    const float *k,
    const float *v,
    float *out,
    int n,
    int d,
    float scale
);

#endif
