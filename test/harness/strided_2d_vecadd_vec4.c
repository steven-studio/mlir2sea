#include <stdio.h>
#include <stdlib.h>
#include <math.h>

extern void strided_2d_vecadd(float* p0, float* p1, float* p2);

int main() {
    // p0 邏輯 2x4，row stride=16，col stride=2
    // 最大 offset = 1*16 + 3*2 = 22，backing 需要至少 23 個 float
    float backing[23];
    float b[8], c[8];

    srand(404);
    for (int i = 0; i < 23; i++) backing[i] = (float)(rand() % 100) / 10.0f;
    for (int i = 0; i < 8; i++) b[i] = (float)(rand() % 100) / 10.0f;

    strided_2d_vecadd(backing, b, c);

    float ref[8]; // 邏輯 2x4，row-major 排列存放參考結果
    for (int row = 0; row < 2; row++) {
        for (int col = 0; col < 4; col++) {
            int physOffset = row * 16 + col * 2;
            int logicalIdx = row * 4 + col;
            ref[logicalIdx] = backing[physOffset] + b[logicalIdx];
        }
    }

    int mismatches = 0;
    for (int i = 0; i < 8; i++) {
        float diff = fabsf(ref[i] - c[i]);
        if (diff > 1e-4f) {
            printf("MISMATCH at %d: ref=%f rvv=%f diff=%f\n", i, ref[i], c[i], diff);
            mismatches++;
        }
    }
    printf(mismatches == 0 ? "PASS: all 8 elements match (2D strided read verified)\n"
                            : "FAIL: %d mismatches\n", mismatches);
    return mismatches != 0;
}
