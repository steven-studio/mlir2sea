#include <stdio.h>
#include <stdlib.h>
#include <math.h>

extern void strided_vecadd(float* p0, float* p1, float* p2);

int main() {
    // p0 邏輯上 4 個元素，物理上間隔 2：backing storage 需要 7 個 float
    // (index 0,2,4,6 對應邏輯 index 0,1,2,3)
    float backing[7];
    float b[4], c[4];

    srand(303);
    for (int i = 0; i < 7; i++) backing[i] = (float)(rand() % 100) / 10.0f;
    for (int i = 0; i < 4; i++) b[i] = (float)(rand() % 100) / 10.0f;

    strided_vecadd(backing, b, c);

    float ref[4];
    for (int i = 0; i < 4; i++) {
        ref[i] = backing[i * 2] + b[i]; // 邏輯 index i 對應物理 index i*2
    }

    int mismatches = 0;
    for (int i = 0; i < 4; i++) {
        float diff = fabsf(ref[i] - c[i]);
        if (diff > 1e-4f) {
            printf("MISMATCH at %d: ref=%f rvv=%f diff=%f\n", i, ref[i], c[i], diff);
            mismatches++;
        }
    }
    printf(mismatches == 0 ? "PASS: all 4 elements match (strided read verified)\n"
                            : "FAIL: %d mismatches\n", mismatches);
    return mismatches != 0;
}
