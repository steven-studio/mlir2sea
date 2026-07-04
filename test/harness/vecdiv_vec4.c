#include <stdio.h>
#include <stdlib.h>
#include <math.h>

extern void vecdiv(float* p0, float* p1, float* p2);

int main() {
    int N = 8;
    float A[8], B[8], C_ref[8], C_rvv[8];

    srand(22);
    for (int i = 0; i < N; i++) A[i] = (float)(rand() % 100) / 10.0f;
    // 避免除以 0，B 從 1.0~10.0 取值
    for (int i = 0; i < N; i++) B[i] = (float)(rand() % 100 + 10) / 10.0f;
    for (int i = 0; i < N; i++) C_ref[i] = A[i] / B[i];

    vecdiv(A, B, C_rvv);

    int mismatches = 0;
    for (int i = 0; i < N; i++) {
        float diff = fabsf(C_ref[i] - C_rvv[i]);
        if (diff > 1e-4f) {
            printf("MISMATCH at %d: ref=%f rvv=%f diff=%f\n", i, C_ref[i], C_rvv[i], diff);
            mismatches++;
        }
    }
    printf(mismatches == 0 ? "PASS: all %d elements match\n" : "FAIL: %d mismatches\n",
        mismatches == 0 ? N : mismatches);
    return mismatches != 0;
}
