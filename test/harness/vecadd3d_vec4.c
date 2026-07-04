#include <stdio.h>
#include <stdlib.h>
#include <math.h>

extern void vecadd3d(float* p0, float* p1, float* p2);

int main() {
    int D0 = 2, D1 = 3, D2 = 8;
    int N = D0 * D1 * D2;
    float A[48], B[48], C_ref[48], C_rvv[48];

    srand(55);
    for (int i = 0; i < N; i++) A[i] = (float)(rand() % 100) / 10.0f;
    for (int i = 0; i < N; i++) B[i] = (float)(rand() % 100) / 10.0f;
    for (int i = 0; i < N; i++) C_ref[i] = A[i] + B[i];

    vecadd3d(A, B, C_rvv);

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
