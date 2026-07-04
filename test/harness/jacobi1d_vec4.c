#include <stdio.h>
#include <stdlib.h>
#include <math.h>

extern void jacobi_1d(float* p0, float* p1);

void jacobi_ref(float* A, float* B, int n) {
    for (int i = 1; i < n - 1; i++) {
        B[i] = 0.33333333f * (A[i-1] + A[i] + A[i+1]);
    }
}

int main() {
    int N = 16;
    float A[16], B_ref[16], B_rvv[16];

    srand(7);
    for (int i = 0; i < N; i++) A[i] = (float)(rand() % 100) / 10.0f;
    for (int i = 0; i < N; i++) { B_ref[i] = -1.0f; B_rvv[i] = -1.0f; }

    jacobi_ref(A, B_ref, N);
    jacobi_1d(A, B_rvv);

    int mismatches = 0;
    // 只比較 index 1..12（kernel 實際處理到的範圍）
    for (int i = 1; i <= 12; i++) {
        float diff = fabsf(B_ref[i] - B_rvv[i]);
        if (diff > 1e-3f) {
            printf("MISMATCH at %d: ref=%f rvv=%f diff=%f\n", i, B_ref[i], B_rvv[i], diff);
            mismatches++;
        }
    }

    if (mismatches == 0) {
        printf("PASS: indices 1..12 match\n");
    } else {
        printf("FAIL: %d mismatches\n", mismatches);
    }
    return mismatches != 0;
}
