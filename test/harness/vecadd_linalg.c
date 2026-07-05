#include <stdio.h>
#include <math.h>

extern void vecadd_linalg(float* p0, float* p1, float* p2);

int main() {
    float A[8], B[8], C[8];

    for (int i = 0; i < 8; i++) {
        A[i] = (float)i * 1.5f;
        B[i] = (float)i * 0.5f;
    }

    vecadd_linalg(A, B, C);

    int mismatches = 0;
    for (int i = 0; i < 8; i++) {
        float expected = A[i] + B[i];
        float diff = fabsf(C[i] - expected);
        if (diff > 1e-4f) {
            printf("MISMATCH at %d: expected=%f got=%f\n", i, expected, C[i]);
            mismatches++;
        }
    }
    if (mismatches == 0) {
        printf("PASS: all 8 elements match (linalg.generic pipeline verified)\n");
    } else {
        printf("FAIL: %d mismatches\n", mismatches);
    }
    return mismatches != 0;
}
