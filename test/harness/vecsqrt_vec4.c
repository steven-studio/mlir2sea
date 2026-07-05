#include <stdio.h>
#include <math.h>

extern void vecsqrt(float* p0, float* p1);

int main() {
    float A[8] = {4.0f, 9.0f, 16.0f, 25.0f, 2.0f, 0.25f, 100.0f, 1.0f};
    float B[8];

    vecsqrt(A, B);

    int mismatches = 0;
    for (int i = 0; i < 8; i++) {
        float expected = sqrtf(A[i]);
        float diff = fabsf(B[i] - expected);
        if (diff > 1e-4f) {
            printf("MISMATCH at %d: in=%f expected=%f got=%f\n", i, A[i], expected, B[i]);
            mismatches++;
        }
    }
    if (mismatches == 0) {
        printf("PASS: all 8 elements match (vecsqrt verified)\n");
    } else {
        printf("FAIL: %d mismatches\n", mismatches);
    }
    return mismatches != 0;
}
