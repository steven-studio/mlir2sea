#include <stdio.h>
#include <stdlib.h>
#include <math.h>

extern void vecadd_f64(double* p0, double* p1, double* p2);

int main() {
    int N = 8;
    double A[8], B[8], C_ref[8], C_rvv[8];

    srand(33);
    for (int i = 0; i < N; i++) A[i] = (double)(rand() % 100) / 10.0;
    for (int i = 0; i < N; i++) B[i] = (double)(rand() % 100) / 10.0;
    for (int i = 0; i < N; i++) C_ref[i] = A[i] + B[i];

    vecadd_f64(A, B, C_rvv);

    int mismatches = 0;
    for (int i = 0; i < N; i++) {
        double diff = fabs(C_ref[i] - C_rvv[i]);
        if (diff > 1e-9) {
            printf("MISMATCH at %d: ref=%f rvv=%f diff=%e\n", i, C_ref[i], C_rvv[i], diff);
            mismatches++;
        }
    }
    printf(mismatches == 0 ? "PASS: all %d elements match\n" : "FAIL: %d mismatches\n",
        mismatches == 0 ? N : mismatches);
    return mismatches != 0;
}
