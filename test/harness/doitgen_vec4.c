#include <stdio.h>
#include <stdlib.h>
#include <math.h>

extern void doitgen(float* p0, float* p1, float* p2);

int main() {
    int NR = 2, NQ = 2, NP = 4;
    float A[16], C4[16], tmp[16];
    float A_ref[16];

    srand(1414);
    for (int i = 0; i < 16; i++) A[i] = (float)(rand() % 100) / 10.0f;
    for (int i = 0; i < 16; i++) A_ref[i] = A[i];
    for (int i = 0; i < 16; i++) C4[i] = (float)(rand() % 100) / 10.0f;
    for (int i = 0; i < 16; i++) tmp[i] = 0.0f;

    // reference: for each (r,q), A[r][q][:] = A[r][q][:] @ C4
    float orig[4];
    for (int r = 0; r < NR; r++) {
        for (int q = 0; q < NQ; q++) {
            for (int s = 0; s < NP; s++) orig[s] = A_ref[(r*NQ+q)*NP + s];
            for (int p = 0; p < NP; p++) {
                float sum = 0.0f;
                for (int s = 0; s < NP; s++) sum += orig[s] * C4[s*NP+p];
                A_ref[(r*NQ+q)*NP + p] = sum;
            }
        }
    }

    doitgen(A, C4, tmp);

    int mismatches = 0;
    for (int i = 0; i < NR*NQ*NP; i++) {
        if (fabsf(A[i] - A_ref[i]) > 1e-1f) {
            printf("MISMATCH at %d: ref=%f rvv=%f\n", i, A_ref[i], A[i]);
            mismatches++;
        }
    }
    if (mismatches == 0) {
        printf("PASS: all %d elements match (doitgen verified)\n", NR*NQ*NP);
    } else {
        printf("FAIL: %d mismatches\n", mismatches);
    }
    return mismatches != 0;
}
