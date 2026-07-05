#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

extern void seidel2d(float* p0);

int main() {
    int N = 8;
    float A_seq[64];   // 真正的循序 Gauss-Seidel（正確語意）
    float A_rvv[64];   // 透過向量化 kernel 算出的結果

    srand(1717);
    for (int i = 0; i < 64; i++) {
        float v = (float)(rand() % 100) / 10.0f;
        A_seq[i] = v;
        A_rvv[i] = v;
    }

    // 正確的循序 Gauss-Seidel：逐點更新，立即使用剛更新過的值
    for (int i = 1; i < 7; i++) {
        for (int j = 1; j < 5; j++) {
            A_seq[i*N+j] = (A_seq[i*N+j] + A_seq[i*N+j-1] + A_seq[i*N+j+1]
                          + A_seq[(i-1)*N+j] + A_seq[(i+1)*N+j]) * (1.0f/9.0f);
        }
    }

    seidel2d(A_rvv);

    int mismatches = 0;
    for (int i = 1; i < 7; i++) {
        for (int j = 1; j < 5; j++) {
            int idx = i*N+j;
            float diff = fabsf(A_seq[idx] - A_rvv[idx]);
            if (diff > 1e-3f) {
                printf("DIVERGE at (%d,%d): sequential=%f vectorized=%f diff=%f\n",
                    i, j, A_seq[idx], A_rvv[idx], diff);
                mismatches++;
            }
        }
    }
    if (mismatches == 0) {
        printf("UNEXPECTED PASS: vectorized result matched sequential Gauss-Seidel\n");
    } else {
        printf("EXPECTED DIVERGENCE: %d/%d points differ from sequential Gauss-Seidel "
               "(vectorization silently changed the algorithm's semantics)\n",
               mismatches, 4*6);
    }
    return 0; // 這裡故意回傳 0，因為這次的重點是觀察現象，不是判斷 pass/fail
}
