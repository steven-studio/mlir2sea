#include <stdio.h>
#include <stdlib.h>

void matvec(float* p0, float* p1, float* p2,
            size_t dim_p0_0, size_t dim_p0_1,
            size_t dim_p1_0, size_t dim_p2_0);

int test_shape(int M, int K) {
    float* A = malloc(sizeof(float) * M * K);
    float* x = malloc(sizeof(float) * K);
    float* y = malloc(sizeof(float) * M);
    float* yref = malloc(sizeof(float) * M);

    for (int i = 0; i < M; i++)
        for (int k = 0; k < K; k++)
            A[i*K + k] = (float)(i*3 - k*2);
    for (int k = 0; k < K; k++)
        x[k] = (float)(k - 1);
    for (int i = 0; i < M; i++) { y[i] = 0.0f; yref[i] = 0.0f; }

    for (int i = 0; i < M; i++)
        for (int k = 0; k < K; k++)
            yref[i] += A[i*K + k] * x[k];

    matvec(A, x, y, M, K, K, M);

    int ok = 1;
    for (int i = 0; i < M; i++)
        if (y[i] != yref[i]) {
            printf("  MISMATCH at [%d]: got %f expected %f\n", i, y[i], yref[i]);
            ok = 0;
        }

    printf("[M=%d K=%d] %s\n", M, K, ok ? "PASS" : "FAIL");
    free(A); free(x); free(y); free(yref);
    return ok;
}

int main() {
    int all_pass = 1;
    all_pass &= test_shape(1, 1);
    all_pass &= test_shape(4, 4);
    all_pass &= test_shape(8, 8);
    all_pass &= test_shape(7, 7);
    all_pass &= test_shape(5, 9);
    all_pass &= test_shape(3, 13);
    all_pass &= test_shape(1, 7);
    all_pass &= test_shape(10, 1);
    printf(all_pass ? "\n=== ALL PASS ===\n" : "\n=== SOME FAILED ===\n");
    return all_pass ? 0 : 1;
}
