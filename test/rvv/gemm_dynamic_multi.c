#include <stdio.h>
#include <stdlib.h>

void gemm(float* p0, float* p1, float* p2,
          size_t dim_p0_0, size_t dim_p0_1,
          size_t dim_p1_0, size_t dim_p1_1,
          size_t dim_p2_0, size_t dim_p2_1);

int test_shape(int M, int K, int N) {
    float* A = malloc(sizeof(float) * M * K);
    float* B = malloc(sizeof(float) * K * N);
    float* C = malloc(sizeof(float) * M * N);
    float* Cref = malloc(sizeof(float) * M * N);

    for (int i = 0; i < M; i++)
        for (int k = 0; k < K; k++)
            A[i*K + k] = (float)(i*3 - k*2);
    for (int k = 0; k < K; k++)
        for (int j = 0; j < N; j++)
            B[k*N + j] = (float)(k - j*2);
    for (int i = 0; i < M*N; i++) { C[i] = 0.0f; Cref[i] = 0.0f; }

    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++)
            for (int k = 0; k < K; k++)
                Cref[i*N + j] += A[i*K + k] * B[k*N + j];

    gemm(A, B, C, M, K, K, N, M, N);

    int ok = 1;
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++)
            if (C[i*N + j] != Cref[i*N + j]) {
                printf("  MISMATCH at [%d][%d]: got %f expected %f\n",
                       i, j, C[i*N + j], Cref[i*N + j]);
                ok = 0;
            }

    printf("[M=%d K=%d N=%d] %s\n", M, K, N, ok ? "PASS" : "FAIL");
    free(A); free(B); free(C); free(Cref);
    return ok;
}

int main() {
    int all_pass = 1;
    all_pass &= test_shape(1, 1, 1);
    all_pass &= test_shape(4, 4, 4);
    all_pass &= test_shape(8, 8, 8);
    all_pass &= test_shape(7, 7, 7);
    all_pass &= test_shape(5, 9, 3);
    all_pass &= test_shape(3, 7, 5);
    all_pass &= test_shape(2, 4, 6);
    printf(all_pass ? "\n=== ALL PASS ===\n" : "\n=== SOME FAILED ===\n");
    return all_pass ? 0 : 1;
}
