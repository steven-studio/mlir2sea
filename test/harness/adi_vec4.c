#include <stdio.h>
#include <stdlib.h>
#include <math.h>

extern void adi_forward(float* X, float* a, float* c);

int main() {
    int N = 8;
    float X_seq[8], X_rvv[8], a[8], c[8];

    srand(1919);
    for (int i = 0; i < N; i++) {
        float v = (float)(rand() % 100) / 10.0f + 1.0f; // 避免 X 初值是 0
        X_seq[i] = v;
        X_rvv[i] = v;
        a[i] = (float)(rand() % 50) / 100.0f;  // 係數故意小一點，避免除法爆炸
        c[i] = (float)(rand() % 50) / 100.0f + 1.0f; // 避免除以接近 0 的值
    }

    // 正確的循序前向消去：X[i] = (X[i] - a[i]*X[i-1]) / c[i]
    for (int i = 1; i <= 4; i++) {
        X_seq[i] = (X_seq[i] - a[i]*X_seq[i-1]) / c[i];
    }

    adi_forward(X_rvv, a, c);

    int mismatches = 0;
    for (int i = 1; i <= 4; i++) {
        float diff = fabsf(X_seq[i] - X_rvv[i]);
        if (diff > 1e-3f) {
            printf("DIVERGE at %d: sequential=%f vectorized=%f diff=%f\n",
                i, X_seq[i], X_rvv[i], diff);
            mismatches++;
        }
    }
    if (mismatches == 0) {
        printf("UNEXPECTED PASS: vectorized matched sequential forward substitution\n");
    } else {
        printf("EXPECTED DIVERGENCE: %d/4 points differ from sequential forward "
               "substitution (vectorizing a recurrence silently changed the result)\n",
               mismatches);
    }
    return 0;
}
