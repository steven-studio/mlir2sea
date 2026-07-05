#include <stdio.h>
#include <stdlib.h>
#include <math.h>

extern void heat3d(float* p0, float* p1);

int main() {
    int D0 = 4, D1 = 4, D2 = 8;
    int N = D0*D1*D2;
    float A[128], B[128], B_ref[128];

    srand(1616);
    for (int i = 0; i < N; i++) A[i] = (float)(rand() % 100) / 10.0f;
    for (int i = 0; i < N; i++) { B[i] = -1.0f; B_ref[i] = -1.0f; }

    for (int i = 1; i < 3; i++) {
        for (int j = 1; j < 3; j++) {
            for (int k = 1; k < 5; k++) {
                float center = A[(i*D1+j)*D2+k];
                float xm = A[((i-1)*D1+j)*D2+k];
                float xp = A[((i+1)*D1+j)*D2+k];
                float ym = A[(i*D1+(j-1))*D2+k];
                float yp = A[(i*D1+(j+1))*D2+k];
                float zm = A[(i*D1+j)*D2+(k-1)];
                float zp = A[(i*D1+j)*D2+(k+1)];
                B_ref[(i*D1+j)*D2+k] = 0.1f * (center+xm+xp+ym+yp+zm+zp);
            }
        }
    }

    heat3d(A, B);

    int mismatches = 0;
    for (int i = 1; i < 3; i++) {
        for (int j = 1; j < 3; j++) {
            for (int k = 1; k < 5; k++) {
                int idx = (i*D1+j)*D2+k;
                float diff = fabsf(B[idx] - B_ref[idx]);
                if (diff > 1e-3f) {
                    printf("MISMATCH at (%d,%d,%d): ref=%f rvv=%f\n", i, j, k, B_ref[idx], B[idx]);
                    mismatches++;
                }
            }
        }
    }
    if (mismatches == 0) {
        printf("PASS: all interior points match (heat3d verified)\n");
    } else {
        printf("FAIL: %d mismatches\n", mismatches);
    }
    return mismatches != 0;
}
