#include <stdio.h>
#include <math.h>

extern void relu_cmpsel(float* p0, float* p1);

int main() {
    float in[8]  = {-3.0f, -0.5f, 0.0f, 2.5f, 4.0f, -7.0f, 1.5f, -0.01f};
    float out[8];

    relu_cmpsel(in, out);

    float ref[8];
    for (int i = 0; i < 8; i++) {
        ref[i] = (in[i] > 0.0f) ? in[i] : 0.0f;
    }

    int mismatches = 0;
    for (int i = 0; i < 8; i++) {
        float diff = fabsf(ref[i] - out[i]);
        if (diff > 1e-4f) {
            printf("MISMATCH at %d: in=%f ref=%f rvv=%f\n", i, in[i], ref[i], out[i]);
            mismatches++;
        }
    }
    if (mismatches == 0) {
        printf("PASS: all 8 elements match (cmpf+select ReLU verified)\n");
    } else {
        printf("FAIL: %d mismatches\n", mismatches);
    }
    return mismatches != 0;
}
