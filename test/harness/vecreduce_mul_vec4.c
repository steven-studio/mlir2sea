#include <stdio.h>
#include <math.h>

extern void vecreduce_mul(float* p0, float* p1);

int main() {
    float in[4] = {1.5f, 2.0f, 3.0f, 2.0f}; // product = 1.5*2.0*3.0*2.0 = 18.0
    float out[4];

    vecreduce_mul(in, out);

    float expected = 18.0f;
    int mismatches = 0;
    for (int i = 0; i < 4; i++) {
        float diff = fabsf(out[i] - expected);
        if (diff > 1e-3f) {
            printf("MISMATCH at %d: expected=%f got=%f\n", i, expected, out[i]);
            mismatches++;
        }
    }
    if (mismatches == 0) {
        printf("PASS: product reduction correct (%.1f broadcast to all lanes)\n", expected);
    } else {
        printf("FAIL: %d mismatches\n", mismatches);
    }
    return mismatches != 0;
}
