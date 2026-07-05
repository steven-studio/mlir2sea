#include <stdio.h>
#include <math.h>

extern void vecreduce_add(float* p0, float* p1);

int main() {
    float in[4] = {1.5f, 2.5f, 3.0f, 4.0f}; // sum = 11.0
    float out[4];

    vecreduce_add(in, out);

    float expected_sum = 11.0f;
    int mismatches = 0;
    for (int i = 0; i < 4; i++) {
        float diff = fabsf(out[i] - expected_sum);
        if (diff > 1e-4f) {
            printf("MISMATCH at %d: expected=%f got=%f\n", i, expected_sum, out[i]);
            mismatches++;
        }
    }
    if (mismatches == 0) {
        printf("PASS: sum reduction correct (%.1f broadcast to all lanes)\n", expected_sum);
    } else {
        printf("FAIL: %d mismatches\n", mismatches);
    }
    return mismatches != 0;
}
