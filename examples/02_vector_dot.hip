#include <hip/hip_runtime.h>
#include <iostream>
#include "../src/kernels/vector_dot.hip"
#include "../include/hipml/common.hpp"

#define BLOCK_SIZE 256

int main() {
    const int n = 1 << 20; // 1,048,576 elements
    size_t bytes = n * sizeof(float);

    float *a, *b, *result;

    HIP_CHECK(hipMallocManaged(&a, bytes));
    HIP_CHECK(hipMallocManaged(&b, bytes));
    HIP_CHECK(hipMallocManaged(&result, sizeof(float)));

    for (int i = 0; i < n; ++i) {
        a[i] = 1.0f;
        b[i] = 2.0f;
    }
    *result = 0.0f;

    int threadsPerBlock = BLOCK_SIZE;
    int blocks = (n + threadsPerBlock - 1) / threadsPerBlock;

    hipLaunchKernelGGL(
        dot_kernel_shared,      // kernel name
        dim3(blocks),           // grid size
        dim3(threadsPerBlock),  // block size
        0,                      // shared memory bytes (0 = default)
        0,                      // stream (0 = default stream)
        a, b, result, n         // kernel arguments
    );
    HIP_CHECK(hipDeviceSynchronize());

    double cpu_dot = 0.0;
    for (int i = 0; i < n; ++i) {
        cpu_dot += static_cast<double>(a[i]) * static_cast<double>(b[i]);
    }

    std::cout << "GPU dot = " << *result << "\n";
    std::cout << "CPU dot = " << cpu_dot << "\n";
    std::cout << "Difference = " << std::fabs(cpu_dot - *result) << "\n";

    HIP_CHECK(hipFree(a));
    HIP_CHECK(hipFree(b));
    HIP_CHECK(hipFree(result));

    return 0;
}