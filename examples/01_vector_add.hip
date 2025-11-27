#include <hip/hip_runtime.h>
#include <iostream>
#include "../src/kernels/vector_add.hip"
#include "../include/hipml/common.hpp"

int main() {
    const int n = 1024; // size of vectors
    size_t bytes = n * sizeof(float);

    float *a, *b, *c;

    // Allocate unified memory accessible from CPU and GPU
    HIP_CHECK(hipMallocManaged(&a, bytes));
    HIP_CHECK(hipMallocManaged(&b, bytes));
    HIP_CHECK(hipMallocManaged(&c, bytes));
    
    for (int i = 0; i < n; i++) {
        a[i] = 1.0f;
        b[i] = 2.0f;
    }

    dim3 block(256); // number of threads per block
    dim3 grid((n + block.x - 1) / block.x); // number of blocks in grid

    hipLaunchKernelGGL(add, grid, block, 0, 0, a, b, c, n);

    // check launch + sync
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());

    std::cout << "c[0] = " << c[0] << " (expected 3)\n";

    HIP_CHECK(hipFree(a));
    HIP_CHECK(hipFree(b));
    HIP_CHECK(hipFree(c));

    return 0;
}
