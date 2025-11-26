#include <hip_runtime.h>
#include <iostream>
#include "../src/kernels/vector_add.hip"

int main() {
    const int n = 1024; // size of vectors
    size_t bytes = n * sizeof(float);

    float *a, *b, *c;
    hipError_t err;

    // Allocate unified memory accessible from CPU and GPU
    err = hipMallocManaged(&a, bytes); 
    if (err != hipSuccess) { std::cerr << "hipMallocManaged a failed\n"; return 1; }
    err = hipMallocManaged(&b, bytes);
    if (err != hipSuccess) { std::cerr << "hipMallocManaged b failed\n"; return 1; }
    err = hipMallocManaged(&c, bytes);
    if (err != hipSuccess) { std::cerr << "hipMallocManaged c failed\n"; return 1; }
    
    // memory is allocated already so a, b, c can be treated as regular arrays
    for (int i = 0; i < n; i++) {
        a[i] = 1.0f;
        b[i] = 2.0f;
    }

    dim3 block(256); //number of threads per block
    dim3 grid((n + block.x - 1) / block.x); //number of blocks in grid

    hipLaunchKernelGGL(add, grid, block, 0, 0, a, b, c, n);
    hipDeviceSynchronize();

    std::cout << "c[0] = " << c[0] << " (expected 3)\n";

    hipFree(a);
    hipFree(b);
    hipFree(c);

    return 0;
}