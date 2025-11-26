// src/kernels/vector_add.hip
#pragma once

#include <hip/hip_runtime.h>
#include <iostream>

// global means the fcn is called from the host (CPU) and runs on the device (GPU)
__global__ void add(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; // Compute the index of current thread
    if (i < n) c[i] = a[i] + b[i];
}
