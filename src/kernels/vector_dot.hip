#include <hip/hip_runtime.h>
#include <iostream>
#include <cmath>
#include "../../include/hipml/common.hpp"

#define BLOCK_SIZE 256

__global__ void dot_kernel_shared(const float* a, const float* b, float* result, int n) {
    __shared__ float cache[BLOCK_SIZE];

    int globalIdx = blockIdx.x * blockDim.x + threadIdx.x; // current thread global index
    int localIdx  = threadIdx.x; //current thread idx within block

    float prod = 0.0f;
    if (globalIdx < n) {
        prod = a[globalIdx] * b[globalIdx]; //compute product if within bounds
    }

    cache[localIdx] = prod; //cache all the thread results
    __syncthreads(); //wait until all threads are done

    for (int stride = BLOCK_SIZE / 2; stride > 0; stride /= 2) { //log_2 reduction so we can do a single atomicAdd
        if (localIdx < stride) {
            cache[localIdx] += cache[localIdx + stride];
        }
        __syncthreads();
    }

    if (localIdx == 0) {
        atomicAdd(result, cache[0]);
    }
}


