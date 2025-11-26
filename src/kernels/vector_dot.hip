#include <hip/hip_runtime.h>
#include <iostream>
#include <cmath>

#define HIP_CHECK(call)                                                          \
    do {                                                                         \
        hipError_t err = call;                                                   \
        if (err != hipSuccess) {                                                 \
            std::cerr << "HIP error at " << __FILE__ << ":" << __LINE__         \
                      << " code=" << static_cast<int>(err)                       \
                      << " \"" << hipGetErrorString(err) << "\"\n";              \
            std::exit(1);                                                        \
        }                                                                        \
    } while (0)

#define BLOCK_SIZE 256

__global__ void dot_kernel_shared(const float* a, const float* b, float* result, int n) {
    __shared__ float cache[BLOCK_SIZE];

    int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int localIdx  = threadIdx.x;

    float prod = 0.0f;
    if (globalIdx < n) {
        prod = a[globalIdx] * b[globalIdx];
    }

    cache[localIdx] = prod;
    __syncthreads();

    for (int stride = BLOCK_SIZE / 2; stride > 0; stride /= 2) {
        if (localIdx < stride) {
            cache[localIdx] += cache[localIdx + stride];
        }
        __syncthreads();
    }

    if (localIdx == 0) {
        atomicAdd(result, cache[0]);
    }
}

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

    dot_kernel_shared<<<blocks, threadsPerBlock>>>(a, b, result, n);
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
