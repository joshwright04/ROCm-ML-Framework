#include <hip/hip_runtime.h>
#include <iostream>

__global__ void hello_kernel() {
    printf("Hello World from HIP! threadIdx.x = %d\n ", threadIdx.x);
}

int main() {
    // Launch 1 block with 1 thread
    dim3 block(1);
    dim3 threads(1);

    hipLaunchKernelGGL(hello_kernel, block, threads, 0, 0); // Launch the GPU kernel
    hipError_t err = hipDeviceSynchronize(); // Wait until the GPU kernel is done

    if(err != hipSuccess){
        std::cerr << "hipDeviceSynchronize failed: " << hipGetErrorString(err) << "\n";
        return 1;
    }

    std::cout << "Hello from CPU!\n";
    return 0;
}