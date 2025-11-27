#pragma once

#include <hip/hip_runtime.h>
#include <iostream>

#define HIP_CHECK(call)                                                          \
    do {                                                                         \
        hipError_t err__ = (call);                                               \
        if (err__ != hipSuccess) {                                               \
            std::cerr << "HIP error at " << __FILE__ << ":" << __LINE__         \
                      << " code=" << static_cast<int>(err__)                     \
                      << " \"" << hipGetErrorString(err__) << "\"\n";            \
            std::exit(1);                                                        \
        }                                                                        \
    } while (0)