// 错误检查宏（Macros）
#pragma once
#include <cuda_runtime.h>
#include <optix.h>
#include <optix_stubs.h>
#include <iostream>
#include <sstream>
#include <stdexcept>

// 更加专业的错误检查，抛出异常以便上层捕获
#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t error = call;                                            \
        if (error != cudaSuccess) {                                          \
            std::stringstream ss;                                            \
            ss << "CUDA Call failed with error " << cudaGetErrorString(error) \
               << " at " << __FILE__ << ":" << __LINE__;                     \
            throw std::runtime_error(ss.str());                              \
        }                                                                    \
    } while (0)

#define OPTIX_CHECK(call)                                                    \
    do {                                                                     \
        OptixResult res = call;                                              \
        if (res != OPTIX_SUCCESS) {                                          \
            std::stringstream ss;                                            \
            ss << "OptiX Call failed with error code " << res                \
               << " at " << __FILE__ << ":" << __LINE__;                     \
            throw std::runtime_error(ss.str());                              \
        }                                                                    \
    } while (0)