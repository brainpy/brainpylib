// This header is not specific to our application and you'll probably want something like this
// for any extension you're building. This includes the infrastructure needed to serialize
// descriptors that are used with the "opaque" parameter of the GPU custom call. In our example
// we'll use this parameter to pass the size of our problem.

#ifndef _BRAINPYLIB_CUDA_OP_MATMUL_HELPERS_H_
#define _BRAINPYLIB_CUDA_OP_MATMUL_HELPERS_H_


#include <cstddef>
#include <cstdint>
#include "cuda_runtime_api.h"
#include "pybind11_kernel_helpers.h"
#include "kernel_helpers_gpu.h"
#include "curand_kernel.h"


namespace brainpy_lib {
    // Descriptors
    struct MatMulDescriptor {
        std::uint32_t m;
        std::uint32_t k;
        std::uint32_t n;
        std::uint32_t seed;
        float p;
    };
    pybind11::bytes build_matmul_descriptor(std::uint32_t m, std::uint32_t k, std::uint32_t n, std::uint32_t seed, float p);


    struct MMMDescriptor {
        std::uint32_t m;
        std::uint32_t k;
        std::uint32_t n;
        float p;
    };
    pybind11::bytes build_mmm_descriptor(std::uint32_t m, std::uint32_t k, std::uint32_t n, float p);



    struct NonZeroDescriptor {
        std::uint32_t event_size;
        std::uint32_t batch_size;
    };
    pybind11::bytes build_nonzero_descriptor(std::uint32_t event_size, std::uint32_t batch_size);


}  // namespace brainpy_lib

#endif
