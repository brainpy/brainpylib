//
// Created by adadu on 2022/11/21.
//

#ifndef BRAINPYLIB_CHAOMING0625_GPU_NONZERO_CUH
#define BRAINPYLIB_CHAOMING0625_GPU_NONZERO_CUH

#include "kernel_helpers_descriptor.cuh"

namespace brainpy_lib{

    void nonzero_64(cudaStream_t stream, void **buffers, const char *opaque, std::size_t opaque_len);

}


#endif //BRAINPYLIB_CHAOMING0625_GPU_NONZERO_CUH
