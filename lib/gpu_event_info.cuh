//
// Created by adadu on 2022/11/21.
//

#ifndef BRAINPYLIB_CHAOMING0625_GPU_NONZERO_CUH
#define BRAINPYLIB_CHAOMING0625_GPU_NONZERO_CUH

#include "kernel_helper_descriptor.cuh"

namespace brainpy_lib{

    void nonzero_bool(cudaStream_t stream, void **buffers, const char *opaque, std::size_t opaque_len);
    void nonzero_int(cudaStream_t stream, void **buffers, const char *opaque, std::size_t opaque_len);
    void nonzero_long(cudaStream_t stream, void **buffers, const char *opaque, std::size_t opaque_len);
    void nonzero_float(cudaStream_t stream, void **buffers, const char *opaque, std::size_t opaque_len);
    void nonzero_double(cudaStream_t stream, void **buffers, const char *opaque, std::size_t opaque_len);

}


#endif //BRAINPYLIB_CHAOMING0625_GPU_NONZERO_CUH
