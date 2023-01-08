//
// Created by adadu on 2023/1/8.
//

#ifndef BRAINPYLIB_GPU_JITCONN_MATMAT_CUH
#define BRAINPYLIB_GPU_JITCONN_MATMAT_CUH

#include "kernel_helper_descriptor.cuh"
#include "kernel_helpers_gpu.cuh"
#include "kernel_helpers_random.cuh"
#include "math.h"
#include <algorithm>
#include "kernel_helper_constant.cuh"

namespace brainpy_lib {

    /* version 2 API */

//    void matmat_jitconn_prob_homo_v2_float(cudaStream_t stream, void **buffers,
//                                           const char *opaque, std::size_t opaque_len);
//
//    void matmat_jitconn_prob_homo_v2_double(cudaStream_t stream, void **buffers,
//                                            const char *opaque, std::size_t opaque_len);
//
//    void matmat_jitconn_prob_uniform_v2_float(cudaStream_t stream, void **buffers,
//                                              const char *opaque, std::size_t opaque_len);
//
//    void matmat_jitconn_prob_uniform_v2_double(cudaStream_t stream, void **buffers,
//                                               const char *opaque, std::size_t opaque_len);

    void jitconn_matmat_prob_normal_float_v3(cudaStream_t stream, void **buffers,
                                             const char *opaque, std::size_t opaque_len);

    void jitconn_matmat_prob_normal_double_v3(cudaStream_t stream, void **buffers,
                                              const char *opaque, std::size_t opaque_len);

    void jitconn_matmat_prob_normal_float_v2(cudaStream_t stream, void **buffers,
                                             const char *opaque, std::size_t opaque_len);

    void jitconn_matmat_prob_normal_double_v2(cudaStream_t stream, void **buffers,
                                              const char *opaque, std::size_t opaque_len);

    void jitconn_matmat_prob_normal_float_v1(cudaStream_t stream, void **buffers,
                                             const char *opaque, std::size_t opaque_len);

    void jitconn_matmat_prob_normal_double_v1(cudaStream_t stream, void **buffers,
                                              const char *opaque, std::size_t opaque_len);

    void jitconn_matmat_prob_uniform_float_v1(cudaStream_t stream, void **buffers,
                                              const char *opaque, std::size_t opaque_len);

    void jitconn_matmat_prob_uniform_double_v1(cudaStream_t stream, void **buffers,
                                               const char *opaque, std::size_t opaque_len);

}

#endif //BRAINPYLIB_GPU_JITCONN_MATMAT_CUH
