//
// Created by adadu on 2022/12/1.
//

#ifndef BRAINPYLIB_GPU_MATVEC_JITCONN_CUH
#define BRAINPYLIB_GPU_MATVEC_JITCONN_CUH

#include "kernel_helper_descriptor.cuh"
#include "kernel_helpers_gpu.cuh"
#include "kernel_helpers_random.cuh"
#include "math.h"

namespace brainpy_lib {

    /* version 1 API */

    void matvec_jitconn_prob_homo_float(cudaStream_t stream, void **buffers,
                                        const char *opaque, std::size_t opaque_len);

    void matvec_jitconn_prob_homo_double(cudaStream_t stream, void **buffers,
                                         const char *opaque, std::size_t opaque_len);

    void matvec_jitconn_prob_uniform_float(cudaStream_t stream, void **buffers,
                                           const char *opaque, std::size_t opaque_len);

    void matvec_jitconn_prob_uniform_double(cudaStream_t stream, void **buffers,
                                            const char *opaque, std::size_t opaque_len);

    void matvec_jitconn_prob_normal_float(cudaStream_t stream, void **buffers,
                                          const char *opaque, std::size_t opaque_len);

    void matvec_jitconn_prob_normal_double(cudaStream_t stream, void **buffers,
                                           const char *opaque, std::size_t opaque_len);

    /* version 2 API */

    void matvec_jitconn_prob_homo_v2_float(cudaStream_t stream, void **buffers,
                                           const char *opaque, std::size_t opaque_len);

    void matvec_jitconn_prob_homo_v2_double(cudaStream_t stream, void **buffers,
                                            const char *opaque, std::size_t opaque_len);

    void matvec_jitconn_prob_uniform_v2_float(cudaStream_t stream, void **buffers,
                                              const char *opaque, std::size_t opaque_len);

    void matvec_jitconn_prob_uniform_v2_double(cudaStream_t stream, void **buffers,
                                               const char *opaque, std::size_t opaque_len);

    void matvec_jitconn_prob_normal_v2_float(cudaStream_t stream, void **buffers,
                                             const char *opaque, std::size_t opaque_len);

    void matvec_jitconn_prob_normal_v2_double(cudaStream_t stream, void **buffers,
                                              const char *opaque, std::size_t opaque_len);

}

#endif //BRAINPYLIB_GPU_MATVEC_JITCONN_CUH
