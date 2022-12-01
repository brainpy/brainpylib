//
// Created by adadu on 2022/11/30.
//

#ifndef BRAINPYLIB_GPU_EVENT_MV_RANDOM_CUH
#define BRAINPYLIB_GPU_EVENT_MV_RANDOM_CUH

#include "kernel_helper_descriptor.cuh"
#include "kernel_helpers_gpu.cuh"
#include "kernel_helpers_random.cuh"
#include "math.h"

namespace brainpy_lib {

    void event_matvec_jitconn_prob_homo_float(cudaStream_t stream, void **buffers,
                                              const char *opaque, std::size_t opaque_len);

    void event_matvec_jitconn_prob_homo_double(cudaStream_t stream, void **buffers,
                                               const char *opaque, std::size_t opaque_len);

    void event_matvec_jitconn_prob_uniform_float(cudaStream_t stream, void **buffers,
                                                 const char *opaque, std::size_t opaque_len);

    void event_matvec_jitconn_prob_uniform_double(cudaStream_t stream, void **buffers,
                                                  const char *opaque, std::size_t opaque_len);

    void event_matvec_jitconn_prob_normal_float(cudaStream_t stream, void **buffers,
                                                const char *opaque, std::size_t opaque_len);

    void event_matvec_jitconn_prob_normal_double(cudaStream_t stream, void **buffers,
                                                 const char *opaque, std::size_t opaque_len);

}

#endif //BRAINPYLIB_GPU_EVENT_MV_RANDOM_CUH
