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

    /* version 1 API */
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

    /* version 2 API */
    void event_matvec_jitconn_prob_homo_v2_float_bool(cudaStream_t stream, void **buffers,
                                                      const char *opaque, std::size_t opaque_len);

    void event_matvec_jitconn_prob_homo_v2_float_float(cudaStream_t stream, void **buffers,
                                                       const char *opaque, std::size_t opaque_len);

    void event_matvec_jitconn_prob_homo_v2_double_bool(cudaStream_t stream, void **buffers,
                                                       const char *opaque, std::size_t opaque_len);

    void event_matvec_jitconn_prob_homo_v2_double_double(cudaStream_t stream, void **buffers,
                                                         const char *opaque, std::size_t opaque_len);

    void event_matvec_jitconn_prob_uniform_v2_float_bool(cudaStream_t stream, void **buffers,
                                                         const char *opaque, std::size_t opaque_len);

    void event_matvec_jitconn_prob_uniform_v2_float_float(cudaStream_t stream, void **buffers,
                                                          const char *opaque, std::size_t opaque_len);

    void event_matvec_jitconn_prob_uniform_v2_double_bool(cudaStream_t stream, void **buffers,
                                                          const char *opaque, std::size_t opaque_len);

    void event_matvec_jitconn_prob_uniform_v2_double_double(cudaStream_t stream, void **buffers,
                                                            const char *opaque, std::size_t opaque_len);

    void event_matvec_jitconn_prob_normal_v2_float_bool(cudaStream_t stream, void **buffers,
                                                        const char *opaque, std::size_t opaque_len);

    void event_matvec_jitconn_prob_normal_v2_float_float(cudaStream_t stream, void **buffers,
                                                         const char *opaque, std::size_t opaque_len);

    void event_matvec_jitconn_prob_normal_v2_double_bool(cudaStream_t stream, void **buffers,
                                                         const char *opaque, std::size_t opaque_len);

    void event_matvec_jitconn_prob_normal_v2_double_double(cudaStream_t stream, void **buffers,
                                                           const char *opaque, std::size_t opaque_len);


}

#endif //BRAINPYLIB_GPU_EVENT_MV_RANDOM_CUH
