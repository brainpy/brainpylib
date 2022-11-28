//
// Created by Chaoming Wang on 2022/11/28.
//

#ifndef BRAINPYLIB_GPU_EVENT_CSR_MATVEC_CUH
#define BRAINPYLIB_GPU_EVENT_CSR_MATVEC_CUH


#include "kernel_helper_descriptor.cuh"
#include "kernel_helper_constant.cuh"

namespace brainpy_lib {

    void event_csr_matvec_heter_float_bool(cudaStream_t stream, void **buffers,
                                             const char *opaque, std::size_t opaque_len);

    void event_csr_matvec_heter_float_float(cudaStream_t stream, void **buffers,
                                             const char *opaque, std::size_t opaque_len);

    void event_csr_matvec_heter_double_bool(cudaStream_t stream, void **buffers,
                                              const char *opaque, std::size_t opaque_len);

    void event_csr_matvec_heter_double_double(cudaStream_t stream, void **buffers,
                                              const char *opaque, std::size_t opaque_len);

    void event_csr_matvec_homo_float_bool(cudaStream_t stream, void **buffers,
                                            const char *opaque, std::size_t opaque_len);

    void event_csr_matvec_homo_float_float(cudaStream_t stream, void **buffers,
                                            const char *opaque, std::size_t opaque_len);

    void event_csr_matvec_homo_double_bool(cudaStream_t stream, void **buffers,
                                             const char *opaque, std::size_t opaque_len);

    void event_csr_matvec_homo_double_double(cudaStream_t stream, void **buffers,
                                             const char *opaque, std::size_t opaque_len);

}


#endif //BRAINPYLIB_GPU_CSR_SPMV_CUH
