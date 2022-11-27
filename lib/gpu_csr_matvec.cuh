//
// Created by adadu on 2022/11/25.
//

#ifndef BRAINPYLIB_GPU_CSR_SPMV_CUH
#define BRAINPYLIB_GPU_CSR_SPMV_CUH


#include "kernel_helper_descriptor.cuh"
#include "kernel_helper_constant.cuh"

namespace brainpy_lib{
    void csr_matvec_heter_scalar_float(cudaStream_t stream, void **buffers,
                                       const char *opaque, std::size_t opaque_len);
    void csr_matvec_heter_scalar_double(cudaStream_t stream, void **buffers,
                                        const char *opaque, std::size_t opaque_len);
    void csr_matvec_homo_scalar_float(cudaStream_t stream, void **buffers,
                                      const char *opaque, std::size_t opaque_len);
    void csr_matvec_homo_scalar_double(cudaStream_t stream, void **buffers,
                                       const char *opaque, std::size_t opaque_len);

    void csr_matvec_heter_vector_float(cudaStream_t stream, void **buffers,
                                       const char *opaque, std::size_t opaque_len);
    void csr_matvec_heter_vector_double(cudaStream_t stream, void **buffers,
                                        const char *opaque, std::size_t opaque_len);
    void csr_matvec_homo_vector_float(cudaStream_t stream, void **buffers,
                                      const char *opaque, std::size_t opaque_len);
    void csr_matvec_homo_vector_double(cudaStream_t stream, void **buffers,
                                       const char *opaque, std::size_t opaque_len);

    void csr_matvec_heter_adaptive_float(cudaStream_t stream, void **buffers,
                                         const char *opaque, std::size_t opaque_len);
    void csr_matvec_heter_adaptive_double(cudaStream_t stream, void **buffers,
                                          const char *opaque, std::size_t opaque_len);
    void csr_matvec_homo_adaptive_float(cudaStream_t stream, void **buffers,
                                        const char *opaque, std::size_t opaque_len);
    void csr_matvec_homo_adaptive_double(cudaStream_t stream, void **buffers,
                                         const char *opaque, std::size_t opaque_len);
}


#endif //BRAINPYLIB_GPU_CSR_SPMV_CUH
