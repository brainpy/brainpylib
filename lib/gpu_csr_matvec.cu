//
// Created by Chaoming Wang on 2022/11/25.
//

#include "gpu_csr_matvec.cuh"

namespace brainpy_lib {
    namespace {


        /*
         * Helper functions
         */

        template<class T>
        __device__ T warp_reduce(T val) {
            /**
             *  For a thread at lane X in the warp, __shfl_down_sync(FULL_MASK, val, offset) gets
             *  the value of the val variable from the thread at lane X+offset of the same warp.
             *  The data exchange is performed between registers, and more efficient than going
             *  through shared memory, which requires a load, a store and an extra register to
             *  hold the address.
             */
            for (int offset = warpSize / 2; offset > 0; offset /= 2)
                val += __shfl_down_sync(FULL_WARP_MASK, val, offset);
            return val;
        }


        __device__ unsigned int prev_power_of_2(unsigned int n) {
            while (n & n - 1)
                n = n & n - 1;
            return n;
        }

        /*
         * General CSR SpMV implementation: CSR-Scalar
         * -------------------------------------------
         *
         * Each thread per row
         */

        template<typename data_type>
        __global__ void _csr_matvec_heter_scalar_kernel(
                const unsigned int n_rows,
                const unsigned int *col_ids,
                const unsigned int *row_ptr,
                const data_type *data,
                const data_type *vec,
                data_type *y
        ) {
            unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;
            if (row < n_rows) {
                const int row_start = row_ptr[row];
                const int row_end = row_ptr[row + 1];
                data_type sum = 0;
                for (unsigned int element = row_start; element < row_end; element++)
                    sum += data[element] * vec[col_ids[element]];
                y[row] = sum;
            }
        }

        template<typename data_type>
        __global__ void _csr_matvec_homo_scalar_kernel(
                const unsigned int n_rows,
                const unsigned int *col_ids,
                const unsigned int *row_ptr,
                const data_type &data,
                const data_type *vec,
                data_type *y
        ) {
            unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;
            if (row < n_rows) {
                const int row_start = row_ptr[row];
                const int row_end = row_ptr[row + 1];
                data_type sum = 0;
                for (unsigned int element = row_start; element < row_end; element++)
                    sum += data * vec[col_ids[element]];
                y[row] = sum;
            }
        }


        template<typename F>
        inline void csr_matvec_heter_scalar(cudaStream_t stream,
                                            void **buffers,
                                            const char *opaque,
                                            std::size_t opaque_len) {
            // size
            const DoubleSizeDescriptor &d = *UnpackDescriptor<DoubleSizeDescriptor>(opaque, opaque_len);
            const unsigned int n_row = d.size_x;
            const unsigned int n_col = d.size_y;

            // data
            const F *data = reinterpret_cast<const F *>(buffers[0]);
            const unsigned int *col_ids = reinterpret_cast<const unsigned int *>(buffers[1]);
            const unsigned int *row_ptr = reinterpret_cast<const unsigned int *>(buffers[2]);
            const F *vec = reinterpret_cast<const F *>(buffers[3]);
            F *y = reinterpret_cast<F *>(buffers[4]);

            // processing
            const int block_dim = 256;
            const int grid_dim = (n_row + block_dim - 1) / block_dim;
            cudaMemset(y, 0, sizeof(F) * n_col);
            _csr_matvec_heter_scalar_kernel<F><<<grid_dim, block_dim, 0, stream>>>(
                    n_row, col_ids, row_ptr, data, vec, y);
            ThrowIfError(cudaGetLastError());
        }


        template<typename F>
        inline void csr_matvec_homo_scalar(cudaStream_t stream,
                                           void **buffers,
                                           const char *opaque,
                                           std::size_t opaque_len) {
            // size
            const DoubleSizeDescriptor &d = *UnpackDescriptor<DoubleSizeDescriptor>(opaque, opaque_len);
            const unsigned int n_row = d.size_x;
            const unsigned int n_col = d.size_y;

            // data
            const F *data = reinterpret_cast<const F *>(buffers[0]);
            const unsigned int *col_ids = reinterpret_cast<const unsigned int *>(buffers[1]);
            const unsigned int *row_ptr = reinterpret_cast<const unsigned int *>(buffers[2]);
            const F *vec = reinterpret_cast<const F *>(buffers[3]);
            F *y = reinterpret_cast<F *>(buffers[4]);

            // processing
            const int block_dim = 256;
            const int grid_dim = (n_row + block_dim - 1) / block_dim;
            cudaMemset(y, 0, sizeof(F) * n_col);
            _csr_matvec_homo_scalar_kernel<F><<<grid_dim, block_dim, 0, stream>>>(
                    n_row, col_ids, row_ptr, data[0], vec, y);
            ThrowIfError(cudaGetLastError());
        }


        /*
         * CSR-Vector algorithm
         * --------------------
         *
         * each warp per row
         */
        template<typename data_type>
        __global__ void _csr_matvec_heter_vector_kernel(
                const unsigned int n_rows,
                const unsigned int *col_ids,
                const unsigned int *row_ptr,
                const data_type *data,
                const data_type *x,
                data_type *y
        ) {
            const unsigned int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
            const unsigned int warp_id = thread_id / 32; // one warp per row
            const unsigned int lane = thread_id % 32;

            data_type sum = 0;
            if (warp_id < n_rows) {
                const unsigned int row_start = row_ptr[warp_id];
                const unsigned int row_end = row_ptr[warp_id + 1];
                for (unsigned int element = row_start + lane; element < row_end; element += 32)
                    sum += data[element] * x[col_ids[element]];
            }
            sum = warp_reduce(sum);
            if (lane == 0 && warp_id < n_rows)
                y[warp_id] = sum;
        }

        template<typename data_type>
        __global__ void _csr_matvec_homo_vector_kernel(
                const unsigned int n_rows,
                const unsigned int *col_ids,
                const unsigned int *row_ptr,
                const data_type &data,
                const data_type *x,
                data_type *y
        ) {
            const unsigned int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
            const unsigned int warp_id = thread_id / 32; // one warp per row
            const unsigned int lane = thread_id % 32;

            data_type sum = 0;
            if (warp_id < n_rows) {
                const unsigned int row_start = row_ptr[warp_id];
                const unsigned int row_end = row_ptr[warp_id + 1];
                for (unsigned int element = row_start + lane; element < row_end; element += 32)
                    sum += data * x[col_ids[element]];
            }
            sum = warp_reduce(sum);
            if (lane == 0 && warp_id < n_rows)
                y[warp_id] = sum;
        }


        template<typename F>
        inline void csr_matvec_heter_vector(cudaStream_t stream,
                                            void **buffers,
                                            const char *opaque,
                                            std::size_t opaque_len) {
            // size
            const DoubleSizeDescriptor &d = *UnpackDescriptor<DoubleSizeDescriptor>(opaque, opaque_len);
            const unsigned int n_row = d.size_x;
            const unsigned int n_col = d.size_y;

            // data
            const F *data = reinterpret_cast<const F *>(buffers[0]);
            const unsigned int *col_ids = reinterpret_cast<const unsigned int *>(buffers[1]);
            const unsigned int *row_ptr = reinterpret_cast<const unsigned int *>(buffers[2]);
            const F *vec = reinterpret_cast<const F *>(buffers[3]);
            F *y = reinterpret_cast<F *>(buffers[4]);

            // processing
            const int block_dim = 512;
            const int grid_dim = (n_row * 32 + block_dim - 1) / block_dim;
            cudaMemset(y, 0, sizeof(F) * n_col);
            _csr_matvec_heter_vector_kernel<F><<<grid_dim, block_dim, 0, stream>>>(
                    n_row, col_ids, row_ptr, data, vec, y);
            ThrowIfError(cudaGetLastError());
        }


        template<typename F>
        inline void csr_matvec_homo_vector(cudaStream_t stream,
                                           void **buffers,
                                           const char *opaque,
                                           std::size_t opaque_len) {
            // size
            const DoubleSizeDescriptor &d = *UnpackDescriptor<DoubleSizeDescriptor>(opaque, opaque_len);
            const unsigned int n_row = d.size_x;
            const unsigned int n_col = d.size_y;

            // data
            const F *data = reinterpret_cast<const F *>(buffers[0]);
            const unsigned int *col_ids = reinterpret_cast<const unsigned int *>(buffers[1]);
            const unsigned int *row_ptr = reinterpret_cast<const unsigned int *>(buffers[2]);
            const F *vec = reinterpret_cast<const F *>(buffers[3]);
            F *y = reinterpret_cast<F *>(buffers[4]);

            // processing
            const int block_dim = 256;
            const int grid_dim = (n_row * 32 + block_dim - 1) / 256;
            cudaMemset(y, 0, sizeof(F) * n_col);
            _csr_matvec_homo_vector_kernel<F><<<grid_dim, block_dim, 0, stream>>>(
                    n_row, col_ids, row_ptr, data[0], vec, y);
            ThrowIfError(cudaGetLastError());
        }


        /*
         * CSR-Adaptive algorithm:
         * -----------------------
         *
         */
        template<typename data_type, unsigned int NNZ_PER_WG>
        __global__ void _csr_matvec_heter_adaptive_kernel(
                const unsigned int n_rows,
                const unsigned int *col_ids,
                const unsigned int *row_ptr,
                const unsigned int *row_blocks,
                const data_type *data,
                const data_type *x,
                data_type *y
        ) {
            const unsigned int block_row_begin = row_blocks[blockIdx.x];
            const unsigned int block_row_end = row_blocks[blockIdx.x + 1];
            const unsigned int nnz = row_ptr[block_row_end] - row_ptr[block_row_begin];

            __shared__ data_type cache[NNZ_PER_WG];

            if (block_row_end - block_row_begin > 1) {
                /// CSR-Stream case
                const unsigned int i = threadIdx.x;
                const unsigned int block_data_begin = row_ptr[block_row_begin];
                const unsigned int thread_data_begin = block_data_begin + i;
                const unsigned int threads_for_reduction = prev_power_of_2(
                        blockDim.x / (block_row_end - block_row_begin));

                if (i < nnz) {
                    cache[i] = data[thread_data_begin] * x[col_ids[thread_data_begin]];
                }
                __syncthreads();

                if (threads_for_reduction > 1) {
                    /// Reduce all non zeroes of row by multiple thread
                    const unsigned int thread_in_block = i % threads_for_reduction;
                    const unsigned int local_row = block_row_begin + i / threads_for_reduction;

                    data_type dot = 0.0;
                    if (local_row < block_row_end) {
                        const unsigned int local_first_element = row_ptr[local_row] - row_ptr[block_row_begin];
                        const unsigned int local_last_element = row_ptr[local_row + 1] - row_ptr[block_row_begin];

                        for (unsigned int local_element = local_first_element + thread_in_block;
                             local_element < local_last_element;
                             local_element += threads_for_reduction) {
                            dot += cache[local_element];
                        }
                    }
                    __syncthreads();
                    cache[i] = dot;

                    /// Now each row has threads_for_reduction values in cache
                    for (int j = threads_for_reduction / 2; j > 0; j /= 2) {
                        /// Reduce for each row
                        __syncthreads();

                        const bool use_result = thread_in_block < j && i + j < NNZ_PER_WG;
                        if (use_result)
                            dot += cache[i + j];
                        __syncthreads();

                        if (use_result)
                            cache[i] = dot;
                    }

                    if (thread_in_block == 0 && local_row < block_row_end) {
                        y[local_row] = dot;
                    }
                } else {
                    /// Reduce all non zeroes of row by single thread
                    unsigned int local_row = block_row_begin + i;
                    while (local_row < block_row_end) {
                        data_type dot = 0.0;
                        for (unsigned int j = row_ptr[local_row] - block_data_begin;
                             j < row_ptr[local_row + 1] - block_data_begin;
                             j++) {
                            dot += cache[j];
                        }
                        y[local_row] = dot;
                        local_row += NNZ_PER_WG;
                    }
                }
            } else {
                const unsigned int row = block_row_begin;
                const unsigned int warp_id = threadIdx.x / 32;
                const unsigned int lane = threadIdx.x % 32;

                data_type dot = 0;

                if (nnz <= 64 || NNZ_PER_WG <= 32) {
                    /// CSR-Vector case
                    if (row < n_rows) {
                        const unsigned int row_start = row_ptr[row];
                        const unsigned int row_end = row_ptr[row + 1];
                        for (unsigned int element = row_start + lane; element < row_end; element += 32)
                            dot += data[element] * x[col_ids[element]];
                    }
                    dot = warp_reduce(dot);
                    if (lane == 0 && warp_id == 0 && row < n_rows) {
                        y[row] = dot;
                    }

                } else {
                    /// CSR-VectorL case
                    if (row < n_rows) {
                        const unsigned int row_start = row_ptr[row];
                        const unsigned int row_end = row_ptr[row + 1];
                        for (unsigned int element = row_start + threadIdx.x; element < row_end; element += blockDim.x)
                            dot += data[element] * x[col_ids[element]];
                    }
                    dot = warp_reduce(dot);
                    if (lane == 0)
                        cache[warp_id] = dot;
                    __syncthreads();

                    if (warp_id == 0) {
                        dot = 0.0;
                        for (unsigned int element = lane; element < blockDim.x / 32; element += 32)
                            dot += cache[element];
                        dot = warp_reduce(dot);
                        if (lane == 0 && row < n_rows) {
                            y[row] = dot;
                        }
                    }
                }
            }
        }

        template<typename data_type, unsigned int NNZ_PER_WG>
        __global__ void _csr_matvec_homo_adaptive_kernel(
                const unsigned int n_rows,
                const unsigned int *col_ids,
                const unsigned int *row_ptr,
                const unsigned int *row_blocks,
                const data_type &data,
                const data_type *x,
                data_type *y
        ) {
            const unsigned int block_row_begin = row_blocks[blockIdx.x];
            const unsigned int block_row_end = row_blocks[blockIdx.x + 1];
            const unsigned int nnz = row_ptr[block_row_end] - row_ptr[block_row_begin];

            __shared__ data_type cache[NNZ_PER_WG];

            if (block_row_end - block_row_begin > 1) {
                /// CSR-Stream case
                const unsigned int i = threadIdx.x;
                const unsigned int block_data_begin = row_ptr[block_row_begin];
                const unsigned int thread_data_begin = block_data_begin + i;
                const unsigned int threads_for_reduction = prev_power_of_2(
                        blockDim.x / (block_row_end - block_row_begin));

                if (i < nnz) {
                    cache[i] = data * x[col_ids[thread_data_begin]];
                }
                __syncthreads();

                if (threads_for_reduction > 1) {
                    /// Reduce all non zeroes of row by multiple thread
                    const unsigned int thread_in_block = i % threads_for_reduction;
                    const unsigned int local_row = block_row_begin + i / threads_for_reduction;

                    data_type dot = 0.0;
                    if (local_row < block_row_end) {
                        const unsigned int local_first_element = row_ptr[local_row] - row_ptr[block_row_begin];
                        const unsigned int local_last_element = row_ptr[local_row + 1] - row_ptr[block_row_begin];

                        for (unsigned int local_element = local_first_element + thread_in_block;
                             local_element < local_last_element;
                             local_element += threads_for_reduction) {
                            dot += cache[local_element];
                        }
                    }
                    __syncthreads();
                    cache[i] = dot;

                    /// Now each row has threads_for_reduction values in cache
                    for (int j = threads_for_reduction / 2; j > 0; j /= 2) {
                        /// Reduce for each row
                        __syncthreads();

                        const bool use_result = thread_in_block < j && i + j < NNZ_PER_WG;
                        if (use_result)
                            dot += cache[i + j];
                        __syncthreads();

                        if (use_result)
                            cache[i] = dot;
                    }

                    if (thread_in_block == 0 && local_row < block_row_end) {
                        y[local_row] = dot;
                    }
                } else {
                    /// Reduce all non zeroes of row by single thread
                    unsigned int local_row = block_row_begin + i;
                    while (local_row < block_row_end) {
                        data_type dot = 0.0;
                        for (unsigned int j = row_ptr[local_row] - block_data_begin;
                             j < row_ptr[local_row + 1] - block_data_begin;
                             j++) {
                            dot += cache[j];
                        }
                        y[local_row] = dot;
                        local_row += NNZ_PER_WG;
                    }
                }
            } else {
                const unsigned int row = block_row_begin;
                const unsigned int warp_id = threadIdx.x / 32;
                const unsigned int lane = threadIdx.x % 32;

                data_type dot = 0;

                if (nnz <= 64 || NNZ_PER_WG <= 32) {
                    /// CSR-Vector case
                    if (row < n_rows) {
                        const unsigned int row_start = row_ptr[row];
                        const unsigned int row_end = row_ptr[row + 1];
                        for (unsigned int element = row_start + lane; element < row_end; element += 32)
                            dot += data * x[col_ids[element]];
                    }
                    dot = warp_reduce(dot);
                    if (lane == 0 && warp_id == 0 && row < n_rows) {
                        y[row] = dot;
                    }

                } else {
                    /// CSR-VectorL case
                    if (row < n_rows) {
                        const unsigned int row_start = row_ptr[row];
                        const unsigned int row_end = row_ptr[row + 1];
                        for (unsigned int element = row_start + threadIdx.x; element < row_end; element += blockDim.x)
                            dot += data * x[col_ids[element]];
                    }
                    dot = warp_reduce(dot);
                    if (lane == 0)
                        cache[warp_id] = dot;
                    __syncthreads();

                    if (warp_id == 0) {
                        dot = 0.0;
                        for (unsigned int element = lane; element < blockDim.x / 32; element += 32)
                            dot += cache[element];
                        dot = warp_reduce(dot);
                        if (lane == 0 && row < n_rows) {
                            y[row] = dot;
                        }
                    }
                }
            }
        }


        template<typename F>
        inline void csr_matvec_heter_adaptive(cudaStream_t stream,
                                              void **buffers,
                                              const char *opaque,
                                              std::size_t opaque_len) {
            // size
            const DoubleSizeDescriptor &d = *UnpackDescriptor<DoubleSizeDescriptor>(opaque, opaque_len);
            const unsigned int n_row = d.size_x;
            const unsigned int n_col = d.size_y;

            // data
            const F *data = reinterpret_cast<const F *>(buffers[0]);
            const unsigned int *col_ids = reinterpret_cast<const unsigned int *>(buffers[1]);
            const unsigned int *row_ptr = reinterpret_cast<const unsigned int *>(buffers[2]);
            const unsigned int *row_blocks = reinterpret_cast<const unsigned int *>(buffers[3]);
            const F *vec = reinterpret_cast<const F *>(buffers[4]);
            F *y = reinterpret_cast<F *>(buffers[5]);

            // processing
            const int block_dim = 512;
            const int grid_dim = (n_row * 32 + block_dim - 1) / block_dim;
            cudaMemset(y, 0, sizeof(F) * n_col);
            _csr_matvec_heter_adaptive_kernel<F, block_dim><<<grid_dim, block_dim, 0, stream>>>(
                    n_row, col_ids, row_ptr, row_blocks, data, vec, y);
            ThrowIfError(cudaGetLastError());
        }


        template<typename F>
        inline void csr_matvec_homo_adaptive(cudaStream_t stream,
                                             void **buffers,
                                             const char *opaque,
                                             std::size_t opaque_len) {
            // size
            const DoubleSizeDescriptor &d = *UnpackDescriptor<DoubleSizeDescriptor>(opaque, opaque_len);
            const unsigned int n_row = d.size_x;
            const unsigned int n_col = d.size_y;

            // data
            const F *data = reinterpret_cast<const F *>(buffers[0]);
            const unsigned int *col_ids = reinterpret_cast<const unsigned int *>(buffers[1]);
            const unsigned int *row_ptr = reinterpret_cast<const unsigned int *>(buffers[2]);
            const unsigned int *row_blocks = reinterpret_cast<const unsigned int *>(buffers[3]);
            const F *vec = reinterpret_cast<const F *>(buffers[4]);
            F *y = reinterpret_cast<F *>(buffers[5]);

            // processing
            const int block_dim = 256;
            const int grid_dim = (n_row * 32 + block_dim - 1) / 256;
            cudaMemset(y, 0, sizeof(F) * n_col);
            _csr_matvec_homo_adaptive_kernel<F, block_dim><<<grid_dim, block_dim, 0, stream>>>(
                    n_row, col_ids, row_ptr, row_blocks, data[0], vec, y);
            ThrowIfError(cudaGetLastError());
        }


    }

    void csr_matvec_heter_scalar_float(cudaStream_t stream, void **buffers,
                                       const char *opaque, std::size_t opaque_len) {
        csr_matvec_heter_scalar<float>(stream, buffers, opaque, opaque_len);
    }

    void csr_matvec_heter_scalar_double(cudaStream_t stream, void **buffers,
                                        const char *opaque, std::size_t opaque_len) {
        csr_matvec_heter_scalar<double>(stream, buffers, opaque, opaque_len);
    }

    void csr_matvec_heter_vector_float(cudaStream_t stream, void **buffers,
                                       const char *opaque, std::size_t opaque_len) {
        csr_matvec_heter_vector<float>(stream, buffers, opaque, opaque_len);
    }

    void csr_matvec_heter_vector_double(cudaStream_t stream, void **buffers,
                                        const char *opaque, std::size_t opaque_len) {
        csr_matvec_heter_vector<double>(stream, buffers, opaque, opaque_len);
    }

    void csr_matvec_heter_adaptive_float(cudaStream_t stream, void **buffers,
                                         const char *opaque, std::size_t opaque_len) {
        csr_matvec_heter_adaptive<float>(stream, buffers, opaque, opaque_len);
    }

    void csr_matvec_heter_adaptive_double(cudaStream_t stream, void **buffers,
                                          const char *opaque, std::size_t opaque_len) {
        csr_matvec_heter_adaptive<double>(stream, buffers, opaque, opaque_len);
    }

    void csr_matvec_homo_scalar_float(cudaStream_t stream, void **buffers,
                                      const char *opaque, std::size_t opaque_len) {
        csr_matvec_homo_scalar<float>(stream, buffers, opaque, opaque_len);
    }

    void csr_matvec_homo_scalar_double(cudaStream_t stream, void **buffers,
                                       const char *opaque, std::size_t opaque_len) {
        csr_matvec_homo_scalar<double>(stream, buffers, opaque, opaque_len);
    }

    void csr_matvec_homo_vector_float(cudaStream_t stream, void **buffers,
                                      const char *opaque, std::size_t opaque_len) {
        csr_matvec_homo_vector<float>(stream, buffers, opaque, opaque_len);
    }

    void csr_matvec_homo_vector_double(cudaStream_t stream, void **buffers,
                                       const char *opaque, std::size_t opaque_len) {
        csr_matvec_homo_vector<double>(stream, buffers, opaque, opaque_len);
    }

    void csr_matvec_homo_adaptive_float(cudaStream_t stream, void **buffers,
                                        const char *opaque, std::size_t opaque_len) {
        csr_matvec_homo_adaptive<float>(stream, buffers, opaque, opaque_len);
    }

    void csr_matvec_homo_adaptive_double(cudaStream_t stream, void **buffers,
                                         const char *opaque, std::size_t opaque_len) {
        csr_matvec_homo_adaptive<double>(stream, buffers, opaque, opaque_len);
    }

}



