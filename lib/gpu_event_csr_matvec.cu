//
// Created by Chaoming Wang on 2022/11/28.
//

#include "gpu_event_csr_matvec.cuh"


#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600
#else
__device__ double atomicAdd(double* address, double val)
        {
            unsigned long long int* address_as_ull = (unsigned long long int*)address;
            unsigned long long int old = *address_as_ull, assumed;
            do {
                assumed = old;
                old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val + __longlong_as_double(assumed)));
            // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
            } while (assumed != old);
            return __longlong_as_double(old);
        }
#endif



namespace brainpy_lib {
    namespace {


        /*
         * Helper functions
         */

        template<class T>
        __device__ T warp_reduce(T val) {
            for (int offset = warpSize / 2; offset > 0; offset /= 2)
                val += __shfl_down_sync(FULL_WARP_MASK, val, offset);
            return val;
        }


        /*
         * CSR-Vector algorithm
         * --------------------
         *
         * each warp per row
         */
        template<typename T1, typename T2>
        __global__ void _event_csr_matvec_heter_kernel(
                const unsigned int n_rows,
                const unsigned int *col_ids,
                const unsigned int *row_ptr,
                const T1 *data,
                const T2 *x,
                T1 *y
        ) {
            const unsigned int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
            const unsigned int warp_id = thread_id / 32; // one warp per row
            const unsigned int lane = thread_id % 32;

            T1 sum = 0;
            if (warp_id < n_rows) {
                const unsigned int row_start = row_ptr[warp_id];
                const unsigned int row_end = row_ptr[warp_id + 1];
                for (unsigned int element = row_start + lane; element < row_end; element += 32)
                    if (x[col_ids[element]])
                        sum += data[element];
            }
            sum = warp_reduce(sum);
            if (lane == 0 && warp_id < n_rows)
                y[warp_id] = sum;
        }


        template<typename T1, typename T2>
        __global__ void _event_csr_matvec_transpose_heter_kernel(
                const unsigned int n_rows,
                const unsigned int *col_ids,
                const unsigned int *row_ptr,
                const T1 *data,
                const T2 *x,
                T1 *y
        ) {
            const unsigned int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
            const unsigned int warp_id = thread_id / 32; // one warp per row
            const unsigned int lane = thread_id % 32;

            if (warp_id < n_rows) {
                if (x[warp_id]) {
                    const unsigned int row_start = row_ptr[warp_id];
                    const unsigned int row_end = row_ptr[warp_id + 1];
                    for (unsigned int element = row_start + lane; element < row_end; element += 32)
                        atomicAdd(&y[col_ids[element]], data[element]);
                }
            }
        }

        template<typename T1, typename T2>
        __global__ void _event_csr_matvec_homo_kernel(
                const unsigned int n_rows,
                const unsigned int *col_ids,
                const unsigned int *row_ptr,
                const T1 &data,
                const T2 *x,
                T1 *y
        ) {
            const unsigned int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
            const unsigned int warp_id = thread_id / 32; // one warp per row
            const unsigned int lane = thread_id % 32;

            T1 sum = 0;
            if (warp_id < n_rows) {
                const unsigned int row_start = row_ptr[warp_id];
                const unsigned int row_end = row_ptr[warp_id + 1];
                for (unsigned int element = row_start + lane; element < row_end; element += 32)
                    if (x[col_ids[element]])
                        sum += data;
            }
            sum = warp_reduce(sum);
            if (lane == 0 && warp_id < n_rows)
                y[warp_id] = sum;
        }


        template<typename T1, typename T2>
        __global__ void _event_csr_matvec_transpose_homo_kernel(
                const unsigned int n_rows,
                const unsigned int *col_ids,
                const unsigned int *row_ptr,
                const T1 &data,
                const T2 *x,
                T1 *y
        ) {
            const unsigned int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
            const unsigned int warp_id = thread_id / 32; // one warp per row
            const unsigned int lane = thread_id % 32;

            if (warp_id < n_rows) {
                if (x[warp_id]) {
                    const unsigned int row_start = row_ptr[warp_id];
                    const unsigned int row_end = row_ptr[warp_id + 1];
                    for (unsigned int element = row_start + lane; element < row_end; element += 32)
                        atomicAdd(&y[col_ids[element]], data);
                }
            }
        }


        template<typename F, typename VT>
        inline void event_csr_matvec_heter(cudaStream_t stream,
                                           void **buffers,
                                           const char *opaque,
                                           std::size_t opaque_len) {
            // size
            const TwoUintOneBoolDescriptor &d = *UnpackDescriptor<TwoUintOneBoolDescriptor>(opaque, opaque_len);
            const unsigned int n_row = d.uint_x;
            const unsigned int n_col = d.uint_y;
            const bool transpose = d.bool_x;

            // data
            const F *data = reinterpret_cast<const F *>(buffers[0]);
            const unsigned int *col_ids = reinterpret_cast<const unsigned int *>(buffers[1]);
            const unsigned int *row_ptr = reinterpret_cast<const unsigned int *>(buffers[2]);
            const VT *vec = reinterpret_cast<const VT *>(buffers[3]);
            F *y = reinterpret_cast<F *>(buffers[4]);

            // processing
            const int block_dim = 512;
            const int grid_dim = (n_row * 32 + block_dim - 1) / block_dim;
            if (transpose) {
                cudaMemset(y, 0, sizeof(F) * n_col);
                _event_csr_matvec_transpose_heter_kernel<F, VT><<<grid_dim, block_dim, 0, stream>>>(
                        n_row, col_ids, row_ptr, data, vec, y
                );
            } else {
                cudaMemset(y, 0, sizeof(F) * n_row);
                _event_csr_matvec_heter_kernel<F, VT><<<grid_dim, block_dim, 0, stream>>>(
                        n_row, col_ids, row_ptr, data, vec, y
                );
            }
            ThrowIfError(cudaGetLastError());
        }


        template<typename F, typename VT>
        inline void event_csr_matvec_homo(cudaStream_t stream,
                                          void **buffers,
                                          const char *opaque,
                                          std::size_t opaque_len) {
            // size
            const TwoUintOneBoolDescriptor &d = *UnpackDescriptor<TwoUintOneBoolDescriptor>(opaque, opaque_len);
            const unsigned int n_row = d.uint_x;
            const unsigned int n_col = d.uint_y;
            const bool transpose = d.bool_x;

            // data
            const F *data = reinterpret_cast<const F *>(buffers[0]);
            const unsigned int *col_ids = reinterpret_cast<const unsigned int *>(buffers[1]);
            const unsigned int *row_ptr = reinterpret_cast<const unsigned int *>(buffers[2]);
            const VT *vec = reinterpret_cast<const VT *>(buffers[3]);
            F *y = reinterpret_cast<F *>(buffers[4]);

            // processing
            const int block_dim = 256;
            const int grid_dim = (n_row * 32 + block_dim - 1) / 256;
            if (transpose) {
                cudaMemset(y, 0, sizeof(F) * n_col);
                _event_csr_matvec_transpose_homo_kernel<F, VT><<<grid_dim, block_dim, 0, stream>>>(
                        n_row, col_ids, row_ptr, data[0], vec, y
                );
            } else {
                cudaMemset(y, 0, sizeof(F) * n_row);
                _event_csr_matvec_homo_kernel<F, VT><<<grid_dim, block_dim, 0, stream>>>(
                        n_row, col_ids, row_ptr, data[0], vec, y
                );
            }
            ThrowIfError(cudaGetLastError());
        }


    }

    void event_csr_matvec_heter_float_bool(cudaStream_t stream, void **buffers,
                                           const char *opaque, std::size_t opaque_len) {
        event_csr_matvec_heter<float, bool>(stream, buffers, opaque, opaque_len);
    }

    void event_csr_matvec_heter_float_float(cudaStream_t stream, void **buffers,
                                            const char *opaque, std::size_t opaque_len) {
        event_csr_matvec_heter<float, float>(stream, buffers, opaque, opaque_len);
    }

    void event_csr_matvec_heter_double_bool(cudaStream_t stream, void **buffers,
                                            const char *opaque, std::size_t opaque_len) {
        event_csr_matvec_heter<double, bool>(stream, buffers, opaque, opaque_len);
    }

    void event_csr_matvec_heter_double_double(cudaStream_t stream, void **buffers,
                                              const char *opaque, std::size_t opaque_len) {
        event_csr_matvec_heter<double, double>(stream, buffers, opaque, opaque_len);
    }

    void event_csr_matvec_homo_float_bool(cudaStream_t stream, void **buffers,
                                          const char *opaque, std::size_t opaque_len) {
        event_csr_matvec_homo<float, bool>(stream, buffers, opaque, opaque_len);
    }

    void event_csr_matvec_homo_float_float(cudaStream_t stream, void **buffers,
                                           const char *opaque, std::size_t opaque_len) {
        event_csr_matvec_homo<float, float>(stream, buffers, opaque, opaque_len);
    }

    void event_csr_matvec_homo_double_bool(cudaStream_t stream, void **buffers,
                                           const char *opaque, std::size_t opaque_len) {
        event_csr_matvec_homo<double, bool>(stream, buffers, opaque, opaque_len);
    }

    void event_csr_matvec_homo_double_double(cudaStream_t stream, void **buffers,
                                             const char *opaque, std::size_t opaque_len) {
        event_csr_matvec_homo<double, double>(stream, buffers, opaque, opaque_len);
    }

}



