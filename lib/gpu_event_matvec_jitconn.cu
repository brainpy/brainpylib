//
// Created by adadu on 2022/11/30.
//

#include "gpu_event_matvec_jitconn.cuh"

namespace brainpy_lib {
    namespace {

        template<typename T, const int BLOCK_SIZE>
        __global__ void _event_mv_C_fixedprob_W_homo(
                const int *event_ids,  /* event */
                const int &event_num,
                const unsigned int conn_seed,  /* matrix */
                const float conn_prob,
                const T &weight,
                const unsigned int num_row,  /* shape */
                const unsigned int num_col,
                T *out  /* output */
        ) {
            __shared__ int shEvents[BLOCK_SIZE];

            const unsigned int idx = threadIdx.x;
            const unsigned int row_i = blockIdx.x * blockDim.x + idx;

            if (idx < event_num)
                shEvents[idx] = event_ids[idx];
            __syncthreads();

            T sum = 0;
            int event_read_i = 0;
            while (true) {
#pragma unroll
                for (int sh_i = 0; sh_i < BLOCK_SIZE; sh_i++) {
                    if (sh_i + event_read_i < event_num) {
                        if (lfsr113_double(conn_seed + num_col * row_i + shEvents[sh_i]) < conn_prob)
                            sum += weight;
                    }
                }
                event_read_i += BLOCK_SIZE;
                if (event_read_i < event_num) {
                    if (idx + event_read_i < event_num)
                        shEvents[idx] = event_ids[idx + event_read_i];
                    __syncthreads();
                } else {
                    break;
                }
            }
            if (row_i < num_row) {
                out[row_i] = sum;
            }
        }

        template<typename T, const int BLOCK_SIZE>
        __global__ void _event_mv_transpose_C_fixedprob_W_homo(
                const int *event_ids,  /* event */
                const int &event_num,
                const unsigned int conn_seed,  /* matrix */
                const float conn_prob,
                const T &weight,
                const unsigned int num_row,  /* shape */
                const unsigned int num_col,
                T *out  /* output */
        ) {
            __shared__ int shEvents[BLOCK_SIZE];

            const unsigned int idx = threadIdx.x;
            const unsigned int col_i = blockIdx.x * blockDim.x + idx;

            if (idx < event_num)
                shEvents[idx] = event_ids[idx];
            __syncthreads();

            T sum = 0;
            int event_read_i = 0;
            while (true) {
#pragma unroll
                for (int sh_i = 0; sh_i < BLOCK_SIZE; sh_i++) {
                    if (sh_i + event_read_i < event_num) {
                        if (lfsr113_double(conn_seed + num_col * shEvents[sh_i] + col_i) < conn_prob)
                            sum += weight;
                    }
                }
                event_read_i += BLOCK_SIZE;
                if (event_read_i < event_num) {
                    if (event_read_i + idx < event_num)
                        shEvents[idx] = event_ids[event_read_i + idx];
                    __syncthreads();
                } else {
                    break;
                }
            }
            if (col_i < num_col) {
                out[col_i] = sum;
            }
        }

//        <typename T>
//        __global__ void _event_mv_transpose_C_fixedprob_W_uniform(
//                /* event */
//                const int *event_ids,
//                const int &event_num,
//
//                /* matrix */
//                const unsigned int weight_seed,
//                const unsigned int conn_seed,
//                const float conn_prob,
//
//                /* shape */
//                const unsigned int num_row,
//                const unsigned int num_col,
//
//                /* output */
//                T *out
//        ) {
//
//        }
//
//        <typename T>
//        __global__ void _event_mv_transpose_C_fixedprob_W_normal(
//                /* event */
//                const int *event_ids,
//                const int &event_num,
//
//                /* matrix */
//                const unsigned int weight_seed,
//                const unsigned int conn_seed,
//                const float conn_prob,
//
//                /* shape */
//                const unsigned int num_row,
//                const unsigned int num_col,
//
//                /* output */
//                T *out
//        ) {
//
//        }


        template<typename T>
        inline void event_mv_C_fixedprob_W_homo(cudaStream_t stream,
                                                void **buffers,
                                                const char *opaque,
                                                std::size_t opaque_len) {
            // size
            const EventMVRandomDescriptor &d = *UnpackDescriptor<EventMVRandomDescriptor>(opaque, opaque_len);
            const unsigned int n_row = d.n_row;
            const unsigned int n_col = d.n_col;
            const unsigned int conn_seed = d.seed;
            const float conn_prob = d.prob;
            const bool transpose = d.transpose;

            // data
            const int *event_ids = reinterpret_cast<const int *>(buffers[0]);
            const int *event_num = reinterpret_cast<const int *>(buffers[1]);
            const T *weight = reinterpret_cast<const T *>(buffers[2]);
            T *y = reinterpret_cast<T *>(buffers[3]);

            // processing
            const int block_dim = 256;
            const int grid_dim = (n_col + block_dim - 1) / block_dim;
            if (transpose) {
                cudaMemset(y, 0, sizeof(T) * n_col);
                _event_mv_transpose_C_fixedprob_W_homo<T, block_dim><<<grid_dim, block_dim, 0, stream>>>(
                        event_ids, event_num[0], conn_seed, conn_prob, weight[0], n_row, n_col, y
                );
            } else {
                cudaMemset(y, 0, sizeof(T) * n_row);
                _event_mv_C_fixedprob_W_homo<T, block_dim><<<grid_dim, block_dim, 0, stream>>>(
                        event_ids, event_num[0], conn_seed, conn_prob, weight[0], n_row, n_col, y
                );
            }
            ThrowIfError(cudaGetLastError());
        }

    }

    void event_mv_C_fixedprob_W_homo_float(cudaStream_t stream, void **buffers,
                                           const char *opaque, std::size_t opaque_len) {
        event_mv_C_fixedprob_W_homo<float>(stream, buffers, opaque, opaque_len);
    }

    void event_mv_C_fixedprob_W_homo_double(cudaStream_t stream, void **buffers,
                                            const char *opaque, std::size_t opaque_len) {
        event_mv_C_fixedprob_W_homo<double>(stream, buffers, opaque, opaque_len);
    }


}

