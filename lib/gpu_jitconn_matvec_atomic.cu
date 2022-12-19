//
// Created by adadu on 2022/12/1.
//

#include "gpu_jitconn_matvec_atomic.cuh"


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

        template<typename T, const int BLOCK_SIZE>
        __global__ void _jitconn_prob_homo_v2(
                const T *vector,  /* vector */
                const unsigned int conn_seed,  /* matrix */
                const float conn_prob,
                const unsigned int num_row,  /* shape */
                const unsigned int num_col,
                T *out  /* output */
        ) {
            const unsigned int col_i = blockIdx.x * blockDim.x + threadIdx.x;

            if (col_i < num_col) {

                // random state
                curandState state;
                curand_init(conn_seed + col_i, 0, 0, &state);

                // summation
                T v = vector[col_i];
                int row_i = (int) ceil(log(curand_uniform(&state)) / conn_prob);
                while (row_i < num_row) {
                    atomicAdd(&out[row_i], v);
                    row_i += (int) ceil(log(curand_uniform(&state)) / conn_prob);
                }
            }
        }


        template<typename T>
        inline void matvec_atomic_jitconn_prob_homo_v2(cudaStream_t stream,
                                                       void **buffers,
                                                       const char *opaque,
                                                       std::size_t opaque_len) {
            // size
            const JITConnProbHomoDescriptor &d = *UnpackDescriptor<JITConnProbHomoDescriptor>(opaque, opaque_len);
            const unsigned int n_row = d.n_row;
            const unsigned int n_col = d.n_col;
            const unsigned int conn_seed = d.seed;
            const float conn_prob = d.prob;

            // data
            const T *vector = reinterpret_cast<const T *>(buffers[0]);
            T *y = reinterpret_cast<T *>(buffers[1]);

            // processing
            const int block_dim = 256;
            const int grid_dim = (n_col + block_dim - 1) / block_dim;
            cudaMemset(y, 0, sizeof(T) * n_row);
            _jitconn_prob_homo_v2<T, block_dim><<<grid_dim, block_dim, 0, stream>>>(
                    vector, conn_seed, conn_prob, n_row, n_col, y
            );
            ThrowIfError(cudaGetLastError());
        }


        template<typename T, const int BLOCK_SIZE>
        __global__ void _jitconn_prob_uniform_v2(
                const T *vector,  /* vector */
                const unsigned int conn_seed,  /* matrix */
                const float conn_prob,
                const float w_min,
                const float w_range,
                const unsigned int num_row,  /* shape */
                const unsigned int num_col,
                T *out  /* output */
        ) {
            const unsigned int col_i = blockIdx.x * blockDim.x + threadIdx.x;

            if (col_i < num_col) {

                // random state
                curandState state;
                curand_init(conn_seed + col_i, 0, 0, &state);

                // summation
                T v = vector[col_i];
                int row_i = (int) ceil(log(curand_uniform(&state)) / conn_prob);
                while (row_i < num_row) {
                    atomicAdd(&out[row_i], v * (curand_uniform(&state) * w_range + w_min));
                    row_i += (int) ceil(log(curand_uniform(&state)) / conn_prob);
                }
            }
        }


        template<typename T>
        inline void matvec_atomic_jitconn_prob_uniform_v2(cudaStream_t stream,
                                                          void **buffers,
                                                          const char *opaque,
                                                          std::size_t opaque_len) {
            // size
            const JITConnProbUniformDescriptor &d = *UnpackDescriptor<JITConnProbUniformDescriptor>(opaque,
                                                                                                    opaque_len);
            const unsigned int n_row = d.n_row;
            const unsigned int n_col = d.n_col;
            const unsigned int conn_seed = d.seed;
            const float conn_prob = d.prob;
            const float w_min = d.w_min;
            const float w_range = d.w_range;

            // data
            const T *vector = reinterpret_cast<const T *>(buffers[0]);
            T *y = reinterpret_cast<T *>(buffers[1]);

            // processing
            const int block_dim = 256;
            const int grid_dim = (n_col + block_dim - 1) / block_dim;
            cudaMemset(y, 0, sizeof(T) * n_row);
            _jitconn_prob_uniform_v2<T, block_dim><<<grid_dim, block_dim, 0, stream>>>(
                    vector, conn_seed, conn_prob, w_min, w_range, n_row, n_col, y
            );
            ThrowIfError(cudaGetLastError());
        }


        template<typename T, const int BLOCK_SIZE>
        __global__ void _jitconn_prob_normal_v2(
                const T *vector,  /* vector */
                const unsigned int conn_seed,  /* matrix */
                const float log_prob,
                const float w_mu,
                const float w_sigma,
                const unsigned int num_row,  /* shape */
                const unsigned int num_col,
                T *out  /* output */
        ) {

            const unsigned int col_i = blockIdx.x * blockDim.x + threadIdx.x;

            if (col_i < num_col) {
                // random state
                curandState state;
                curand_init(conn_seed + col_i, 0, 0, &state);

                // summation
                T v = vector[col_i];
                int row_i = (int) ceil(log(curand_uniform(&state)) / log_prob);
                while (row_i < num_row) {
                    atomicAdd(&out[row_i], v * (curand_normal(&state) * w_sigma + w_mu));
                    row_i += (int) ceil(log(curand_uniform(&state)) / log_prob);
                }
            }
        }


        template<typename T>
        inline void matvec_atomic_jitconn_prob_normal_v2(cudaStream_t stream,
                                                         void **buffers,
                                                         const char *opaque,
                                                         std::size_t opaque_len) {
            // size
            const JITConnProbNormalDescriptor &d = *UnpackDescriptor<JITConnProbNormalDescriptor>(opaque,
                                                                                                  opaque_len);
            const unsigned int n_row = d.n_row;
            const unsigned int n_col = d.n_col;
            const unsigned int conn_seed = d.seed;
            const float conn_prob = d.prob;
            const float w_mu = d.w_mu;
            const float w_sigma = d.w_sigma;

            // data
            const T *vector = reinterpret_cast<const T *>(buffers[0]);
            T *y = reinterpret_cast<T *>(buffers[1]);

            // processing
            const int block_dim = 256;
            const int grid_dim = (n_col + block_dim - 1) / block_dim;
            cudaMemset(y, 0, sizeof(T) * n_row);
            _jitconn_prob_normal_v2<T, block_dim><<<grid_dim, block_dim, 0, stream>>>(
                    vector, conn_seed, conn_prob, w_mu, w_sigma, n_row, n_col, y
            );
            ThrowIfError(cudaGetLastError());
        }


    }

    void matvec_atomic_jitconn_prob_homo_v2_float(cudaStream_t stream, void **buffers,
                                                  const char *opaque, std::size_t opaque_len) {
        matvec_atomic_jitconn_prob_homo_v2<float>(stream, buffers, opaque, opaque_len);
    }

    void matvec_atomic_jitconn_prob_homo_v2_double(cudaStream_t stream, void **buffers,
                                                   const char *opaque, std::size_t opaque_len) {
        matvec_atomic_jitconn_prob_homo_v2<double>(stream, buffers, opaque, opaque_len);
    }

    void matvec_atomic_jitconn_prob_uniform_v2_float(cudaStream_t stream, void **buffers,
                                                     const char *opaque, std::size_t opaque_len) {
        matvec_atomic_jitconn_prob_uniform_v2<float>(stream, buffers, opaque, opaque_len);
    }

    void matvec_atomic_jitconn_prob_uniform_v2_double(cudaStream_t stream, void **buffers,
                                                      const char *opaque, std::size_t opaque_len) {
        matvec_atomic_jitconn_prob_uniform_v2<double>(stream, buffers, opaque, opaque_len);
    }

    void matvec_atomic_jitconn_prob_normal_v2_float(cudaStream_t stream, void **buffers,
                                                    const char *opaque, std::size_t opaque_len) {
        matvec_atomic_jitconn_prob_normal_v2<float>(stream, buffers, opaque, opaque_len);
    }

    void matvec_atomic_jitconn_prob_normal_v2_double(cudaStream_t stream, void **buffers,
                                                     const char *opaque, std::size_t opaque_len) {
        matvec_atomic_jitconn_prob_normal_v2<double>(stream, buffers, opaque, opaque_len);
    }


}

