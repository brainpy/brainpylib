//
// Created by adadu on 2022/12/1.
//

#include "gpu_jitconn_matvec.cuh"

namespace brainpy_lib {
    namespace {

        template<typename T, const int BLOCK_SIZE>
        __global__ void _jitconn_prob_homo(
                const T *vector,  /* vector */
                const unsigned int conn_seed,  /* matrix */
                const float conn_prob,
                const unsigned int num_row,  /* shape */
                const unsigned int num_col,
                T *out  /* output */
        ) {
            __shared__ T shVector[BLOCK_SIZE];

            const unsigned int idx = threadIdx.x;
            const unsigned int row_i = blockIdx.x * blockDim.x + idx;
            const unsigned int num_block = num_col / BLOCK_SIZE;
            const unsigned int num_other = num_col % BLOCK_SIZE;

            // random state
            curandState state;
            curand_init(conn_seed + row_i, 0, 0, &state);

            // summation
            T sum = 0;
            for (int i = 0; i < num_block; i++) {
                shVector[idx] = vector[i * BLOCK_SIZE + idx];
                __syncthreads();
#pragma unroll
                for (int sh_i = 0; sh_i < BLOCK_SIZE; sh_i++) {
                    if (curand_uniform(&state) < conn_prob)
                        sum += shVector[sh_i];
                }
            }
            if (idx < num_other) {
                shVector[idx] = vector[num_block * BLOCK_SIZE + idx];
            }
            __syncthreads();
            for (int sh_i = 0; sh_i < num_other; sh_i++) {
                if (curand_uniform(&state) < conn_prob)
                    sum += shVector[sh_i];
            }

            // write
            if (row_i < num_row) {
                out[row_i] = sum;
            }
        }


        template<typename T>
        inline void matvec_jitconn_prob_homo(cudaStream_t stream,
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
            const int grid_dim = (n_row + block_dim - 1) / block_dim;
            cudaMemset(y, 0, sizeof(T) * n_row);
            _jitconn_prob_homo<T, block_dim><<<grid_dim, block_dim, 0, stream>>>(
                    vector, conn_seed, conn_prob, n_row, n_col, y
            );
            ThrowIfError(cudaGetLastError());
        }


        template<typename T, const int BLOCK_SIZE>
        __global__ void _jitconn_prob_homo_v2(
                const T *vector,  /* vector */
                const unsigned int conn_seed,  /* matrix */
                const float conn_prob,
                const unsigned int num_row,  /* shape */
                const unsigned int num_col,
                T *out  /* output */
        ) {
            const unsigned int row_i = blockIdx.x * blockDim.x + threadIdx.x;

            if (row_i < num_row) {

                // random state
                curandState state;
                curand_init(conn_seed + row_i, 0, 0, &state);

                // summation
                T sum = 0;
                int syn_arrival_id = (int) ceil(log(curand_uniform(&state)) / conn_prob);
                while (syn_arrival_id < num_col){
                    sum += vector[syn_arrival_id];
                    syn_arrival_id += (int) ceil(log(curand_uniform(&state)) / conn_prob);
                }

                // write
                out[row_i] = sum;
            }
        }


        template<typename T>
        inline void matvec_jitconn_prob_homo_v2(cudaStream_t stream,
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
            const int grid_dim = (n_row + block_dim - 1) / block_dim;
            cudaMemset(y, 0, sizeof(T) * n_row);
            _jitconn_prob_homo_v2<T, block_dim><<<grid_dim, block_dim, 0, stream>>>(
                    vector, conn_seed, conn_prob, n_row, n_col, y
            );
            ThrowIfError(cudaGetLastError());
        }

        template<typename T, const int BLOCK_SIZE>
        __global__ void _jitconn_prob_uniform(
                const T *vector,  /* vector */
                const unsigned int conn_seed,  /* matrix */
                const float conn_prob,
                const float w_min,
                const float w_range,
                const unsigned int num_row,  /* shape */
                const unsigned int num_col,
                T *out  /* output */
        ) {
            __shared__ T shVector[BLOCK_SIZE];

            const unsigned int idx = threadIdx.x;
            const unsigned int row_i = blockIdx.x * blockDim.x + idx;
            const unsigned int num_block = num_col / BLOCK_SIZE;
            const unsigned int num_other = num_col % BLOCK_SIZE;

            // random state
            curandState state;
            curand_init(conn_seed + row_i, 0, 0, &state);

            // summation
            T sum = 0;
            for (int i = 0; i < num_block; i++) {
                shVector[idx] = vector[i * BLOCK_SIZE + idx];
                __syncthreads();
#pragma unroll
                for (int sh_i = 0; sh_i < BLOCK_SIZE; sh_i++) {
                    if (curand_uniform(&state) < conn_prob)
                        sum += shVector[sh_i] * (curand_uniform(&state) * w_range + w_min);
                }
            }
            if (idx < num_other) {
                shVector[idx] = vector[num_block * BLOCK_SIZE + idx];
            }
            __syncthreads();
            for (int sh_i = 0; sh_i < num_other; sh_i++) {
                if (curand_uniform(&state) < conn_prob)
                    sum += shVector[sh_i] * (curand_uniform(&state) * w_range + w_min);
            }

            // write
            if (row_i < num_row) {
                out[row_i] = sum;
            }
        }


        template<typename T>
        inline void matvec_jitconn_prob_uniform(cudaStream_t stream,
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
            const int grid_dim = (n_row + block_dim - 1) / block_dim;
            cudaMemset(y, 0, sizeof(T) * n_row);
            _jitconn_prob_uniform<T, block_dim><<<grid_dim, block_dim, 0, stream>>>(
                    vector, conn_seed, conn_prob, w_min, w_range, n_row, n_col, y
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
            const unsigned int row_i = blockIdx.x * blockDim.x + threadIdx.x;

            if (row_i < num_row) {

                // random state
                curandState state;
                curand_init(conn_seed + row_i, 0, 0, &state);

                // summation
                T sum = 0;
                int syn_arrival_id = (int) ceil(log(curand_uniform(&state)) / conn_prob);
                while (syn_arrival_id < num_col){
                    sum += (vector[syn_arrival_id] * (curand_uniform(&state) * w_range + w_min));
                    syn_arrival_id += (int) ceil(log(curand_uniform(&state)) / conn_prob);
                }

                // write
                out[row_i] = sum;
            }
        }


        template<typename T>
        inline void matvec_jitconn_prob_uniform_v2(cudaStream_t stream,
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
            const int grid_dim = (n_row + block_dim - 1) / block_dim;
            cudaMemset(y, 0, sizeof(T) * n_row);
            _jitconn_prob_uniform_v2<T, block_dim><<<grid_dim, block_dim, 0, stream>>>(
                    vector, conn_seed, conn_prob, w_min, w_range, n_row, n_col, y
            );
            ThrowIfError(cudaGetLastError());
        }


        template<typename T, const int BLOCK_SIZE>
        __global__ void _jitconn_prob_normal(
                const T *vector,  /* vector */
                const unsigned int conn_seed,  /* matrix */
                const float conn_prob,
                const float w_mu,
                const float w_sigma,
                const unsigned int num_row,  /* shape */
                const unsigned int num_col,
                T *out  /* output */
        ) {
            __shared__ T shVector[BLOCK_SIZE];

            const unsigned int idx = threadIdx.x;
            const unsigned int row_i = blockIdx.x * blockDim.x + idx;
            const unsigned int num_block = num_col / BLOCK_SIZE;
            const unsigned int num_other = num_col % BLOCK_SIZE;

            // random state
            curandState state;
            curand_init(conn_seed + row_i, 0, 0, &state);

            // summation
            T sum = 0;
            for (int i = 0; i < num_block; i++) {
                shVector[idx] = vector[i * BLOCK_SIZE + idx];
                __syncthreads();
#pragma unroll
                for (int sh_i = 0; sh_i < BLOCK_SIZE; sh_i++) {
                    if (curand_uniform(&state) < conn_prob) {
                        sum += shVector[sh_i] * (curand_normal(&state) * w_sigma + w_mu);
                    }
                }
            }
            if (idx < num_other) {
                shVector[idx] = vector[num_block * BLOCK_SIZE + idx];
            }
            __syncthreads();
            for (int sh_i = 0; sh_i < num_other; sh_i++) {
                if (curand_uniform(&state) < conn_prob)
                    sum += shVector[sh_i] * (curand_normal(&state) * w_sigma + w_mu);
            }

            // write
            if (row_i < num_row) {
                out[row_i] = sum;
            }
        }

        template<typename T>
        inline void matvec_jitconn_prob_normal(cudaStream_t stream,
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
            const int grid_dim = (n_row + block_dim - 1) / block_dim;
            cudaMemset(y, 0, sizeof(T) * n_row);
            _jitconn_prob_normal<T, block_dim><<<grid_dim, block_dim, 0, stream>>>(
                    vector, conn_seed, conn_prob, w_mu, w_sigma, n_row, n_col, y
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

            const unsigned int row_i = blockIdx.x * blockDim.x + threadIdx.x;

            if (row_i < num_row) {
                // random state
                curandState state;
                curand_init(conn_seed + row_i, 0, 0, &state);

                // summation
                T sum = 0;
                int syn_arrival_id = (int) ceil(log(curand_uniform(&state)) / log_prob);
                while (syn_arrival_id < num_col){
                    sum += (vector[syn_arrival_id] * (curand_normal(&state) * w_sigma + w_mu));
                    syn_arrival_id += (int) ceil(log(curand_uniform(&state)) / log_prob);
                }

                // write
                out[row_i] = sum;
            }
        }


        template<typename T>
        inline void matvec_jitconn_prob_normal_v2(cudaStream_t stream,
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
            const int grid_dim = (n_row + block_dim - 1) / block_dim;
            cudaMemset(y, 0, sizeof(T) * n_row);
            _jitconn_prob_normal_v2<T, block_dim><<<grid_dim, block_dim, 0, stream>>>(
                    vector, conn_seed, conn_prob, w_mu, w_sigma, n_row, n_col, y
            );
            ThrowIfError(cudaGetLastError());
        }


    }

    void matvec_jitconn_prob_homo_float(cudaStream_t stream, void **buffers,
                                        const char *opaque, std::size_t opaque_len) {
        matvec_jitconn_prob_homo<float>(stream, buffers, opaque, opaque_len);
    }

    void matvec_jitconn_prob_homo_double(cudaStream_t stream, void **buffers,
                                         const char *opaque, std::size_t opaque_len) {
        matvec_jitconn_prob_homo<double>(stream, buffers, opaque, opaque_len);
    }

    void matvec_jitconn_prob_uniform_float(cudaStream_t stream, void **buffers,
                                           const char *opaque, std::size_t opaque_len) {
        matvec_jitconn_prob_uniform<float>(stream, buffers, opaque, opaque_len);
    }

    void matvec_jitconn_prob_uniform_double(cudaStream_t stream, void **buffers,
                                            const char *opaque, std::size_t opaque_len) {
        matvec_jitconn_prob_uniform<double>(stream, buffers, opaque, opaque_len);
    }

    void matvec_jitconn_prob_normal_float(cudaStream_t stream, void **buffers,
                                          const char *opaque, std::size_t opaque_len) {
        matvec_jitconn_prob_normal<float>(stream, buffers, opaque, opaque_len);
    }

    void matvec_jitconn_prob_normal_double(cudaStream_t stream, void **buffers,
                                           const char *opaque, std::size_t opaque_len) {
        matvec_jitconn_prob_normal<double>(stream, buffers, opaque, opaque_len);
    }


    void matvec_jitconn_prob_homo_v2_float(cudaStream_t stream, void **buffers,
                                           const char *opaque, std::size_t opaque_len) {
        matvec_jitconn_prob_homo_v2<float>(stream, buffers, opaque, opaque_len);
    }

    void matvec_jitconn_prob_homo_v2_double(cudaStream_t stream, void **buffers,
                                            const char *opaque, std::size_t opaque_len) {
        matvec_jitconn_prob_homo_v2<double>(stream, buffers, opaque, opaque_len);
    }

    void matvec_jitconn_prob_uniform_v2_float(cudaStream_t stream, void **buffers,
                                              const char *opaque, std::size_t opaque_len) {
        matvec_jitconn_prob_uniform_v2<float>(stream, buffers, opaque, opaque_len);
    }

    void matvec_jitconn_prob_uniform_v2_double(cudaStream_t stream, void **buffers,
                                               const char *opaque, std::size_t opaque_len) {
        matvec_jitconn_prob_uniform_v2<double>(stream, buffers, opaque, opaque_len);
    }

    void matvec_jitconn_prob_normal_v2_float(cudaStream_t stream, void **buffers,
                                             const char *opaque, std::size_t opaque_len) {
        matvec_jitconn_prob_normal_v2<float>(stream, buffers, opaque, opaque_len);
    }

    void matvec_jitconn_prob_normal_v2_double(cudaStream_t stream, void **buffers,
                                              const char *opaque, std::size_t opaque_len) {
        matvec_jitconn_prob_normal_v2<double>(stream, buffers, opaque, opaque_len);
    }


}

