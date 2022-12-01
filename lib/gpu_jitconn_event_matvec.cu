//
// Created by adadu on 2022/11/30.
//

#include "gpu_jitconn_event_matvec.cuh"

namespace brainpy_lib {
    namespace {

        template<typename T, const int BLOCK_SIZE>
        __global__ void _event_jitconn_prob_homo(
                const int *event_ids,  /* event */
                const int &event_num,
                const unsigned int conn_seed,  /* matrix */
                const float conn_prob,
                const unsigned int num_row,  /* shape */
                const unsigned int num_col,
                T *out  /* output */
        ) {
            __shared__ int shEvents[BLOCK_SIZE];

            const unsigned int idx = threadIdx.x;
            const unsigned int row_i = blockIdx.x * blockDim.x + idx;
            const unsigned int num_block = event_num / BLOCK_SIZE;
            const unsigned int num_other = event_num % BLOCK_SIZE;
            const unsigned int seed = conn_seed + num_col * row_i;

            T sum = 0;
            for (int i = 0; i < num_block; i++) {
                shEvents[idx] = event_ids[i * BLOCK_SIZE + idx];
                __syncthreads();
#pragma unroll
                for (int sh_i = 0; sh_i < BLOCK_SIZE; sh_i++) {
                    if (lfsr113_double(shEvents[sh_i] + seed) < conn_prob)
                        sum += 1;
                }
            }
            if (idx < num_other) {
                shEvents[idx] = event_ids[num_block * BLOCK_SIZE + idx];
            }
            __syncthreads();
            for (int sh_i = 0; sh_i < num_other; sh_i++) {
                if (lfsr113_double(shEvents[sh_i] + seed) < conn_prob)
                    sum += 1;
            }

            // write
            if (row_i < num_row) {
                out[row_i] = sum;
            }
        }

        template<typename T>
        inline void event_jitconn_prob_homo(cudaStream_t stream,
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
            const int *event_ids = reinterpret_cast<const int *>(buffers[0]);
            const int *event_num = reinterpret_cast<const int *>(buffers[1]);
            T *y = reinterpret_cast<T *>(buffers[2]);

            // processing
            const int block_dim = 256;
            const int grid_dim = (n_col + block_dim - 1) / block_dim;
            _event_jitconn_prob_homo<T, block_dim><<<grid_dim, block_dim, 0, stream>>>(
                    event_ids, event_num[0], conn_seed, conn_prob, n_row, n_col, y
            );
            ThrowIfError(cudaGetLastError());
        }


        template<typename T1, typename T2, const int BLOCK_SIZE>
        __global__ void _event_jitconn_prob_homo_v2(
                const T2 *events,  /* event */
                const unsigned int conn_seed,  /* matrix */
                const float conn_prob,
                const unsigned int num_row,  /* shape */
                const unsigned int num_col,
                T1 *out  /* output */
        ) {
            const unsigned int row_i = blockIdx.x * blockDim.x + threadIdx.x;

            if (row_i < num_row) {
                // random state
                curandState state;
                curand_init(conn_seed + row_i, 0, 0, &state);

                // computing
                T1 sum = 0;
                int syn_arrival_id = (int) (log(curand_uniform(&state)) / conn_prob);
                while (syn_arrival_id < num_col) {
                    sum += events[syn_arrival_id];
                    syn_arrival_id += (int) (log(curand_uniform(&state)) / conn_prob);
                }

                // write
                out[row_i] = sum;
            }
        }


        template<typename T1, typename T2>
        inline void event_jitconn_prob_homo_v2(cudaStream_t stream,
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
            const T2 *events = reinterpret_cast<const T2 *>(buffers[0]);
            T1 *y = reinterpret_cast<T1 *>(buffers[1]);

            // processing
            const int block_dim = 256;
            const int grid_dim = (n_col + block_dim - 1) / block_dim;
            _event_jitconn_prob_homo_v2<T1, T2, block_dim><<<grid_dim, block_dim, 0, stream>>>(
                    events, conn_seed, conn_prob, n_row, n_col, y
            );
            ThrowIfError(cudaGetLastError());
        }


        template<typename T, const int BLOCK_SIZE>
        __global__ void _event_jitconn_prob_uniform(
                const int *event_ids,  /* event */
                const int &event_num,
                const unsigned int conn_seed,  /* matrix */
                const float conn_prob,
                const float w_min,
                const float w_range,
                const unsigned int num_row,  /* shape */
                const unsigned int num_col,
                T *out  /* output */
        ) {
            __shared__ int shEvents[BLOCK_SIZE];

            const unsigned int idx = threadIdx.x;
            const unsigned int row_i = blockIdx.x * blockDim.x + idx;
            const unsigned int num_block = event_num / BLOCK_SIZE;
            const unsigned int num_other = event_num % BLOCK_SIZE;
            unsigned int seed;

            T sum = 0;
            for (int i = 0; i < num_block; i++) {
                shEvents[idx] = event_ids[i * BLOCK_SIZE + idx];
                __syncthreads();
#pragma unroll
                for (int sh_i = 0; sh_i < BLOCK_SIZE; sh_i++) {
                    seed = conn_seed + num_col * row_i + shEvents[sh_i];
                    if (lfsr113_double(seed) < conn_prob)
                        sum += (taus88_double(seed) * w_range + w_min);
                }
            }
            if (idx < num_other) {
                shEvents[idx] = event_ids[num_block * BLOCK_SIZE + idx];
            }
            __syncthreads();
            for (int sh_i = 0; sh_i < num_other; sh_i++) {
                seed = conn_seed + num_col * row_i + shEvents[sh_i];
                if (lfsr113_double(seed) < conn_prob)
                    sum += (taus88_double(seed) * w_range + w_min);
            }

            // write
            if (row_i < num_row) {
                out[row_i] = sum;
            }
        }


        template<typename T>
        inline void event_jitconn_prob_uniform(cudaStream_t stream,
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
            const int *event_ids = reinterpret_cast<const int *>(buffers[0]);
            const int *event_num = reinterpret_cast<const int *>(buffers[1]);
            T *y = reinterpret_cast<T *>(buffers[2]);

            // processing
            const int block_dim = 256;
            const int grid_dim = (n_col + block_dim - 1) / block_dim;
            _event_jitconn_prob_uniform<T, block_dim><<<grid_dim, block_dim, 0, stream>>>(
                    event_ids, event_num[0], conn_seed, conn_prob, w_min, w_range, n_row, n_col, y
            );
            ThrowIfError(cudaGetLastError());
        }


        template<typename T1, typename T2, const int BLOCK_SIZE>
        __global__ void _event_jitconn_prob_uniform_v2(
                const T2 *events,  /* event */
                const unsigned int conn_seed,  /* matrix */
                const float conn_prob,
                const float w_min,
                const float w_range,
                const unsigned int num_row,  /* shape */
                const unsigned int num_col,
                T1 *out  /* output */
        ) {
            const unsigned int row_i = blockIdx.x * blockDim.x + threadIdx.x;

            if (row_i < num_row) {
                // random state
                curandState state;
                curand_init(conn_seed + row_i, 0, 0, &state);

                // computing
                T1 sum = 0;
                int syn_arrival_id = (int) (log(curand_uniform(&state)) / conn_prob);
                float rand = curand_uniform(&state) * w_range + w_min;
                while (syn_arrival_id < num_col) {
                    if (events[syn_arrival_id])
                        sum += rand;
                    syn_arrival_id += (int) ceil(log(curand_uniform(&state)) / conn_prob);
                    rand = curand_uniform(&state) * w_range + w_min;
                }

                // write
                out[row_i] = sum;
            }
        }


        template<typename T1, typename T2>
        inline void event_jitconn_prob_uniform_v2(cudaStream_t stream,
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
            const T2 *events = reinterpret_cast<const T2 *>(buffers[0]);
            T1 *y = reinterpret_cast<T1 *>(buffers[1]);

            // processing
            const int block_dim = 256;
            const int grid_dim = (n_col + block_dim - 1) / block_dim;
            _event_jitconn_prob_uniform_v2<T1, T2, block_dim><<<grid_dim, block_dim, 0, stream>>>(
                    events, conn_seed, conn_prob, w_min, w_range, n_row, n_col, y
            );
            ThrowIfError(cudaGetLastError());
        }


        template<typename T, const int BLOCK_SIZE>
        __global__ void _event_jitconn_prob_normal(
                const int *event_ids,  /* event */
                const int &event_num,
                const unsigned int conn_seed,  /* matrix */
                const float conn_prob,
                const float w_mu,
                const float w_sigma,
                const unsigned int num_row,  /* shape */
                const unsigned int num_col,
                T *out  /* output */
        ) {
            __shared__ int shEvents[BLOCK_SIZE];

            const unsigned int idx = threadIdx.x;
            const unsigned int row_i = blockIdx.x * blockDim.x + idx;
            const unsigned int num_block = event_num / BLOCK_SIZE;
            const unsigned int num_other = event_num % BLOCK_SIZE;
            unsigned int seed_base = conn_seed + num_col * row_i, seed;
            double u, v;

            T sum = 0;
            for (int i = 0; i < num_block; i++) {
                shEvents[idx] = event_ids[i * BLOCK_SIZE + idx];
                __syncthreads();
#pragma unroll
                for (int sh_i = 0; sh_i < BLOCK_SIZE; sh_i++) {
                    seed = seed_base + shEvents[sh_i];
                    if (lfsr113_double(seed) < conn_prob) {
                        taus88_double2(seed, &u, &v);
                        sum += (w_mu + sqrt(-2 * log(u)) * cos(2 * M_PI * v) * w_sigma);
                    }
                }
            }
            if (idx < num_other) {
                shEvents[idx] = event_ids[num_block * BLOCK_SIZE + idx];
            }
            __syncthreads();
            for (int sh_i = 0; sh_i < num_other; sh_i++) {
                seed = seed_base + shEvents[sh_i];
                if (lfsr113_double(seed) < conn_prob) {
                    taus88_double2(seed, &u, &v);
                    sum += (w_mu + sqrt(-2 * log(u)) * cos(2 * M_PI * v) * w_sigma);
                }
            }

            // write
            if (row_i < num_row) {
                out[row_i] = sum;
            }
        }


        template<typename T>
        inline void event_jitconn_prob_normal(cudaStream_t stream,
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
            const int *event_ids = reinterpret_cast<const int *>(buffers[0]);
            const int *event_num = reinterpret_cast<const int *>(buffers[1]);
            T *y = reinterpret_cast<T *>(buffers[2]);

            // processing
            const int block_dim = 256;
            const int grid_dim = (n_col + block_dim - 1) / block_dim;
            _event_jitconn_prob_normal<T, block_dim><<<grid_dim, block_dim, 0, stream>>>(
                    event_ids, event_num[0], conn_seed, conn_prob, w_mu, w_sigma, n_row, n_col, y
            );
            ThrowIfError(cudaGetLastError());
        }


        template<typename T1, typename T2, const int BLOCK_SIZE>
        __global__ void _event_jitconn_prob_normal_v2(
                const T2 *events,  /* event */
                const unsigned int conn_seed,  /* matrix */
                const float conn_prob,
                const float w_mu,
                const float w_sigma,
                const unsigned int num_row,  /* shape */
                const unsigned int num_col,
                T1 *out  /* output */
        ) {
            const unsigned int row_i = blockIdx.x * blockDim.x + threadIdx.x;

            if (row_i < num_row) {
                // random state
                curandState state;
                curand_init(conn_seed + row_i, 0, 0, &state);

                // computing
                T1 sum = 0;
                int syn_arrival_id = (int) ceil(log(curand_uniform(&state)) / conn_prob);
                float rand = curand_normal(&state) * w_sigma + w_mu;
                while (syn_arrival_id < num_col) {
                    if (events[i])
                        sum += rand;
                    syn_arrival_id += (int) ceil(log(curand_uniform(&state)) / conn_prob);
                    rand = curand_normal(&state) * w_sigma + w_mu;
                }

                // write
                out[row_i] = sum;
            }
        }


        template<typename T1, typename T2>
        inline void event_jitconn_prob_normal_v2(cudaStream_t stream,
                                                 void **buffers,
                                                 const char *opaque,
                                                 std::size_t opaque_len) {
            // size
            const JITConnProbNormalDescriptor &d = *UnpackDescriptor<JITConnProbNormalDescriptor>(opaque,
                                                                                                  opaque_len);
            const unsigned int n_row = d.n_row;
            const unsigned int n_col = d.n_col;
            const unsigned int conn_seed = d.seed;
            const float log_p = d.prob;
            const float w_mu = d.w_mu;
            const float w_sigma = d.w_sigma;

            // data
            const T2 *events = reinterpret_cast<const T2 *>(buffers[0]);
            T1 *y = reinterpret_cast<T1 *>(buffers[1]);

            // processing
            const int block_dim = 256;
            const int grid_dim = (n_col + block_dim - 1) / block_dim;
            _event_jitconn_prob_normal_v2<T1, T2, block_dim><<<grid_dim, block_dim, 0, stream>>>(
                    events, conn_seed, log_p, w_mu, w_sigma, n_row, n_col, y
            );
            ThrowIfError(cudaGetLastError());
        }


    }

    /* version 1 API */
    void event_matvec_jitconn_prob_homo_float(cudaStream_t stream, void **buffers,
                                              const char *opaque, std::size_t opaque_len) {
        event_jitconn_prob_homo<float>(stream, buffers, opaque, opaque_len);
    }

    void event_matvec_jitconn_prob_homo_double(cudaStream_t stream, void **buffers,
                                               const char *opaque, std::size_t opaque_len) {
        event_jitconn_prob_homo<double>(stream, buffers, opaque, opaque_len);
    }

    void event_matvec_jitconn_prob_uniform_float(cudaStream_t stream, void **buffers,
                                                 const char *opaque, std::size_t opaque_len) {
        event_jitconn_prob_uniform<float>(stream, buffers, opaque, opaque_len);
    }

    void event_matvec_jitconn_prob_uniform_double(cudaStream_t stream, void **buffers,
                                                  const char *opaque, std::size_t opaque_len) {
        event_jitconn_prob_uniform<double>(stream, buffers, opaque, opaque_len);
    }

    void event_matvec_jitconn_prob_normal_float(cudaStream_t stream, void **buffers,
                                                const char *opaque, std::size_t opaque_len) {
        event_jitconn_prob_normal<float>(stream, buffers, opaque, opaque_len);
    }

    void event_matvec_jitconn_prob_normal_double(cudaStream_t stream, void **buffers,
                                                 const char *opaque, std::size_t opaque_len) {
        event_jitconn_prob_normal<double>(stream, buffers, opaque, opaque_len);
    }

    /* version 2 API */

    void event_matvec_jitconn_prob_homo_v2_float_bool(cudaStream_t stream, void **buffers,
                                                      const char *opaque, std::size_t opaque_len) {
        event_jitconn_prob_homo_v2<float, bool>(stream, buffers, opaque, opaque_len);
    }

    void event_matvec_jitconn_prob_homo_v2_double_bool(cudaStream_t stream, void **buffers,
                                                       const char *opaque, std::size_t opaque_len) {
        event_jitconn_prob_homo_v2<double, bool>(stream, buffers, opaque, opaque_len);
    }

    void event_matvec_jitconn_prob_homo_v2_float_float(cudaStream_t stream, void **buffers,
                                                       const char *opaque, std::size_t opaque_len) {
        event_jitconn_prob_homo_v2<float, float>(stream, buffers, opaque, opaque_len);
    }

    void event_matvec_jitconn_prob_homo_v2_double_double(cudaStream_t stream, void **buffers,
                                                         const char *opaque, std::size_t opaque_len) {
        event_jitconn_prob_homo_v2<double, double>(stream, buffers, opaque, opaque_len);
    }

    void event_matvec_jitconn_prob_uniform_v2_float_bool(cudaStream_t stream, void **buffers,
                                                         const char *opaque, std::size_t opaque_len) {
        event_jitconn_prob_uniform_v2<float, bool>(stream, buffers, opaque, opaque_len);
    }

    void event_matvec_jitconn_prob_uniform_v2_double_bool(cudaStream_t stream, void **buffers,
                                                          const char *opaque, std::size_t opaque_len) {
        event_jitconn_prob_uniform_v2<double, bool>(stream, buffers, opaque, opaque_len);
    }

    void event_matvec_jitconn_prob_uniform_v2_float_float(cudaStream_t stream, void **buffers,
                                                          const char *opaque, std::size_t opaque_len) {
        event_jitconn_prob_uniform_v2<float, float>(stream, buffers, opaque, opaque_len);
    }

    void event_matvec_jitconn_prob_uniform_v2_double_double(cudaStream_t stream, void **buffers,
                                                            const char *opaque, std::size_t opaque_len) {
        event_jitconn_prob_uniform_v2<double, double>(stream, buffers, opaque, opaque_len);
    }

    void event_matvec_jitconn_prob_normal_v2_float_bool(cudaStream_t stream, void **buffers,
                                                        const char *opaque, std::size_t opaque_len) {
        event_jitconn_prob_normal_v2<float, bool>(stream, buffers, opaque, opaque_len);
    }

    void event_matvec_jitconn_prob_normal_v2_double_bool(cudaStream_t stream, void **buffers,
                                                         const char *opaque, std::size_t opaque_len) {
        event_jitconn_prob_normal_v2<double, bool>(stream, buffers, opaque, opaque_len);
    }

    void event_matvec_jitconn_prob_normal_v2_float_float(cudaStream_t stream, void **buffers,
                                                         const char *opaque, std::size_t opaque_len) {
        event_jitconn_prob_normal_v2<float, float>(stream, buffers, opaque, opaque_len);
    }

    void event_matvec_jitconn_prob_normal_v2_double_double(cudaStream_t stream, void **buffers,
                                                           const char *opaque, std::size_t opaque_len) {
        event_jitconn_prob_normal_v2<double, double>(stream, buffers, opaque, opaque_len);
    }


}

