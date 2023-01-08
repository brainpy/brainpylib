//
// Created by adadu on 2023/1/8.
//

#include "gpu_jitconn_matmat.cuh"

namespace brainpy_lib {
    namespace {


        template<class T>
        __device__ T warp_reduce(T val) {
            for (int offset = warpSize / 2; offset > 0; offset /= 2)
                val += __shfl_down_sync(FULL_WARP_MASK, val, offset);
            return val;
        }


        template<typename T, const int BLOCK_SIZE>
        __global__ void _prob_normal(
                const T *X,  /* matrix X */
                const unsigned int seed,  /* matrix M */
                const double log_prob,
                const double w_mu,
                const double w_sigma,
                const unsigned int m,  /* shape */
                const unsigned int k,
                const unsigned int n,
                T *Y  /* output */
        ) {
            __shared__ T Ms_val[BLOCK_SIZE];
            __shared__ int Ms_ind[BLOCK_SIZE];
            const unsigned int thread_i = threadIdx.x;
            const unsigned int m_i = blockIdx.y * BLOCK_SIZE + thread_i;
            int syn_arrival_id = 0;
            T sum = 0;
            curandState state;
            curand_init(seed + blockIdx.x, m_i * n, 0, &state);

            while (syn_arrival_id < k) {
                Ms_ind[thread_i] = (int) ceil(log(curand_uniform(&state)) / log_prob);
                Ms_val[thread_i] = (T) (curand_normal(&state) * w_sigma + w_mu);
                __syncthreads();
                if (m_i < m) {
#pragma unroll
                    for (int i = 0; i < BLOCK_SIZE; ++i) {
                        syn_arrival_id += Ms_ind[i];
                        if (syn_arrival_id < k) {
                            sum += (X[m_i * k + syn_arrival_id] * Ms_val[i]);
                        } else {
                            break;
                        }
                    }
                }
            }
            if (m_i < m) {
                Y[m_i * n + blockIdx.x] = sum;
            }
        }


        template<typename T>
        inline void matmat_prob_normal(
                cudaStream_t stream,
                void **buffers,
                const char *opaque,
                std::size_t opaque_len
        ) {
            // size
            const MatMatJITProbDescriptor1 &d =
                    *UnpackDescriptor<MatMatJITProbDescriptor1>(opaque, opaque_len);
            const unsigned int m = d.m;
            const unsigned int k = d.k;
            const unsigned int n = d.n;
            const unsigned int seed = d.seed;
            const double log_p = d.log_p;
            const double w_mu = d.w1;
            const double w_sigma = d.w2;

            // data
            const T *X = reinterpret_cast<const T *>(buffers[0]);
            T *Y = reinterpret_cast<T *>(buffers[1]);

            // processing
            if (m <= 32) {
                dim3 grid(n, (m + 31) / 32);
                _prob_normal<T, 32><<<grid, 32, 0, stream>>>(X, seed, log_p, w_mu, w_sigma, m, k, n, Y);
            } else if (m <= 64) {
                dim3 grid(n, (m + 63) / 64);
                _prob_normal<T, 64><<<grid, 64, 0, stream>>>(X, seed, log_p, w_mu, w_sigma, m, k, n, Y);
            } else if (m <= 96) {
                dim3 grid(n, (m + 95) / 96);
                _prob_normal<T, 96><<<grid, 96, 0, stream>>>(X, seed, log_p, w_mu, w_sigma, m, k, n, Y);
            } else if (m <= 128) {
                dim3 grid(n, (m + 127) / 128);
                _prob_normal<T, 128><<<grid, 128, 0, stream>>>(X, seed, log_p, w_mu, w_sigma, m, k, n, Y);
            } else if (m <= 160) {
                dim3 grid(n, (m + 159) / 160);
                _prob_normal<T, 160><<<grid, 160, 0, stream>>>(X, seed, log_p, w_mu, w_sigma, m, k, n, Y);
            } else if (m <= 192) {
                dim3 grid(n, (m + 191) / 192);
                _prob_normal<T, 192><<<grid, 192, 0, stream>>>(X, seed, log_p, w_mu, w_sigma, m, k, n, Y);
            } else if (m <= 224) {
                dim3 grid(n, (m + 223) / 224);
                _prob_normal<T, 224><<<grid, 224, 0, stream>>>(X, seed, log_p, w_mu, w_sigma, m, k, n, Y);
            } else if (m <= 256) {
                dim3 grid(n, (m + 255) / 256);
                _prob_normal<T, 256><<<grid, 256, 0, stream>>>(X, seed, log_p, w_mu, w_sigma, m, k, n, Y);
            } else if (m <= 288) {
                dim3 grid(n, (m + 287) / 288);
                _prob_normal<T, 288><<<grid, 288, 0, stream>>>(X, seed, log_p, w_mu, w_sigma, m, k, n, Y);
            } else if (m <= 320) {
                dim3 grid(n, (m + 319) / 320);
                _prob_normal<T, 320><<<grid, 320, 0, stream>>>(X, seed, log_p, w_mu, w_sigma, m, k, n, Y);
            } else if (m <= 352) {
                dim3 grid(n, (m + 351) / 352);
                _prob_normal<T, 352><<<grid, 352, 0, stream>>>(X, seed, log_p, w_mu, w_sigma, m, k, n, Y);
            } else if (m <= 384) {
                dim3 grid(n, (m + 383) / 384);
                _prob_normal<T, 384><<<grid, 384, 0, stream>>>(X, seed, log_p, w_mu, w_sigma, m, k, n, Y);
            } else if (m <= 416) {
                dim3 grid(n, (m + 415) / 416);
                _prob_normal<T, 416><<<grid, 416, 0, stream>>>(X, seed, log_p, w_mu, w_sigma, m, k, n, Y);
            } else if (m <= 448) {
                dim3 grid(n, (m + 447) / 448);
                _prob_normal<T, 448><<<grid, 448, 0, stream>>>(X, seed, log_p, w_mu, w_sigma, m, k, n, Y);
            } else if (m <= 480) {
                dim3 grid(n, (m + 479) / 480);
                _prob_normal<T, 480><<<grid, 480, 0, stream>>>(X, seed, log_p, w_mu, w_sigma, m, k, n, Y);
            } else {
                dim3 grid(n, (m + 511) / 512);
                _prob_normal<T, 512><<<grid, 512, 0, stream>>>(X, seed, log_p, w_mu, w_sigma, m, k, n, Y);
            }
            ThrowIfError(cudaGetLastError());
        }


        template<typename T, const int SIZE>
        __global__ void _prob_normal_v2(
                const T *X,  /* matrix X */
                const unsigned int seed,  /* matrix M */
                const double log_prob,
                const double w_mu,
                const double w_sigma,
                const unsigned int m,  /* shape */
                const unsigned int k,
                const unsigned int n,
                const unsigned int k_per_thread,
                T *Y  /* output */
        ) {
            const unsigned int total_thread = blockIdx.x * blockDim.x / 32 + threadIdx.x / 32;
            const unsigned int m_i = total_thread / n * SIZE;
            const unsigned int n_i = total_thread % n;
            const unsigned int thread_i = threadIdx.x % 32;
            unsigned int syn_arrival_id = k_per_thread * thread_i;
            const unsigned int k_max = min(syn_arrival_id + k_per_thread, k);

            curandState state;
            curand_init(seed + n_i, k_per_thread * thread_i, 0, &state);
            syn_arrival_id += (unsigned int) ceil(log(curand_uniform(&state)) / log_prob);
            T rand = 0;
            T sum[SIZE];
#pragma unroll
            for (int i = 0; i < SIZE; ++i) {
                sum[i] = 0;
            }

            if (n_i < n) {
                while (syn_arrival_id < k_max) {
                    rand = (T) (curand_normal(&state) * w_sigma + w_mu);
#pragma unroll
                    for (int i = 0; i < SIZE; ++i) {
                        sum[i] += (X[(m_i + i) * k + syn_arrival_id] * rand);
                    }
                    syn_arrival_id += (unsigned int) ceil(log(curand_uniform(&state)) / log_prob);
                }
#pragma unroll
                for (int i = 0; i < SIZE; ++i) {
                    sum[i] = warp_reduce(sum[i]);
                    if (m_i + i < m && thread_i == 0) {
                        Y[(m_i + i) * n + n_i] = sum[i];
                    }
                }
            }
        }


        template<typename T>
        inline void matmat_prob_normal_v2(
                cudaStream_t stream,
                void **buffers,
                const char *opaque,
                std::size_t opaque_len
        ) {
            // size
            const MatMatJITProbDescriptor1 &d =
                    *UnpackDescriptor<MatMatJITProbDescriptor1>(opaque, opaque_len);
            const unsigned int m = d.m;
            const unsigned int k = d.k;
            const unsigned int n = d.n;
            const unsigned int seed = d.seed;
            const double log_p = d.log_p;
            const double w_mu = d.w1;
            const double w_sigma = d.w2;
            const unsigned int k_per_thread = (k + 31) / 32;

            // data
            const T *X = reinterpret_cast<const T *>(buffers[0]);
            T *Y = reinterpret_cast<T *>(buffers[1]);

            // processing
            const int block_dim = 512;
            if (m <= 8) {
                const int grid_dim = ((m + 7) / 8 * n * 32 + block_dim - 1) / block_dim;
                _prob_normal_v2<T, 8><<<grid_dim, block_dim, 0, stream>>>(
                        X, seed, log_p, w_mu, w_sigma, m, k, n, k_per_thread, Y);
            } else if (m <= 16) {
                const int grid_dim = ((m + 15) / 16 * n * 32 + block_dim - 1) / block_dim;
                _prob_normal_v2<T, 16><<<grid_dim, block_dim, 0, stream>>>(
                        X, seed, log_p, w_mu, w_sigma, m, k, n, k_per_thread, Y);
            } else if (m <= 24) {
                const int grid_dim = ((m + 23) / 24 * n * 32 + block_dim - 1) / block_dim;
                _prob_normal_v2<T, 24><<<grid_dim, block_dim, 0, stream>>>(
                        X, seed, log_p, w_mu, w_sigma, m, k, n, k_per_thread, Y);
            } else {
                const int grid_dim = ((m + 31) / 32 * n * 32 + block_dim - 1) / block_dim;
                _prob_normal_v2<T, 32><<<grid_dim, block_dim, 0, stream>>>(
                        X, seed, log_p, w_mu, w_sigma, m, k, n, k_per_thread, Y);
            }
            ThrowIfError(cudaGetLastError());
        }

        /*
         * Each thread computes SIZE rows x 1 col of Y
         */
        template<typename T, const int SIZE>
        __global__ void _prob_normal_v1(
                const T *X,  /* matrix X */
                const unsigned int seed,  /* matrix M */
                const double log_prob,
                const double w_mu,
                const double w_sigma,
                const unsigned int m,  /* shape */
                const unsigned int k,
                const unsigned int n,
                T *Y  /* output */
        ) {
            const unsigned int m_i = (blockIdx.x * blockDim.x + threadIdx.x) / n * SIZE;
            const unsigned int n_i = (blockIdx.x * blockDim.x + threadIdx.x) % n;
            curandState state;
            curand_init(seed + n_i, 0, 0, &state);
            unsigned int syn_arrival_id = (unsigned int) ceil(log(curand_uniform(&state)) / log_prob);
            T rand = 0;
            T sum[SIZE];
#pragma unroll
            for (int i = 0; i < SIZE; ++i) {
                sum[i] = 0;
            }

            if (n_i < n) {
                while (syn_arrival_id < k) {
                    rand = (T) (curand_normal(&state) * w_sigma + w_mu);
#pragma unroll
                    for (int i = 0; i < SIZE; ++i) {
                        sum[i] += (X[(m_i + i) * k + syn_arrival_id] * rand);
                    }
                    syn_arrival_id += (unsigned int) ceil(log(curand_uniform(&state)) / log_prob);
                }
#pragma unroll
                for (int i = 0; i < SIZE; ++i) {
                    if (m_i + i < m) {
                        Y[(m_i + i) * n + n_i] = sum[i];
                    }
                }
            }
        }


        template<typename T>
        inline void matmat_prob_normal_v1(
                cudaStream_t stream,
                void **buffers,
                const char *opaque,
                std::size_t opaque_len
        ) {
            // size
            const MatMatJITProbDescriptor1 &d =
                    *UnpackDescriptor<MatMatJITProbDescriptor1>(opaque, opaque_len);
            const unsigned int m = d.m;
            const unsigned int k = d.k;
            const unsigned int n = d.n;
            const unsigned int seed = d.seed;
            const double log_p = d.log_p;
            const double w_mu = d.w1;
            const double w_sigma = d.w2;

            // data
            const T *X = reinterpret_cast<const T *>(buffers[0]);
            T *Y = reinterpret_cast<T *>(buffers[1]);

            // processing
            const int block_dim = 256;
            if (m <= 8) {
                const int grid_dim = ((m + 7) / 8 * n + block_dim - 1) / block_dim;
                _prob_normal_v1<T, 8><<<grid_dim, block_dim, 0, stream>>>(X, seed, log_p, w_mu, w_sigma, m, k, n, Y);
            } else if (m <= 16) {
                const int grid_dim = ((m + 15) / 16 * n + block_dim - 1) / block_dim;
                _prob_normal_v1<T, 16><<<grid_dim, block_dim, 0, stream>>>(X, seed, log_p, w_mu, w_sigma, m, k, n, Y);
            } else if (m <= 24) {
                const int grid_dim = ((m + 23) / 24 * n + block_dim - 1) / block_dim;
                _prob_normal_v1<T, 24><<<grid_dim, block_dim, 0, stream>>>(X, seed, log_p, w_mu, w_sigma, m, k, n, Y);
            } else {
                const int grid_dim = ((m + 31) / 32 * n + block_dim - 1) / block_dim;
                _prob_normal_v1<T, 32><<<grid_dim, block_dim, 0, stream>>>(X, seed, log_p, w_mu, w_sigma, m, k, n, Y);
            }
            ThrowIfError(cudaGetLastError());
        }



        template<typename T, const int SIZE>
        __global__ void _prob_uniform_v1(
                const T *X,  /* matrix X */
                const unsigned int seed,  /* matrix M */
                const double log_prob,
                const double w_min,
                const double w_range,
                const unsigned int m,  /* shape */
                const unsigned int k,
                const unsigned int n,
                T *Y  /* output */
        ) {
            const unsigned int m_i = (blockIdx.x * blockDim.x + threadIdx.x) / n * SIZE;
            const unsigned int n_i = (blockIdx.x * blockDim.x + threadIdx.x) % n;
            curandState state;
            curand_init(seed + n_i, 0, 0, &state);
            unsigned int syn_arrival_id = (unsigned int) ceil(log(curand_uniform(&state)) / log_prob);
            T rand = 0;
            T sum[SIZE];
#pragma unroll
            for (int i = 0; i < SIZE; ++i) {
                sum[i] = 0;
            }

            if (n_i < n) {
                while (syn_arrival_id < k) {
                    rand = (T) (curand_uniform(&state) * w_range + w_min);
#pragma unroll
                    for (int i = 0; i < SIZE; ++i) {
                        sum[i] += (X[(m_i + i) * k + syn_arrival_id] * rand);
                    }
                    syn_arrival_id += (unsigned int) ceil(log(curand_uniform(&state)) / log_prob);
                }
#pragma unroll
                for (int i = 0; i < SIZE; ++i) {
                    if (m_i + i < m) {
                        Y[(m_i + i) * n + n_i] = sum[i];
                    }
                }
            }
        }


        template<typename T>
        inline void matmat_prob_uniform_v1(
                cudaStream_t stream,
                void **buffers,
                const char *opaque,
                std::size_t opaque_len
        ) {
            // size
            const MatMatJITProbDescriptor1 &d =
                    *UnpackDescriptor<MatMatJITProbDescriptor1>(opaque, opaque_len);
            const unsigned int m = d.m;
            const unsigned int k = d.k;
            const unsigned int n = d.n;
            const unsigned int seed = d.seed;
            const double log_p = d.log_p;
            const double w_min = d.w1;
            const double w_range = d.w2;

            // data
            const T *X = reinterpret_cast<const T *>(buffers[0]);
            T *Y = reinterpret_cast<T *>(buffers[1]);

            // processing
            const int block_dim = 256;
            if (m <= 8) {
                const int grid_dim = (m * n + block_dim * 8 - 1) / block_dim / 8;
                _prob_uniform_v1<T, 8><<<grid_dim, block_dim, 0, stream>>>(X, seed, log_p, w_min, w_range, m, k, n, Y);
            } else if (m <= 16) {
                const int grid_dim = (m * n + block_dim * 16 - 1) / block_dim / 16;
                _prob_uniform_v1<T, 16><<<grid_dim, block_dim, 0, stream>>>(X, seed, log_p, w_min, w_range, m, k, n, Y);
            } else if (m <= 24) {
                const int grid_dim = (m * n + block_dim * 24 - 1) / block_dim / 24;
                _prob_uniform_v1<T, 24><<<grid_dim, block_dim, 0, stream>>>(X, seed, log_p, w_min, w_range, m, k, n, Y);
            } else {
                const int grid_dim = (m * n + block_dim * 32 - 1) / block_dim / 32;
                _prob_uniform_v1<T, 32><<<grid_dim, block_dim, 0, stream>>>(X, seed, log_p, w_min, w_range, m, k, n, Y);
            }
            ThrowIfError(cudaGetLastError());
        }




    }

    void jitconn_matmat_prob_normal_float_v3(cudaStream_t stream, void **buffers,
                                          const char *opaque, std::size_t opaque_len) {
        matmat_prob_normal<float>(stream, buffers, opaque, opaque_len);
    }

    void jitconn_matmat_prob_normal_double_v3(cudaStream_t stream, void **buffers,
                                           const char *opaque, std::size_t opaque_len) {
        matmat_prob_normal<double>(stream, buffers, opaque, opaque_len);
    }


    void jitconn_matmat_prob_normal_float_v2(cudaStream_t stream, void **buffers,
                                             const char *opaque, std::size_t opaque_len) {
        matmat_prob_normal_v2<float>(stream, buffers, opaque, opaque_len);
    }

    void jitconn_matmat_prob_normal_double_v2(cudaStream_t stream, void **buffers,
                                              const char *opaque, std::size_t opaque_len) {
        matmat_prob_normal_v2<double>(stream, buffers, opaque, opaque_len);
    }


    void jitconn_matmat_prob_normal_float_v1(cudaStream_t stream, void **buffers,
                                             const char *opaque, std::size_t opaque_len) {
        matmat_prob_normal_v1<float>(stream, buffers, opaque, opaque_len);
    }

    void jitconn_matmat_prob_normal_double_v1(cudaStream_t stream, void **buffers,
                                              const char *opaque, std::size_t opaque_len) {
        matmat_prob_normal_v1<double>(stream, buffers, opaque, opaque_len);
    }



    void jitconn_matmat_prob_uniform_float_v1(cudaStream_t stream, void **buffers,
                                             const char *opaque, std::size_t opaque_len) {
        matmat_prob_uniform_v1<float>(stream, buffers, opaque, opaque_len);
    }

    void jitconn_matmat_prob_uniform_double_v1(cudaStream_t stream, void **buffers,
                                              const char *opaque, std::size_t opaque_len) {
        matmat_prob_uniform_v1<double>(stream, buffers, opaque, opaque_len);
    }


}

