// This file contains the GPU implementation of our op. It's a pretty typical CUDA kernel
// and I make no promises about the quality of the code or the choices made therein, but
// it should get the point across.

#include "gpu_event_mmm_gpu.cuh"

namespace brainpy_lib {

    namespace {

        template<const int K_TILE, const int M_TILE>
        __global__ void event_mmm_fixedprob_kernel_v1(
                const bool *V,
                const float *A,
                const uint32_t m,
                const uint32_t n,
                const uint32_t k,
                const std::uint32_t seed,
                const uint32_t A_row_step,
                const float log_p,
                float *O
        ) {
            // shared memory
            __shared__ bool V_smem[K_TILE * M_TILE];
            __shared__ float A_smem[K_TILE * M_TILE];

            // register fragment
            float A_frag[4];
            float O_frag[K_TILE];
#pragma unroll
            for (int i = 0; i < K_TILE; ++i) { O_frag[i] = 0; }

            // V load register
            bool V_ldg_reg;

            // A load register
            float A_ldg_reg;

            // thread position
            const uint32_t n_id = threadIdx.x + blockIdx.x * blockDim.x; // position of n
            const uint32_t y_bid = threadIdx.x / M_TILE;  // y position at this block (for loading A)
            const uint32_t x_bid = threadIdx.x % M_TILE;  // x position at this block (for loading A)

            // synapse state
            curandState state;
            curand_init(seed + n_id, 0, 0, &state);
            int syn_arrival_id = (int) ceil(log(curand_uniform(&state)) / log_p) - 1;

            // index for data loading and writing
            const char *A_ldg_ptr = (const char *) (A + y_bid * m + x_bid);
            const uint32_t A_sts_addr = smem_u32addr(A_smem + x_bid * K_TILE + y_bid);
            const uint32_t A_lds_addr = smem_u32addr(A_smem);
            uint32_t m_id_for_load_V = threadIdx.x;
            uint32_t m_id_for_load_A = x_bid;

            // ldg_guard to avoid LDG out of bound
            bool V_m_guard = m_id_for_load_V < m;
            bool A_m_guard = m_id_for_load_A < m;
            bool A_k_guard = y_bid < k;

            // load first V_tile into shared memory
            V_ldg_reg = V_m_guard ? V[m_id_for_load_V] : false;
            V_smem[threadIdx.x] = V_ldg_reg;

            // load first A_tile to shared memory
            ldg32_nc_0(A_ldg_reg, A_ldg_ptr, A_k_guard && A_m_guard);
            sts32(A_ldg_reg, A_sts_addr);
            __syncthreads();

            // index for data loading
            m_id_for_load_V += M_TILE;
            m_id_for_load_A += M_TILE;
            V_m_guard = m_id_for_load_V < m;
            A_m_guard = m_id_for_load_A < m;
            A_ldg_ptr += M_TILE * sizeof(float);

            /*
             * num_m_tile_loop loop
             */
            const int num_m_tile_loop = (m + M_TILE - 1) / M_TILE - 1;
            const int last_m_tile = k - num_m_tile_loop * M_TILE;

            // number of m tile to read A
            for (int mm = 0; mm < num_m_tile_loop; ++mm) {
#pragma unroll
                for (int m_frag = 0; m_frag < M_TILE; ++m_frag) {

                    if (m_frag == 0) {
                        V_ldg_reg = V_m_guard ? V[m_id_for_load_V] : false;
                        ldg32_nc(A_ldg_reg, A_ldg_ptr, A_k_guard && A_m_guard);
                        m_id_for_load_V += M_TILE;
                        m_id_for_load_A += M_TILE;
                        V_m_guard = m_id_for_load_V < m;
                        A_m_guard = m_id_for_load_A < m;
                        A_ldg_ptr += M_TILE * sizeof(float);
                    }

                    // if spike and synapse id arrived, FFMA
                    if (m_frag == syn_arrival_id) {
                        if (V_smem[m_frag]) {
#pragma unroll
                            for (int i = 0; i < K_TILE; i += 4) {
                                lds128(A_frag[0], A_frag[1], A_frag[2], A_frag[3],
                                       A_lds_addr + (m_frag * K_TILE + i) * sizeof(float));
                                O_frag[i] += A_frag[0];
                                O_frag[i + 1] += A_frag[1];
                                O_frag[i + 2] += A_frag[2];
                                O_frag[i + 3] += A_frag[3];
                            }
                        }
                        syn_arrival_id += (int) ceil(log(curand_uniform(&state)) / log_p);
                    }

                    if (m_frag == M_TILE - 1) {
                        syn_arrival_id -= M_TILE;
                        V_smem[threadIdx.x] = V_ldg_reg;
                        sts32(A_ldg_reg, A_sts_addr);
                        __syncthreads();
                    }
                }
            }

            // FFMA for the last tile
#pragma unroll
            for (int m_frag = 0; m_frag < M_TILE; ++m_frag) {
                if ((m_frag < last_m_tile) && (m_frag == syn_arrival_id)) {
                    if (V_smem[m_frag]) {
#pragma unroll
                        for (int i = 0; i < K_TILE; i += 4) {
                            lds128(A_frag[0], A_frag[1], A_frag[2], A_frag[3],
                                   A_lds_addr + (m_frag * K_TILE + i) * sizeof(float));
                            O_frag[i] += A_frag[0];
                            O_frag[i + 1] += A_frag[1];
                            O_frag[i + 2] += A_frag[2];
                            O_frag[i + 3] += A_frag[3];
                        }
                    }
                    syn_arrival_id += (int) ceil(log(curand_uniform(&state)) / log_p);
                }
            }

            // O_tile write back
            if (n_id < n) {
#pragma unroll
                for (int i = 0; i < K_TILE; ++i) {
                    if (i < k) { O[i * n + n_id] = O_frag[i]; }
                }
            }
        }



        __device__ double taus88_double(unsigned int seed) {
            /* Generates numbers between 0 and 1. */
            unsigned int s1 = seed << 1, s2 = seed << 2, s3 = seed << 3, b;
            b = (((s1 << 13) ^ s1) >> 19);
            s1 = (((s1 & 4294967294) << 12) ^ b);
            b = (((s2 << 2) ^ s2) >> 25);
            s2 = (((s2 & 4294967288) << 4) ^ b);
            b = (((s3 << 3) ^ s3) >> 11);
            s3 = (((s3 & 4294967280) << 17) ^ b);
            return ((s1 ^ s2 ^ s3) * 2.3283064365386963e-10);
        }



        template<const int K_TILE, const int M_TILE>
        __global__ void event_mmm_fixedprob_kernel_v2(
                const bool *V,
                const float *A,
                const uint32_t m,
                const uint32_t n,
                const uint32_t k,
                const std::uint32_t seed,
                const uint32_t A_row_step,
                const float log_p,
                float *O
        ) {
            // shared memory
            __shared__ bool V_smem[K_TILE * M_TILE];
            __shared__ float A_smem[K_TILE * M_TILE];

            // register fragment
            float A_frag[4];
            float O_frag[K_TILE];
#pragma unroll
            for (int i = 0; i < K_TILE; ++i) { O_frag[i] = 0; }

            // V load register
            bool V_ldg_reg;

            // A load register
            float A_ldg_reg;

            // thread position
            const uint32_t n_id = threadIdx.x + blockIdx.x * blockDim.x; // position of n
            const uint32_t y_bid = threadIdx.x / M_TILE;  // y position at this block (for loading A)
            const uint32_t x_bid = threadIdx.x % M_TILE;  // x position at this block (for loading A)

            // index for data loading and writing
            const char *A_ldg_ptr = (const char *) (A + y_bid * m + x_bid);
            const uint32_t A_sts_addr = smem_u32addr(A_smem + x_bid * K_TILE + y_bid);
            const uint32_t A_lds_addr = smem_u32addr(A_smem);
            uint32_t m_id_for_load_V = threadIdx.x;
            uint32_t m_id_for_load_A = x_bid;

            // ldg_guard to avoid LDG out of bound
            bool V_m_guard = m_id_for_load_V < m;
            bool A_m_guard = m_id_for_load_A < m;
            bool A_k_guard = y_bid < k;

            // load first V_tile into shared memory
            V_ldg_reg = V_m_guard ? V[m_id_for_load_V] : false;
            V_smem[threadIdx.x] = V_ldg_reg;

            // load first A_tile to shared memory
            ldg32_nc_0(A_ldg_reg, A_ldg_ptr, A_k_guard && A_m_guard);
            sts32(A_ldg_reg, A_sts_addr);
            __syncthreads();

            // index for data loading
            m_id_for_load_V += M_TILE;
            m_id_for_load_A += M_TILE;
            V_m_guard = m_id_for_load_V < m;
            A_m_guard = m_id_for_load_A < m;
            A_ldg_ptr += M_TILE * sizeof(float);

            const int num_m_tile_loop = (m + M_TILE - 1) / M_TILE - 1;
            const int last_m_tile = k - num_m_tile_loop * M_TILE;

            // number of m tile to read A
            for (int mm = 0; mm < num_m_tile_loop; ++mm) {
#pragma unroll
                for (int m_frag = 0; m_frag < M_TILE; ++m_frag) {
                    if (m_frag == 0) {
                        V_ldg_reg = V_m_guard ? V[m_id_for_load_V] : false;
                        ldg32_nc(A_ldg_reg, A_ldg_ptr, A_k_guard && A_m_guard);
                    }

                    // if spike and synapse id arrived, FFMA
                    if (V_smem[m_frag]) {
                        if (taus88_double((m_frag + m_id_for_load_A) * n + n_id) < log_p) {
#pragma unroll
                            for (int i = 0; i < K_TILE; i += 4) {
                                lds128(A_frag[0], A_frag[1], A_frag[2], A_frag[3],
                                       A_lds_addr + (m_frag * K_TILE + i) * sizeof(float));
                                O_frag[i] += A_frag[0];
                                O_frag[i + 1] += A_frag[1];
                                O_frag[i + 2] += A_frag[2];
                                O_frag[i + 3] += A_frag[3];
                            }
                        }
                    }

                    if (m_frag == M_TILE - 1) {
                        V_smem[threadIdx.x] = V_ldg_reg;
                        sts32(A_ldg_reg, A_sts_addr);
                        m_id_for_load_V += M_TILE;
                        m_id_for_load_A += M_TILE;
                        V_m_guard = m_id_for_load_V < m;
                        A_m_guard = m_id_for_load_A < m;
                        A_ldg_ptr += M_TILE * sizeof(float);
                        __syncthreads();
                    }
                }
            }

            // FFMA for the last tile
#pragma unroll
            for (int m_frag = 0; m_frag < M_TILE; ++m_frag) {
                if (m_frag < last_m_tile) {
                    if (V_smem[m_frag]) {
                        if (taus88_double((m_frag + m_id_for_load_A) * n + n_id) < log_p) {
#pragma unroll
                            for (int i = 0; i < K_TILE; i += 4) {
                                lds128(A_frag[0], A_frag[1], A_frag[2], A_frag[3],
                                       A_lds_addr + (m_frag * K_TILE + i) * sizeof(float));
                                O_frag[i] += A_frag[0];
                                O_frag[i + 1] += A_frag[1];
                                O_frag[i + 2] += A_frag[2];
                                O_frag[i + 3] += A_frag[3];
                            }
                        }
                    }
                }
            }

            // O_tile write back
            if (n_id < n) {
#pragma unroll
                for (int i = 0; i < K_TILE; ++i) {
                    if (i < k) { O[i * n + n_id] = O_frag[i]; }
                }
            }
        }
    }  // namespace


    void event_mmm_fp_v1_4x64x256(
            cudaStream_t stream, void **buffers, const char *opaque, std::size_t opaque_len
    ) {
        const MatMulDescriptor &d = *UnpackDescriptor<MatMulDescriptor>(opaque, opaque_len);
        const std::uint32_t m = d.m;
        const std::uint32_t k = d.k;
        const std::uint32_t n = d.n;
        const std::uint32_t seed = d.seed;
        const float log_p = d.p;

        const bool *V = reinterpret_cast<const bool *>(buffers[0]);
        const float *A = reinterpret_cast<const float *>(buffers[1]);
        float *O = reinterpret_cast<float *>(buffers[2]);

        cudaMemset(O, 0, sizeof(float) * n * k);
        dim3 grid((n + 255) / 256, 1);
        event_mmm_fixedprob_kernel_v1<4, 64><<<grid, 256, 0, stream>>>(V, A, m, n, k, seed, m * sizeof(float), log_p, O);
        ThrowIfError(cudaGetLastError());
    }

    void event_mmm_fp_v2_4x64x256(
            cudaStream_t stream, void **buffers, const char *opaque, std::size_t opaque_len
    ) {
        const MatMulDescriptor &d = *UnpackDescriptor<MatMulDescriptor>(opaque, opaque_len);
        const std::uint32_t m = d.m;
        const std::uint32_t k = d.k;
        const std::uint32_t n = d.n;
        const std::uint32_t seed = d.seed;
        const float log_p = d.p;

        const bool *V = reinterpret_cast<const bool *>(buffers[0]);
        const float *A = reinterpret_cast<const float *>(buffers[1]);
        float *O = reinterpret_cast<float *>(buffers[2]);

        cudaMemset(O, 0, sizeof(float) * n * k);
        dim3 grid((n + 255) / 256, 1);
        event_mmm_fixedprob_kernel_v2<4, 64><<<grid, 256, 0, stream>>>(V, A, m, n, k, seed, m * sizeof(float), log_p, O);
        ThrowIfError(cudaGetLastError());
    }

    void event_mmm_fp_v1_8x32x256(
            cudaStream_t stream, void **buffers, const char *opaque, std::size_t opaque_len
    ) {
        const MatMulDescriptor &d = *UnpackDescriptor<MatMulDescriptor>(opaque, opaque_len);
        const std::uint32_t m = d.m;
        const std::uint32_t k = d.k;
        const std::uint32_t n = d.n;
        const std::uint32_t seed = d.seed;
        const float log_p = d.p;

        const bool *V = reinterpret_cast<const bool *>(buffers[0]);
        const float *A = reinterpret_cast<const float *>(buffers[1]);
        float *O = reinterpret_cast<float *>(buffers[2]);

        cudaMemset(O, 0, sizeof(float) * n * k);
        dim3 grid((n + 255) / 256, 1);
        event_mmm_fixedprob_kernel_v1<8, 32><<<grid, 256, 0, stream>>>(V, A, m, n, k, seed, m * sizeof(float), log_p, O);
        ThrowIfError(cudaGetLastError());
    }

    void event_mmm_fp_v2_8x32x256(
            cudaStream_t stream, void **buffers, const char *opaque, std::size_t opaque_len
    ) {
        const MatMulDescriptor &d = *UnpackDescriptor<MatMulDescriptor>(opaque, opaque_len);
        const std::uint32_t m = d.m;
        const std::uint32_t k = d.k;
        const std::uint32_t n = d.n;
        const std::uint32_t seed = d.seed;
        const float log_p = d.p;

        const bool *V = reinterpret_cast<const bool *>(buffers[0]);
        const float *A = reinterpret_cast<const float *>(buffers[1]);
        float *O = reinterpret_cast<float *>(buffers[2]);

        cudaMemset(O, 0, sizeof(float) * n * k);
        dim3 grid((n + 255) / 256, 1);
        event_mmm_fixedprob_kernel_v2<8, 32><<<grid, 256, 0, stream>>>(V, A, m, n, k, seed, m * sizeof(float), log_p, O);
        ThrowIfError(cudaGetLastError());
    }

    void event_mmm_fp_v1_16x16x256(
            cudaStream_t stream, void **buffers, const char *opaque, std::size_t opaque_len
    ) {
        const MatMulDescriptor &d = *UnpackDescriptor<MatMulDescriptor>(opaque, opaque_len);
        const std::uint32_t m = d.m;
        const std::uint32_t k = d.k;
        const std::uint32_t n = d.n;
        const std::uint32_t seed = d.seed;
        const float log_p = d.p;

        const bool *V = reinterpret_cast<const bool *>(buffers[0]);
        const float *A = reinterpret_cast<const float *>(buffers[1]);
        float *O = reinterpret_cast<float *>(buffers[2]);

        cudaMemset(O, 0, sizeof(float) * n * k);
        dim3 grid((n + 255) / 256, 1);
        event_mmm_fixedprob_kernel_v1<16, 16><<<grid, 256, 0, stream>>>(V, A, m, n, k, seed, m * sizeof(float), log_p, O);
        ThrowIfError(cudaGetLastError());
    }

    void event_mmm_fp_v2_16x16x256(
            cudaStream_t stream, void **buffers, const char *opaque, std::size_t opaque_len
    ) {
        const MatMulDescriptor &d = *UnpackDescriptor<MatMulDescriptor>(opaque, opaque_len);
        const std::uint32_t m = d.m;
        const std::uint32_t k = d.k;
        const std::uint32_t n = d.n;
        const std::uint32_t seed = d.seed;
        const float log_p = d.p;

        const bool *V = reinterpret_cast<const bool *>(buffers[0]);
        const float *A = reinterpret_cast<const float *>(buffers[1]);
        float *O = reinterpret_cast<float *>(buffers[2]);

        cudaMemset(O, 0, sizeof(float) * n * k);
        dim3 grid((n + 255) / 256, 1);
        event_mmm_fixedprob_kernel_v2<16, 16><<<grid, 256, 0, stream>>>(V, A, m, n, k, seed, m * sizeof(float), log_p, O);
        ThrowIfError(cudaGetLastError());
    }

    void event_mmm_fp_v1_32x8x256(
            cudaStream_t stream, void **buffers, const char *opaque, std::size_t opaque_len
    ) {
        const MatMulDescriptor &d = *UnpackDescriptor<MatMulDescriptor>(opaque, opaque_len);
        const std::uint32_t m = d.m;
        const std::uint32_t k = d.k;
        const std::uint32_t n = d.n;
        const std::uint32_t seed = d.seed;
        const float log_p = d.p;

        const bool *V = reinterpret_cast<const bool *>(buffers[0]);
        const float *A = reinterpret_cast<const float *>(buffers[1]);
        float *O = reinterpret_cast<float *>(buffers[2]);

        cudaMemset(O, 0, sizeof(float) * n * k);
        dim3 grid((n + 255) / 256, 1);
        event_mmm_fixedprob_kernel_v1<32, 8><<<grid, 256, 0, stream>>>(V, A, m, n, k, seed, m * sizeof(float), log_p, O);
        ThrowIfError(cudaGetLastError());
    }

    void event_mmm_fp_v2_32x8x256(
            cudaStream_t stream, void **buffers, const char *opaque, std::size_t opaque_len
    ) {
        const MatMulDescriptor &d = *UnpackDescriptor<MatMulDescriptor>(opaque, opaque_len);
        const std::uint32_t m = d.m;
        const std::uint32_t k = d.k;
        const std::uint32_t n = d.n;
        const std::uint32_t seed = d.seed;
        const float log_p = d.p;

        const bool *V = reinterpret_cast<const bool *>(buffers[0]);
        const float *A = reinterpret_cast<const float *>(buffers[1]);
        float *O = reinterpret_cast<float *>(buffers[2]);

        cudaMemset(O, 0, sizeof(float) * n * k);
        dim3 grid((n + 255) / 256, 1);
        event_mmm_fixedprob_kernel_v2<32, 8><<<grid, 256, 0, stream>>>(V, A, m, n, k, seed, m * sizeof(float), log_p, O);
        ThrowIfError(cudaGetLastError());
    }

}  // namespace brainpylib