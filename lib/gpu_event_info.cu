//
// Created by adadu on 2022/11/21.
//

#include "gpu_event_info.cuh"


namespace brainpy_lib {
    namespace {

        template<typename T, const int NUM_THREAD>
        __global__ void collect_spike_info(
                const std::uint32_t size,
                const T *events,
                int *event_ids,
                int *event_num
        ) {
            const int id = blockDim.x * blockIdx.x + threadIdx.x;
            const int gid = size * blockIdx.y + id;
            __shared__ unsigned int shSpk[NUM_THREAD];
            __shared__ unsigned int shPosSpk;
            __shared__ unsigned int shSpkCount;
            if (threadIdx.x == 0) {
                shSpkCount = 0;
            }
            __syncthreads();

            if (id < size) {
                if (events[gid]) {
                    const unsigned int spkIdx = atomicAdd(&shSpkCount, 1);
                    shSpk[spkIdx] = id;
                }
                __syncthreads();

                if (threadIdx.x == 0) {
                    shPosSpk = atomicAdd(&event_num[blockIdx.y], shSpkCount);
                }
                __syncthreads();

                if (threadIdx.x < shSpkCount) {
                    const int n = shSpk[threadIdx.x];
                    event_ids[blockIdx.y * size + shPosSpk + threadIdx.x] = n;
                }
            }
        }

        template<typename T>
        inline void nonzero_256(cudaStream_t stream, void **buffers, const char *opaque, std::size_t opaque_len) {
            const NonZeroDescriptor &d = *UnpackDescriptor<NonZeroDescriptor>(opaque, opaque_len);
            const std::uint32_t event_size = d.event_size;
            const std::uint32_t batch_size = d.batch_size;

            const T *events = reinterpret_cast<const bool *>(buffers[0]);
            int *event_ids = reinterpret_cast<int *>(buffers[1]);
            int *event_num = reinterpret_cast<int *>(buffers[2]);

            cudaMemset(event_ids, -1, sizeof(int) * event_size * batch_size);
            cudaMemset(event_num, 0, sizeof(int) * batch_size);
            dim3 grid((event_size + 255) / 256, batch_size);
            collect_spike_info<256><<<grid, 256, 0, stream>>>(event_size, events, event_ids, event_num);
            ThrowIfError(cudaGetLastError());
        }


    }


    void nonzero_bool(cudaStream_t stream, void **buffers, const char *opaque, std::size_t opaque_len) {
        nonzero_256<bool>(stream, buffers, opaque, opaque_len);
    }


    void nonzero_int(cudaStream_t stream, void **buffers, const char *opaque, std::size_t opaque_len) {
        nonzero_256<int>(stream, buffers, opaque, opaque_len);
    }


    void nonzero_long(cudaStream_t stream, void **buffers, const char *opaque, std::size_t opaque_len) {
        nonzero_256<long>(stream, buffers, opaque, opaque_len);
    }


    void nonzero_float(cudaStream_t stream, void **buffers, const char *opaque, std::size_t opaque_len) {
        nonzero_256<float>(stream, buffers, opaque, opaque_len);
    }

    void nonzero_double(cudaStream_t stream, void **buffers, const char *opaque, std::size_t opaque_len) {
        nonzero_256<double>(stream, buffers, opaque, opaque_len);
    }


}


