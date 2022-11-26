//
// Created by adadu on 2022/11/21.
//

#include "gpu_nonzero_op.cuh"


namespace brainpy_lib {
    namespace {

        template<const int NUM_THREAD>
        __global__ void collect_spike_info(
                const std::uint32_t size,
                const bool *events,
                unsigned int *event_ids,
                unsigned int *event_num
        ) {
            const unsigned int id = blockDim.x * blockIdx.x + threadIdx.x;
            __shared__ unsigned int shSpk[NUM_THREAD];
            __shared__ unsigned int shPosSpk;
            __shared__ unsigned int shSpkCount;
            if (threadIdx.x == 0) {
                shSpkCount = 0;
            }
            __syncthreads();

            if (id < size) {
                if (events[id]) {
                    const unsigned int spkIdx = atomicAdd(&shSpkCount, 1);
                    shSpk[spkIdx] = id;
                }
                __syncthreads();

                if (threadIdx.x == 0) {
                    shPosSpk = atomicAdd(&event_num[0], shSpkCount);
                }
                __syncthreads();

                if (threadIdx.x < shSpkCount) {
                    const unsigned int n = shSpk[threadIdx.x];
                    event_ids[shPosSpk + threadIdx.x] = n;
                }
            }
        }


    }


    void nonzero_64(
            cudaStream_t stream, void **buffers, const char *opaque, std::size_t opaque_len
    ) {
        const SizeDescriptor &d = *UnpackDescriptor<SizeDescriptor>(opaque, opaque_len);
        const std::uint32_t m = d.size;

        const bool *events = reinterpret_cast<const bool *>(buffers[0]);
        int *event_ids = reinterpret_cast<int *>(buffers[1]);
        int *event_num = reinterpret_cast<int *>(buffers[2]);

        cudaMemset(event_ids, -1, sizeof(int) * m);
        dim3 grid((n + 63) / 64, 1);
        collect_spike_info < 64 ><<<grid, 64, 0, stream>>>(m, events, event_ids, event_num);
        ThrowIfError(cudaGetLastError());
    }


}


