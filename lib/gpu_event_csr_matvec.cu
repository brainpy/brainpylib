//
// Created by Chaoming Wang on 2022/11/26.
//

#include "gpu_event_csr_matvec.cuh"


namespace brainpy_lib{
    namespace {

        template<typename data_type>
        __global__ void csr_spmv_vector_kernel(
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
            if (lane == 0 && row < n_rows)
                y[row] = sum;
        }

    }




}

