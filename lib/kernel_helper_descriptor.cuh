// This header is not specific to our application and you'll probably want something like this
// for any extension you're building. This includes the infrastructure needed to serialize
// descriptors that are used with the "opaque" parameter of the GPU custom call. In our example
// we'll use this parameter to pass the size of our problem.

#ifndef _BRAINPYLIB_CUDA_OP_MATMUL_HELPERS_H_
#define _BRAINPYLIB_CUDA_OP_MATMUL_HELPERS_H_


#include <cstddef>
#include <cstdint>
#include "cuda_runtime_api.h"
#include "pybind11_kernel_helpers.h"
#include "kernel_helpers_gpu.cuh"


namespace brainpy_lib {
    // Descriptors
    struct MatMulDescriptor {
        std::uint32_t m;
        std::uint32_t k;
        std::uint32_t n;
        std::uint32_t seed;
        float p;
    };

    pybind11::bytes
    build_matmul_descriptor(std::uint32_t m, std::uint32_t k, std::uint32_t n, std::uint32_t seed, float p);


    struct MMMDescriptor {
        std::uint32_t m;
        std::uint32_t k;
        std::uint32_t n;
        float p;
    };

    pybind11::bytes build_mmm_descriptor(std::uint32_t m, std::uint32_t k, std::uint32_t n, float p);


    struct NonZeroDescriptor {
        std::uint32_t event_size;
        std::uint32_t batch_size;
    };

    pybind11::bytes build_nonzero_descriptor(std::uint32_t event_size, std::uint32_t batch_size);


    struct JITConnProbHomoDescriptor {
        unsigned int n_row;
        unsigned int n_col;
        unsigned int seed;
        float prob;
    };

    pybind11::bytes build_jitconn_prob_homo_descriptor(unsigned int n_row,
                                                       unsigned int n_col,
                                                       unsigned int seed,
                                                       float prob);


    struct JITConnProbUniformDescriptor {
        unsigned int n_row;
        unsigned int n_col;
        unsigned int seed;
        float prob;
        float w_min;
        float w_range;
    };

    pybind11::bytes build_jitconn_prob_uniform_descriptor(unsigned int n_row,
                                                       unsigned int n_col,
                                                       unsigned int seed,
                                                       float prob,
                                                       float w_min,
                                                       float w_range);


    struct JITConnProbNormalDescriptor {
        unsigned int n_row;
        unsigned int n_col;
        unsigned int seed;
        float prob;
        float w_mu;
        float w_sigma;
    };

    pybind11::bytes build_jitconn_prob_normal_descriptor(unsigned int n_row,
                                                       unsigned int n_col,
                                                       unsigned int seed,
                                                       float prob,
                                                       float w_mu,
                                                       float w_sigma);


    struct SingleSizeDescriptor {
        unsigned int size;
    };

    pybind11::bytes build_single_size_descriptor(unsigned int size);

    struct DoubleSizeDescriptor {
        unsigned int size_x;
        unsigned int size_y;
    };

    pybind11::bytes build_double_size_descriptor(unsigned int size_x, unsigned int size_y);

    struct TripleSizeDescriptor {
        unsigned int size_x;
        unsigned int size_y;
        unsigned int size_z;
    };

    pybind11::bytes build_triple_size_descriptor(unsigned int size_x,
                                                 unsigned int size_y,
                                                 unsigned int size_z);


    struct TwoUintOneBoolDescriptor {
        unsigned int uint_x;
        unsigned int uint_y;
        bool bool_x;
    };

    pybind11::bytes build_twouint_onebool_descriptor(unsigned int uint_x, unsigned int uint_y, bool bool_x);


    struct OneFloatDescriptor {
        float x;
    };

    pybind11::bytes build_onefloat_descriptor(float x);

    struct TwoFloatDescriptor {
        float x;
        float y;
    };

    pybind11::bytes build_twofloat_descriptor(float x, float y);

    struct ThreeFloatDescriptor {
        float x;
        float y;
        float z;
    };

    pybind11::bytes build_threefloat_descriptor(float x, float y, float z);


}  // namespace brainpy_lib

#endif
