// This header is not specific to our application and you'll probably want something like this
// for any extension you're building. This includes the infrastructure needed to serialize
// descriptors that are used with the "opaque" parameter of the GPU custom call. In our example
// we'll use this parameter to pass the size of our problem.


#include "kernel_helper_descriptor.cuh"


namespace brainpy_lib {

    // Descriptors
    pybind11::bytes build_matmul_descriptor(std::uint32_t m,
                                            std::uint32_t k,
                                            std::uint32_t n,
                                            std::uint32_t seed,
                                            float p) {
        return PackDescriptor(MatMulDescriptor{m, k, n, seed, p});
    }

    pybind11::bytes build_mmm_descriptor(std::uint32_t m,
                                         std::uint32_t k,
                                         std::uint32_t n,
                                         float p) {
        return PackDescriptor(MMMDescriptor{m, k, n, p});
    }

    pybind11::bytes build_nonzero_descriptor(std::uint32_t event_size,
                                             std::uint32_t batch_size) {
        return PackDescriptor(NonZeroDescriptor{event_size, batch_size});
    }


    pybind11::bytes build_single_size_descriptor(unsigned int size) {
        return PackDescriptor(SingleSizeDescriptor{size});
    };

    pybind11::bytes build_double_size_descriptor(unsigned int size_x,
                                                 unsigned int size_y) {
        return PackDescriptor(DoubleSizeDescriptor{size_x, size_y});
    };

    pybind11::bytes build_triple_size_descriptor(unsigned int size_x,
                                                 unsigned int size_y,
                                                 unsigned int size_z) {
        return PackDescriptor(TripleSizeDescriptor{size_x, size_y, size_z});
    };


}  // namespace brainpy_lib

