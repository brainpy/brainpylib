// This file defines the Python interface to the XLA custom call implemented on the GPU.
// Like in cpu_ops.cc, we export a separate capsule for each supported dtype, but we also
// include one extra method "build_kepler_descriptor" to generate an opaque representation
// of the problem size that will be passed to the op. The actual implementation of the
// custom call can be found in kernels.cc.cu.

#include "pybind11_kernel_helpers.h"
#include "kernel_helper_descriptor.cuh"
#include "gpu_event_sum.h"
#include "gpu_atomic_sum.h"
#include "gpu_atomic_prod.h"
#include "gpu_nonzero_op.cuh"
#include "gpu_csr_matvec.cuh"

using namespace brainpy_lib;

namespace {
    pybind11::dict Registrations() {
        pybind11::dict dict;

        // OP: homogeneous csr event_sum
        dict["gpu_csr_event_sum_homo_f32_i32"] = EncapsulateFunction(gpu_csr_event_sum_homo_f32_i32);
        dict["gpu_csr_event_sum_homo_f32_i64"] = EncapsulateFunction(gpu_csr_event_sum_homo_f32_i64);
        dict["gpu_csr_event_sum_homo_f64_i32"] = EncapsulateFunction(gpu_csr_event_sum_homo_f64_i32);
        dict["gpu_csr_event_sum_homo_f64_i64"] = EncapsulateFunction(gpu_csr_event_sum_homo_f64_i64);
        // OP: heterogeneous csr event_sum
        dict["gpu_csr_event_sum_heter_f32_i32"] = EncapsulateFunction(gpu_csr_event_sum_heter_f32_i32);
        dict["gpu_csr_event_sum_heter_f32_i64"] = EncapsulateFunction(gpu_csr_event_sum_heter_f32_i64);
        dict["gpu_csr_event_sum_heter_f64_i32"] = EncapsulateFunction(gpu_csr_event_sum_heter_f64_i32);
        dict["gpu_csr_event_sum_heter_f64_i64"] = EncapsulateFunction(gpu_csr_event_sum_heter_f64_i64);

        // OP: homogeneous coo event_sum
        dict["gpu_coo_event_sum_homo_f32_i32"] = EncapsulateFunction(gpu_coo_event_sum_homo_f32_i32);
        dict["gpu_coo_event_sum_homo_f32_i64"] = EncapsulateFunction(gpu_coo_event_sum_homo_f32_i64);
        dict["gpu_coo_event_sum_homo_f64_i32"] = EncapsulateFunction(gpu_coo_event_sum_homo_f64_i32);
        dict["gpu_coo_event_sum_homo_f64_i64"] = EncapsulateFunction(gpu_coo_event_sum_homo_f64_i64);
        // OP: heterogeneous coo event_sum
        dict["gpu_coo_event_sum_heter_f32_i32"] = EncapsulateFunction(gpu_coo_event_sum_heter_f32_i32);
        dict["gpu_coo_event_sum_heter_f32_i64"] = EncapsulateFunction(gpu_coo_event_sum_heter_f32_i64);
        dict["gpu_coo_event_sum_heter_f64_i32"] = EncapsulateFunction(gpu_coo_event_sum_heter_f64_i32);
        dict["gpu_coo_event_sum_heter_f64_i64"] = EncapsulateFunction(gpu_coo_event_sum_heter_f64_i64);

        // OP: homogeneous atomic_sum
        dict["gpu_coo_atomic_sum_homo_f32_i32"] = EncapsulateFunction(gpu_coo_atomic_sum_homo_f32_i32);
        dict["gpu_coo_atomic_sum_homo_f32_i64"] = EncapsulateFunction(gpu_coo_atomic_sum_homo_f32_i64);
        dict["gpu_coo_atomic_sum_homo_f64_i32"] = EncapsulateFunction(gpu_coo_atomic_sum_homo_f64_i32);
        dict["gpu_coo_atomic_sum_homo_f64_i64"] = EncapsulateFunction(gpu_coo_atomic_sum_homo_f64_i64);
        // OP: heterogeneous atomic_sum
        dict["gpu_coo_atomic_sum_heter_f32_i32"] = EncapsulateFunction(gpu_coo_atomic_sum_heter_f32_i32);
        dict["gpu_coo_atomic_sum_heter_f32_i64"] = EncapsulateFunction(gpu_coo_atomic_sum_heter_f32_i64);
        dict["gpu_coo_atomic_sum_heter_f64_i32"] = EncapsulateFunction(gpu_coo_atomic_sum_heter_f64_i32);
        dict["gpu_coo_atomic_sum_heter_f64_i64"] = EncapsulateFunction(gpu_coo_atomic_sum_heter_f64_i64);

        // OP: homogeneous atomic_prod
        dict["gpu_coo_atomic_prod_homo_f32_i32"] = EncapsulateFunction(gpu_coo_atomic_prod_homo_f32_i32);
        dict["gpu_coo_atomic_prod_homo_f32_i64"] = EncapsulateFunction(gpu_coo_atomic_prod_homo_f32_i64);
        dict["gpu_coo_atomic_prod_homo_f64_i32"] = EncapsulateFunction(gpu_coo_atomic_prod_homo_f64_i32);
        dict["gpu_coo_atomic_prod_homo_f64_i64"] = EncapsulateFunction(gpu_coo_atomic_prod_homo_f64_i64);
        // OP: heterogeneous atomic_prod
        dict["gpu_coo_atomic_prod_heter_f32_i32"] = EncapsulateFunction(gpu_coo_atomic_prod_heter_f32_i32);
        dict["gpu_coo_atomic_prod_heter_f32_i64"] = EncapsulateFunction(gpu_coo_atomic_prod_heter_f32_i64);
        dict["gpu_coo_atomic_prod_heter_f64_i32"] = EncapsulateFunction(gpu_coo_atomic_prod_heter_f64_i32);
        dict["gpu_coo_atomic_prod_heter_f64_i64"] = EncapsulateFunction(gpu_coo_atomic_prod_heter_f64_i64);

        // OP: nonzero
        dict["nonzero_64"] = EncapsulateFunction(nonzero_64);
        dict["nonzero_128"] = EncapsulateFunction(nonzero_128);
        dict["nonzero_256"] = EncapsulateFunction(nonzero_256);

        // OP: heterogeneous csr matvec
        dict["csr_matvec_heter_scalar_float"] = EncapsulateFunction(csr_matvec_heter_scalar_float);
        dict["csr_matvec_heter_scalar_double"] = EncapsulateFunction(csr_matvec_heter_scalar_double);
        dict["csr_matvec_heter_vector_float"] = EncapsulateFunction(csr_matvec_heter_vector_float);
        dict["csr_matvec_heter_vector_double"] = EncapsulateFunction(csr_matvec_heter_vector_double);
        dict["csr_matvec_heter_adaptive_float"] = EncapsulateFunction(csr_matvec_heter_adaptive_float);
        dict["csr_matvec_heter_adaptive_double"] = EncapsulateFunction(csr_matvec_heter_adaptive_double);
        // OP: homogeneous csr matvec
        dict["csr_matvec_homo_scalar_float"] = EncapsulateFunction(csr_matvec_homo_scalar_float);
        dict["csr_matvec_homo_scalar_double"] = EncapsulateFunction(csr_matvec_homo_scalar_double);
        dict["csr_matvec_homo_vector_float"] = EncapsulateFunction(csr_matvec_homo_vector_float);
        dict["csr_matvec_homo_vector_double"] = EncapsulateFunction(csr_matvec_homo_vector_double);
        dict["csr_matvec_homo_adaptive_float"] = EncapsulateFunction(csr_matvec_homo_adaptive_float);
        dict["csr_matvec_homo_adaptive_double"] = EncapsulateFunction(csr_matvec_homo_adaptive_double);

        return dict;
    }

    PYBIND11_MODULE(gpu_ops, m
    ) {
    m.def("registrations", &Registrations);
    m.def("build_csr_event_sum_descriptor", &build_csr_event_sum_descriptor);
    m.def("build_coo_event_sum_descriptor", &build_coo_event_sum_descriptor);
    m.def("build_coo_atomic_sum_descriptor", &build_coo_atomic_sum_descriptor);
    m.def("build_coo_atomic_prod_descriptor", &build_coo_atomic_prod_descriptor);
    m.def("build_matmul_descriptor", &build_matmul_descriptor);
    m.def("build_mmm_descriptor", &build_mmm_descriptor);
    m.def("build_nonzero_descriptor", &build_nonzero_descriptor);
    m.def("build_single_size_descriptor", &build_single_size_descriptor);
    m.def("build_double_size_descriptor", &build_double_size_descriptor);
    m.def("build_triple_size_descriptor", &build_triple_size_descriptor);
}
}  // namespace
