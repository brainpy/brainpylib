// This file defines the Python interface to the XLA custom call implemented on the GPU.
// Like in cpu_ops.cc, we export a separate capsule for each supported dtype, but we also
// include one extra method "build_kepler_descriptor" to generate an opaque representation
// of the problem size that will be passed to the op. The actual implementation of the
// custom call can be found in kernels.cc.cu.

#include "pybind11_kernel_helpers.h"
#include "kernel_helper_descriptor.cuh"
#include "gpu_event_sum.cuh"
#include "gpu_atomic_sum.cuh"
#include "gpu_atomic_prod.cuh"
#include "gpu_event_info.cuh"
#include "gpu_csr_matvec.cuh"
#include "gpu_event_csr_matvec.cuh"
#include "gpu_jitconn_event_matvec.cuh"
#include "gpu_jitconn_event_matvec_atomic.cuh"
#include "gpu_jitconn_matvec.cuh"
#include "gpu_jitconn_matvec_atomic.cuh"
#include "gpu_jitconn_matmat.cuh"

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
        dict["nonzero_bool"] = EncapsulateFunction(nonzero_bool);
        dict["nonzero_int"] = EncapsulateFunction(nonzero_int);
        dict["nonzero_long"] = EncapsulateFunction(nonzero_long);
        dict["nonzero_float"] = EncapsulateFunction(nonzero_float);
        dict["nonzero_double"] = EncapsulateFunction(nonzero_double);

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

        // OP: heterogeneous event csr matvec
        dict["event_csr_matvec_heter_float_bool"] = EncapsulateFunction(event_csr_matvec_heter_float_bool);
        dict["event_csr_matvec_heter_float_float"] = EncapsulateFunction(event_csr_matvec_heter_float_float);
        dict["event_csr_matvec_heter_double_bool"] = EncapsulateFunction(event_csr_matvec_heter_double_bool);
        dict["event_csr_matvec_heter_double_double"] = EncapsulateFunction(event_csr_matvec_heter_double_double);

        // OP: homogeneous event csr matvec
        dict["event_csr_matvec_homo_float_bool"] = EncapsulateFunction(event_csr_matvec_homo_float_bool);
        dict["event_csr_matvec_homo_float_float"] = EncapsulateFunction(event_csr_matvec_homo_float_float);
        dict["event_csr_matvec_homo_double_bool"] = EncapsulateFunction(event_csr_matvec_homo_double_bool);
        dict["event_csr_matvec_homo_double_double"] = EncapsulateFunction(event_csr_matvec_homo_double_double);

        // OP: mat (with jitconn) @ events
        dict["gpu_event_matvec_prob_homo_float"] = EncapsulateFunction(event_matvec_jitconn_prob_homo_float);
        dict["gpu_event_matvec_prob_homo_double"] = EncapsulateFunction(event_matvec_jitconn_prob_homo_double);
        dict["gpu_event_matvec_prob_uniform_float"] = EncapsulateFunction(event_matvec_jitconn_prob_uniform_float);
        dict["gpu_event_matvec_prob_uniform_double"] = EncapsulateFunction(event_matvec_jitconn_prob_uniform_double);
        dict["gpu_event_matvec_prob_normal_float"] = EncapsulateFunction(event_matvec_jitconn_prob_normal_float);
        dict["gpu_event_matvec_prob_normal_double"] = EncapsulateFunction(event_matvec_jitconn_prob_normal_double);

        // OP: mat (with jitconn) @ events V2
        dict["gpu_event_matvec_prob_homo_v2_float_bool"] = EncapsulateFunction(event_matvec_jitconn_prob_homo_v2_float_bool);
        dict["gpu_event_matvec_prob_homo_v2_float_float"] = EncapsulateFunction(event_matvec_jitconn_prob_homo_v2_float_float);
        dict["gpu_event_matvec_prob_homo_v2_double_bool"] = EncapsulateFunction(event_matvec_jitconn_prob_homo_v2_double_bool);
        dict["gpu_event_matvec_prob_homo_v2_double_double"] = EncapsulateFunction(event_matvec_jitconn_prob_homo_v2_double_double);
        dict["gpu_event_matvec_prob_uniform_v2_float_bool"] = EncapsulateFunction(event_matvec_jitconn_prob_uniform_v2_float_bool);
        dict["gpu_event_matvec_prob_uniform_v2_float_float"] = EncapsulateFunction(event_matvec_jitconn_prob_uniform_v2_float_float);
        dict["gpu_event_matvec_prob_uniform_v2_double_bool"] = EncapsulateFunction(event_matvec_jitconn_prob_uniform_v2_double_bool);
        dict["gpu_event_matvec_prob_uniform_v2_double_double"] = EncapsulateFunction(event_matvec_jitconn_prob_uniform_v2_double_double);
        dict["gpu_event_matvec_prob_normal_v2_float_bool"] = EncapsulateFunction(event_matvec_jitconn_prob_normal_v2_float_bool);
        dict["gpu_event_matvec_prob_normal_v2_float_float"] = EncapsulateFunction(event_matvec_jitconn_prob_normal_v2_float_float);
        dict["gpu_event_matvec_prob_normal_v2_double_bool"] = EncapsulateFunction(event_matvec_jitconn_prob_normal_v2_double_bool);
        dict["gpu_event_matvec_prob_normal_v2_double_double"] = EncapsulateFunction(event_matvec_jitconn_prob_normal_v2_double_double);

        // OP: mat (with jitconn) @ events atomic V2
        dict["gpu_event_matvec_atomic_prob_homo_v2_float_bool"] = EncapsulateFunction(event_matvec_atomic_jitconn_prob_homo_v2_float_bool);
        dict["gpu_event_matvec_atomic_prob_homo_v2_float_float"] = EncapsulateFunction(event_matvec_atomic_jitconn_prob_homo_v2_float_float);
        dict["gpu_event_matvec_atomic_prob_homo_v2_double_bool"] = EncapsulateFunction(event_matvec_atomic_jitconn_prob_homo_v2_double_bool);
        dict["gpu_event_matvec_atomic_prob_homo_v2_double_double"] = EncapsulateFunction(event_matvec_atomic_jitconn_prob_homo_v2_double_double);
        dict["gpu_event_matvec_atomic_prob_uniform_v2_float_bool"] = EncapsulateFunction(event_matvec_atomic_jitconn_prob_uniform_v2_float_bool);
        dict["gpu_event_matvec_atomic_prob_uniform_v2_float_float"] = EncapsulateFunction(event_matvec_atomic_jitconn_prob_uniform_v2_float_float);
        dict["gpu_event_matvec_atomic_prob_uniform_v2_double_bool"] = EncapsulateFunction(event_matvec_atomic_jitconn_prob_uniform_v2_double_bool);
        dict["gpu_event_matvec_atomic_prob_uniform_v2_double_double"] = EncapsulateFunction(event_matvec_atomic_jitconn_prob_uniform_v2_double_double);
        dict["gpu_event_matvec_atomic_prob_normal_v2_float_bool"] = EncapsulateFunction(event_matvec_atomic_jitconn_prob_normal_v2_float_bool);
        dict["gpu_event_matvec_atomic_prob_normal_v2_float_float"] = EncapsulateFunction(event_matvec_atomic_jitconn_prob_normal_v2_float_float);
        dict["gpu_event_matvec_atomic_prob_normal_v2_double_bool"] = EncapsulateFunction(event_matvec_atomic_jitconn_prob_normal_v2_double_bool);
        dict["gpu_event_matvec_atomic_prob_normal_v2_double_double"] = EncapsulateFunction(event_matvec_atomic_jitconn_prob_normal_v2_double_double);

        // OP: mat (with jitconn) @ vector
        dict["gpu_matvec_prob_homo_v1_float"] = EncapsulateFunction(matvec_jitconn_prob_homo_float);
        dict["gpu_matvec_prob_homo_v1_double"] = EncapsulateFunction(matvec_jitconn_prob_homo_double);
        dict["gpu_matvec_prob_uniform_v1_float"] = EncapsulateFunction(matvec_jitconn_prob_uniform_float);
        dict["gpu_matvec_prob_uniform_v1_double"] = EncapsulateFunction(matvec_jitconn_prob_uniform_double);
        dict["gpu_matvec_prob_normal_v1_float"] = EncapsulateFunction(matvec_jitconn_prob_normal_float);
        dict["gpu_matvec_prob_normal_v1_double"] = EncapsulateFunction(matvec_jitconn_prob_normal_double);

        // OP: mat (with jitconn) @ vector V2
        dict["gpu_matvec_prob_homo_v2_float"] = EncapsulateFunction(matvec_jitconn_prob_homo_v2_float);
        dict["gpu_matvec_prob_homo_v2_double"] = EncapsulateFunction(matvec_jitconn_prob_homo_v2_double);
        dict["gpu_matvec_prob_uniform_v2_float"] = EncapsulateFunction(matvec_jitconn_prob_uniform_v2_float);
        dict["gpu_matvec_prob_uniform_v2_double"] = EncapsulateFunction(matvec_jitconn_prob_uniform_v2_double);
        dict["gpu_matvec_prob_normal_v2_float"] = EncapsulateFunction(matvec_jitconn_prob_normal_v2_float);
        dict["gpu_matvec_prob_normal_v2_double"] = EncapsulateFunction(matvec_jitconn_prob_normal_v2_double);

        // OP: mat (with jitconn) @ vector atomic V2
        dict["gpu_matvec_atomic_prob_homo_v2_float"] = EncapsulateFunction(matvec_atomic_jitconn_prob_homo_v2_float);
        dict["gpu_matvec_atomic_prob_homo_v2_double"] = EncapsulateFunction(matvec_atomic_jitconn_prob_homo_v2_double);
        dict["gpu_matvec_atomic_prob_uniform_v2_float"] = EncapsulateFunction(matvec_atomic_jitconn_prob_uniform_v2_float);
        dict["gpu_matvec_atomic_prob_uniform_v2_double"] = EncapsulateFunction(matvec_atomic_jitconn_prob_uniform_v2_double);
        dict["gpu_matvec_atomic_prob_normal_v2_float"] = EncapsulateFunction(matvec_atomic_jitconn_prob_normal_v2_float);
        dict["gpu_matvec_atomic_prob_normal_v2_double"] = EncapsulateFunction(matvec_atomic_jitconn_prob_normal_v2_double);

        // OP: X @ mat (with jitconn)
        dict["gpu_matmat_prob_prob_normal_float_v3"] = EncapsulateFunction(jitconn_matmat_prob_normal_float_v3);
        dict["gpu_matmat_prob_prob_normal_double_v3"] = EncapsulateFunction(jitconn_matmat_prob_normal_double_v3);
        dict["gpu_matmat_prob_prob_normal_float_v2"] = EncapsulateFunction(jitconn_matmat_prob_normal_float_v2);
        dict["gpu_matmat_prob_prob_normal_double_v2"] = EncapsulateFunction(jitconn_matmat_prob_normal_double_v2);
        dict["gpu_matmat_prob_prob_normal_float_v1"] = EncapsulateFunction(jitconn_matmat_prob_normal_float_v1);
        dict["gpu_matmat_prob_prob_normal_double_v1"] = EncapsulateFunction(jitconn_matmat_prob_normal_double_v1);
        dict["gpu_matmat_prob_prob_uniform_float_v1"] = EncapsulateFunction(jitconn_matmat_prob_uniform_float_v1);
        dict["gpu_matmat_prob_prob_uniform_double_v1"] = EncapsulateFunction(jitconn_matmat_prob_uniform_double_v1);

        return dict;
    }

    PYBIND11_MODULE(gpu_ops, m) {
    m.def("registrations", &Registrations);
    m.def("build_csr_event_sum_descriptor", &build_csr_event_sum_descriptor);
    m.def("build_coo_event_sum_descriptor", &build_coo_event_sum_descriptor);
    m.def("build_coo_atomic_sum_descriptor", &build_coo_atomic_sum_descriptor);
    m.def("build_coo_atomic_prod_descriptor", &build_coo_atomic_prod_descriptor);
    m.def("build_matmul_descriptor", &build_matmul_descriptor);
    m.def("build_mmm_descriptor", &build_mmm_descriptor);
    m.def("build_nonzero_descriptor", &build_nonzero_descriptor);
    m.def("build_jitconn_prob_homo_descriptor", &build_jitconn_prob_homo_descriptor);
    m.def("build_jitconn_prob_uniform_descriptor", &build_jitconn_prob_uniform_descriptor);
    m.def("build_jitconn_prob_normal_descriptor", &build_jitconn_prob_normal_descriptor);
    m.def("build_single_size_descriptor", &build_single_size_descriptor);
    m.def("build_double_size_descriptor", &build_double_size_descriptor);
    m.def("build_triple_size_descriptor", &build_triple_size_descriptor);
    m.def("build_twouint_onebool_descriptor", &build_twouint_onebool_descriptor);
    m.def("build_onefloat_descriptor", &build_onefloat_descriptor);
    m.def("build_twofloat_descriptor", &build_twofloat_descriptor);
    m.def("build_threefloat_descriptor", &build_threefloat_descriptor);
    m.def("build_matmat_jit_prob_descriptor1", &build_matmat_jit_prob_descriptor1);
}
}  // namespace
