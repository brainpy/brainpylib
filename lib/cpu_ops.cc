// This file defines the Python interface to the XLA custom call implemented on the CPU.
// It is exposed as a standard pybind11 module defining "capsule" objects containing our
// method. For simplicity, we export a separate capsule for each supported dtype.

#include "pybind11_kernel_helpers.h"
#include "cpu_event_sum.h"
#include "cpu_event_prod.h"
#include "cpu_atomic_sum.h"
#include "cpu_atomic_prod.h"
#include "cpu_jitconn_event_matvec.h"
#include "cpu_jitconn_matvec.h"
#include "cpu_jitconn_vecmat.h"

using namespace brainpy_lib;

namespace {
    pybind11::dict Registrations() {
      pybind11::dict dict;

      // event_sum for homogeneous value
      dict["cpu_csr_event_sum_homo_f32_i32"] = EncapsulateFunction(cpu_csr_event_sum_homo_f32_i32);
      dict["cpu_csr_event_sum_homo_f32_i64"] = EncapsulateFunction(cpu_csr_event_sum_homo_f32_i64);
      dict["cpu_csr_event_sum_homo_f64_i32"] = EncapsulateFunction(cpu_csr_event_sum_homo_f64_i32);
      dict["cpu_csr_event_sum_homo_f64_i64"] = EncapsulateFunction(cpu_csr_event_sum_homo_f64_i64);
      // event_sum for heterogeneous values
      dict["cpu_csr_event_sum_heter_f32_i32"] = EncapsulateFunction(cpu_csr_event_sum_heter_f32_i32);
      dict["cpu_csr_event_sum_heter_f32_i64"] = EncapsulateFunction(cpu_csr_event_sum_heter_f32_i64);
      dict["cpu_csr_event_sum_heter_f64_i32"] = EncapsulateFunction(cpu_csr_event_sum_heter_f64_i32);
      dict["cpu_csr_event_sum_heter_f64_i64"] = EncapsulateFunction(cpu_csr_event_sum_heter_f64_i64);

      // event_prod for homogeneous value
      dict["cpu_csr_event_prod_homo_f32_i32"] = EncapsulateFunction(cpu_csr_event_prod_homo_f32_i32);
      dict["cpu_csr_event_prod_homo_f32_i64"] = EncapsulateFunction(cpu_csr_event_prod_homo_f32_i64);
      dict["cpu_csr_event_prod_homo_f64_i32"] = EncapsulateFunction(cpu_csr_event_prod_homo_f64_i32);
      dict["cpu_csr_event_prod_homo_f64_i64"] = EncapsulateFunction(cpu_csr_event_prod_homo_f64_i64);
      // event_prod for heterogeneous values
      dict["cpu_csr_event_prod_heter_f32_i32"] = EncapsulateFunction(cpu_csr_event_prod_heter_f32_i32);
      dict["cpu_csr_event_prod_heter_f32_i64"] = EncapsulateFunction(cpu_csr_event_prod_heter_f32_i64);
      dict["cpu_csr_event_prod_heter_f64_i32"] = EncapsulateFunction(cpu_csr_event_prod_heter_f64_i32);
      dict["cpu_csr_event_prod_heter_f64_i64"] = EncapsulateFunction(cpu_csr_event_prod_heter_f64_i64);

      // atomic_sum for heterogeneous values
      dict["cpu_coo_atomic_sum_heter_f32_i32"] = EncapsulateFunction(cpu_coo_atomic_sum_heter_f32_i32);
      dict["cpu_coo_atomic_sum_heter_f32_i64"] = EncapsulateFunction(cpu_coo_atomic_sum_heter_f32_i64);
      dict["cpu_coo_atomic_sum_heter_f64_i32"] = EncapsulateFunction(cpu_coo_atomic_sum_heter_f64_i32);
      dict["cpu_coo_atomic_sum_heter_f64_i64"] = EncapsulateFunction(cpu_coo_atomic_sum_heter_f64_i64);
      // atomic_sum for homogeneous value
      dict["cpu_coo_atomic_sum_homo_f32_i32"] = EncapsulateFunction(cpu_coo_atomic_sum_homo_f32_i32);
      dict["cpu_coo_atomic_sum_homo_f32_i64"] = EncapsulateFunction(cpu_coo_atomic_sum_homo_f32_i64);
      dict["cpu_coo_atomic_sum_homo_f64_i32"] = EncapsulateFunction(cpu_coo_atomic_sum_homo_f64_i32);
      dict["cpu_coo_atomic_sum_homo_f64_i64"] = EncapsulateFunction(cpu_coo_atomic_sum_homo_f64_i64);
      
      // atomic_prod for heterogeneous values
      dict["cpu_coo_atomic_prod_heter_f32_i32"] = EncapsulateFunction(cpu_coo_atomic_prod_heter_f32_i32);
      dict["cpu_coo_atomic_prod_heter_f32_i64"] = EncapsulateFunction(cpu_coo_atomic_prod_heter_f32_i64);
      dict["cpu_coo_atomic_prod_heter_f64_i32"] = EncapsulateFunction(cpu_coo_atomic_prod_heter_f64_i32);
      dict["cpu_coo_atomic_prod_heter_f64_i64"] = EncapsulateFunction(cpu_coo_atomic_prod_heter_f64_i64);
      // atomic_prod for homogeneous value
      dict["cpu_coo_atomic_prod_homo_f32_i32"] = EncapsulateFunction(cpu_coo_atomic_prod_homo_f32_i32);
      dict["cpu_coo_atomic_prod_homo_f32_i64"] = EncapsulateFunction(cpu_coo_atomic_prod_homo_f32_i64);
      dict["cpu_coo_atomic_prod_homo_f64_i32"] = EncapsulateFunction(cpu_coo_atomic_prod_homo_f64_i32);
      dict["cpu_coo_atomic_prod_homo_f64_i64"] = EncapsulateFunction(cpu_coo_atomic_prod_homo_f64_i64);

      // OP: jitconn event-matvec ops
      dict["cpu_event_matvec_prob_homo_float_bool"] = EncapsulateFunction(event_matvec_prob_homo_float_bool);
      dict["cpu_event_matvec_prob_homo_float_float"] = EncapsulateFunction(event_matvec_prob_homo_float_float);
      dict["cpu_event_matvec_prob_homo_double_bool"] = EncapsulateFunction(event_matvec_prob_homo_double_bool);
      dict["cpu_event_matvec_prob_homo_double_double"] = EncapsulateFunction(event_matvec_prob_homo_double_double);

      dict["cpu_event_matvec_prob_uniform_float_bool"] = EncapsulateFunction(event_matvec_prob_uniform_float_bool);
      dict["cpu_event_matvec_prob_uniform_float_float"] = EncapsulateFunction(event_matvec_prob_uniform_float_float);
      dict["cpu_event_matvec_prob_uniform_double_bool"] = EncapsulateFunction(event_matvec_prob_uniform_double_bool);
      dict["cpu_event_matvec_prob_uniform_double_double"] = EncapsulateFunction(event_matvec_prob_uniform_double_double);

      dict["cpu_event_matvec_prob_normal_float_bool"] = EncapsulateFunction(event_matvec_prob_normal_float_bool);
      dict["cpu_event_matvec_prob_normal_float_float"] = EncapsulateFunction(event_matvec_prob_normal_float_float);
      dict["cpu_event_matvec_prob_normal_double_bool"] = EncapsulateFunction(event_matvec_prob_normal_double_bool);
      dict["cpu_event_matvec_prob_normal_double_double"] = EncapsulateFunction(event_matvec_prob_normal_double_double);

      // OP: jitconn matvec ops
      dict["cpu_matvec_prob_homo_float"] = EncapsulateFunction(matvec_prob_homo_float);
      dict["cpu_matvec_prob_homo_double"] = EncapsulateFunction(matvec_prob_homo_double);

      dict["cpu_matvec_prob_uniform_float"] = EncapsulateFunction(matvec_prob_uniform_float);
      dict["cpu_matvec_prob_uniform_double"] = EncapsulateFunction(matvec_prob_uniform_double);

      dict["cpu_matvec_prob_normal_float"] = EncapsulateFunction(matvec_prob_normal_float);
      dict["cpu_matvec_prob_normal_double"] = EncapsulateFunction(matvec_prob_normal_double);

      // OP: jitconn vecmat ops
      dict["cpu_vecmat_prob_homo_float"] = EncapsulateFunction(vecmat_prob_homo_float);
      dict["cpu_vecmat_prob_homo_double"] = EncapsulateFunction(vecmat_prob_homo_double);

      dict["cpu_vecmat_prob_uniform_float"] = EncapsulateFunction(vecmat_prob_uniform_float);
      dict["cpu_vecmat_prob_uniform_double"] = EncapsulateFunction(vecmat_prob_uniform_double);

      dict["cpu_vecmat_prob_normal_float"] = EncapsulateFunction(vecmat_prob_normal_float);
      dict["cpu_vecmat_prob_normal_double"] = EncapsulateFunction(vecmat_prob_normal_double);

      return dict;
    }

    PYBIND11_MODULE(cpu_ops, m) {
        m.def("registrations", &Registrations);
    }

}  // namespace
