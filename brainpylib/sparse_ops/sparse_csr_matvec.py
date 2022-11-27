# -*- coding: utf-8 -*-


import numba
from jax import dtypes, numpy as jnp
from jax.core import ShapedArray
from brainpylib.op_register import register_op_with_numba
from brainpylib.base import BrainPyLibOp

try:
  from brainpylib import gpu_ops
except ImportError:
  gpu_ops = None

__all__ = [
  'CSRMatVec'
]


class CSRMatVec(BrainPyLibOp):
  def __init__(self, indices, indptr, method='vector'):
    assert method in ['scalar', 'vector', 'adaptive']

    self.method = method
    self.indices = indices
    self.indptr = indptr

    if method == 'adaptive':
      pass
      # TODO:
      self._op = csr_matvec_adaptive_p

    elif method == 'vector':
      self._op = csr_matvec_vector_p

    elif method == 'scalar':
      self._op = csr_matvec_scalar_p

    else:
      raise ValueError

  def __call__(self, data, vector, shape):
    if self.method == 'adaptive':
      self._op.bind(*args, **kwargs)

    else:
      self._op.bind(data, self.indices, self.indptr, vector, shape=shape)


def csr_matvec_scalar(data, indices, indptr, vector):
  pass


def _csr_matvec_scalar_abstract(data, indices, indptr, vector, *, shape):
  return ShapedArray(dtype=data.dtype, shape=(shape[1],))


@numba.njit(fastmath=True, parallel=True, nogil=True)
def _csr_matvec(outs, ins):
  data, indices, indptr, vector, shape = ins

  if len(data) == 1:
    data = data[0]
    for i in numba.prange(shape[0]):
      res = 0.
      for j in range(indptr[i], indptr[i + 1]):
        res += vector[indices[j]] * data
      outs[i] = res

  else:
    for i in numba.prange(shape[0]):
      res = 0.
      for j in range(indptr[i], indptr[i + 1]):
        res += vector[indices[j]] * data[j]
      outs[i] = res


csr_matvec_scalar_p = register_op_with_numba(
  'csr_matvec_scalr',
  cpu_func=_csr_matvec,
  out_shapes=_csr_matvec_scalar_abstract,
  apply_cpu_func_to_gpu=True if gpu_ops is None else False,
)


def _csr_matvec_vector_abstract(data, indices, indptr, vector, *, shape):
  return ShapedArray(dtype=data.dtype, shape=(shape[1],))


csr_matvec_vector_p = register_op_with_numba(
  'csr_matvec_vector',
  cpu_func=_csr_matvec,
  out_shapes=_csr_matvec_vector_abstract,
  apply_cpu_func_to_gpu=True if gpu_ops is None else False,
)


csr_matvec_adaptive_p = register_op_with_numba(
  'csr_matvec_adaptive',
  cpu_func=_csr_matvec,
  out_shapes=_csr_matvec_vector_abstract,
  apply_cpu_func_to_gpu=True if gpu_ops is None else False,
)
