# -*- coding: utf-8 -*-


import warnings
from functools import partial
from typing import Tuple, NamedTuple

import numba
import numpy as np
from jax import core, numpy as jnp
from jax.interpreters import ad, mlir
from jaxlib import gpu_sparse

from .custom_op import register_op_with_numba
from .utils import register_general_batching

__all__ = [
  'COOInfo', 'csr_to_coo', 'coo_to_csr',

  'csr_matvec',
  'coo_matvec',
]


def coo_to_csr(pre_ids, post_ids, num_pre):
  """convert pre_ids, post_ids to (indices, indptr)."""
  # sorting
  sort_ids = jnp.argsort(pre_ids, kind='stable')
  post_ids = post_ids[sort_ids]

  indices = post_ids
  unique_pre_ids, pre_count = jnp.unique(pre_ids, return_counts=True)
  final_pre_count = jnp.zeros(num_pre)
  final_pre_count[unique_pre_ids] = pre_count
  indptr = final_pre_count.cumsum()
  indptr = jnp.insert(indptr, 0, 0)
  return indices, indptr


def csr_to_coo(indices, indptr):
  """Given CSR (indices, indptr) return COO (row, col)"""
  return jnp.cumsum(jnp.zeros_like(indices).at[indptr].add(1)) - 1, indices


class COOInfo(NamedTuple):
  shape: Tuple[int, ...]
  rows_sorted: bool = False
  cols_sorted: bool = False


# --------------------------------------------------------------------
# csr_matvec
# --------------------------------------------------------------------


# operator for `csr_matvec` batching rule #

def _csr_matvec_numba_batching_abstract(
    data, indices, indptr, vector, *,
    batch_size, shape, transpose=False,
):
  return core.ShapedArray(dtype=data.dtype, shape=(batch_size, shape[1] if transpose else shape[0]))


@numba.njit(fastmath=True)
def _csr_matvec_numba_batching(outs, ins):
  res_val = outs
  res_val.fill(0)
  values, indices, indptr, vector, _, shape, transpose = ins
  batch_size = res_val.shape[0]
  event_batch_dim = vector.shape[0]
  indices_batch_dim = indices.shape[0]
  indptr_batch_dim = indptr.shape[0]
  values_batch_dim = values.shape[0]
  transpose = transpose[()]

  if transpose:  # vec @ csr mat
    num_pre = vector.shape[1]
    if values.shape[1] == 1:  # homogeneous value
      for bi in numba.prange(batch_size):
        event_bi = bi % event_batch_dim
        indptr_bi = bi % indptr_batch_dim
        indices_bi = bi % indices_batch_dim
        value_bi = bi % values_batch_dim
        for pre_i in range(num_pre):
          value = values[value_bi, 0] * vector[event_bi, pre_i]
          for syn_i in range(indptr[indptr_bi, pre_i + 1] - indptr[indptr_bi, pre_i]):
            post_i = indices[indices_bi, syn_i]
            res_val[bi, post_i] += value

    else:  # heterogeneous values
      for bi in numba.prange(batch_size):
        event_bi = bi % event_batch_dim
        indptr_bi = bi % indptr_batch_dim
        indices_bi = bi % indices_batch_dim
        values_bi = bi % values_batch_dim
        for pre_i in range(num_pre):
          v = vector[event_bi, pre_i]
          for syn_i in range(indptr[indptr_bi, pre_i + 1] - indptr[indptr_bi, pre_i]):
            post_i = indices[indices_bi, syn_i]
            res_val[bi, post_i] += values[values_bi, post_i] * v

  else:  # csr mat @ vec
    num_pre = shape[0]
    if values.shape[1] == 1:  # homogeneous value
      for bi in numba.prange(batch_size):
        event_bi = bi % event_batch_dim
        indptr_bi = bi % indptr_batch_dim
        indices_bi = bi % indices_batch_dim
        value_bi = bi % values_batch_dim
        for pre_i in range(num_pre):
          value = values[value_bi, 0]
          for syn_i in range(indptr[indptr_bi, pre_i + 1] - indptr[indptr_bi, pre_i]):
            post_i = indices[indices_bi, syn_i]
            res_val[bi, pre_i] += value * vector[event_bi, post_i]

    else:  # heterogeneous values
      for bi in numba.prange(batch_size):
        event_bi = bi % event_batch_dim
        indptr_bi = bi % indptr_batch_dim
        indices_bi = bi % indices_batch_dim
        values_bi = bi % values_batch_dim
        for pre_i in range(num_pre):
          for syn_i in range(indptr[indptr_bi, pre_i + 1] - indptr[indptr_bi, pre_i]):
            post_i = indices[indices_bi, syn_i]
            res_val[bi, pre_i] += values[values_bi, post_i] * vector[event_bi, post_i]


csr_matvec_numba_batching_p = register_op_with_numba(
  op_name='csr_matvec_numba_batching',
  cpu_func=_csr_matvec_numba_batching,
  out_shapes=_csr_matvec_numba_batching_abstract,
  apply_cpu_func_to_gpu=not gpu_sparse.cuda_is_supported,
)


# operator for `csr_matvec` #

def _csr_matvec_numba_abstract(data, indices, indptr, v, *, shape, transpose):
  out_shape = shape[1] if transpose else shape[0]
  return core.ShapedArray((out_shape,), data.dtype)


@numba.njit(fastmath=True)
def _csr_matvec_numba(outs, ins):
  res_val = outs
  res_val.fill(0)
  values, indices, indptr, vector, shape, transpose = ins

  if transpose:  # vec @ csr mat
    if values.shape[0] == 1:
      values = values[0]
      for pre_i in range(vector.shape[0]):
        v = vector[pre_i]
        for syn_j in numba.prange(indptr[pre_i + 1] - indptr[pre_i]):
          post_i = indices[syn_j]
          res_val[post_i] += values * v
    else:
      for pre_i in range(vector.shape[0]):
        v = vector[pre_i]
        for syn_j in numba.prange(indptr[pre_i + 1] - indptr[pre_i]):
          post_i = indices[syn_j]
          res_val[post_i] += values[post_i] * v

  else:  # csr mat @ vec
    if values.shape[0] == 1:
      values = values[0]
      for pre_i in range(shape[1]):
        for syn_j in numba.prange(indptr[pre_i + 1] - indptr[pre_i]):
          post_i = indices[syn_j]
          res_val[pre_i] += values * vector[post_i]
    else:
      for pre_i in range(shape[1]):
        for syn_j in numba.prange(indptr[pre_i + 1] - indptr[pre_i]):
          post_i = indices[syn_j]
          res_val[pre_i] += values[post_i] * vector[post_i]


def _csr_matvec_numba_batching_rule(args, axes, *, shape, transpose=False):
  batch_size = 0
  args_processed = []
  for arg, axis in zip(args, axes):
    if axis is None:
      arg = jnp.expand_dims(jnp.atleast_1d(arg), 0)
    else:
      batch_size = arg.shape[axis]
      if axis > 0:
        arg = jnp.moveaxis(arg, axis, 0)
    args_processed.append(arg)

  return csr_matvec_numba_batching_p.bind(
    *args_processed, batch_size=batch_size, shape=shape, transpose=transpose
  ), 0


csr_matvec_p = register_op_with_numba(
  op_name='csr_matvec',
  cpu_func=_csr_matvec_numba,
  out_shapes=_csr_matvec_numba_abstract,
  apply_cpu_func_to_gpu=not gpu_sparse.cuda_is_supported,
)


def csr_matvec(data, indices, indptr, vector, *, shape, transpose=False):
  """Product of CSR sparse matrix and a dense vector.

  Parameters
  ----------
  data: ndarray
    An array of shape ``(nse,)``.
  indices: ndarray
    An array of shape ``(nse,)``.
  indptr: ndarray
    An array of shape ``(shape[0] + 1,)`` and dtype ``indices.dtype``.
  vector: ndarray
    An array of shape ``(shape[0] if transpose else shape[1],)``
    and dtype ``data.dtype``.
  shape: tuple
    A length-2 tuple representing the matrix shape.
  transpose: bool
    A boolean specifying whether to transpose the sparse matrix
    before computing.

  Returns
  -------
  y : ndarry
    The array of shape ``(shape[1] if transpose else shape[0],)`` representing
    the matrix vector product.
  """
  # checking
  data = jnp.atleast_1d(data)
  assert len(shape) == 2
  assert vector.ndim == data.ndim == indices.ndim == indptr.ndim == 1
  if data.shape[0] not in [1, indices.shape[0]]:
    raise ValueError('The size of values should be 1 or be consistent with indices.'
                     f'But we got {data.shape} != {indices.shape}, {data.shape} != 1.')
  if not jnp.issubdtype(indices.dtype, jnp.integer):
    raise ValueError('indices should be a 1D vector with integer type.')
  if not jnp.issubdtype(indptr.dtype, jnp.integer):
    raise ValueError('indptr should be a 1D vector with integer type.')
  assert data.dtype == vector.dtype
  assert indptr.shape[0] == shape[0] + 1
  assert vector.shape[0] == (shape[0] if transpose else shape[1])
  # computing
  return csr_matvec_p.bind(data, indices, indptr, vector, shape=shape, transpose=transpose)


def _csr_matvec_gpu_lowering(csr_matvec_mhlo, ctx,
                             data, indices, indptr, v, *,
                             shape, transpose):
  data_aval, indices_aval, _, v_aval = ctx.avals_in
  dtype = data_aval.dtype
  if dtype not in [np.float32, np.float64, np.complex64, np.complex128]:
    raise TypeError(f"csr_matvec cusparse/hipsparse lowering not available for dtype={dtype}. "
                    "Falling back to default implementation.")
  return [csr_matvec_mhlo(data, indices, indptr, v,
                          shape=shape, transpose=transpose,
                          data_dtype=dtype, x_dtype=v_aval.dtype,
                          index_dtype=indices_aval.dtype)]


def _csr_matvec_jvp_mat(data_dot, data, indices, indptr, v, *, shape, transpose):
  return csr_matvec(data_dot, indices, indptr, v, shape=shape, transpose=transpose)


def _csr_matvec_jvp_vec(v_dot, data, indices, indptr, v, *, shape, transpose):
  return csr_matvec(data, indices, indptr, v_dot, shape=shape, transpose=transpose)


def _csr_matvec_transpose(ct, data, indices, indptr, v, *, shape, transpose):
  assert not ad.is_undefined_primal(indices)
  assert not ad.is_undefined_primal(indptr)

  if ad.is_undefined_primal(v):
    return data, indices, indptr, csr_matvec(data, indices, indptr, ct, shape=shape, transpose=not transpose)
  else:
    v = jnp.asarray(v)
    row, col = csr_to_coo(indices, indptr)
    return ct[row] * v[col], indices, indptr, v


register_general_batching(csr_matvec_p)
ad.defjvp(csr_matvec_p, _csr_matvec_jvp_mat, None, None, _csr_matvec_jvp_vec)
ad.primitive_transposes[csr_matvec_p] = _csr_matvec_transpose

if gpu_sparse.cuda_is_supported:
  mlir.register_lowering(
    csr_matvec_p,
    partial(_csr_matvec_gpu_lowering, gpu_sparse.cuda_csr_matvec),
    platform='cuda'
  )

if gpu_sparse.rocm_is_supported:
  mlir.register_lowering(
    csr_matvec_p,
    partial(_csr_matvec_gpu_lowering, gpu_sparse.rocm_csr_matvec),
    platform='rocm'
  )

# --------------------------------------------------------------------
# coo_matvec

coo_matvec_p = core.Primitive('coo_matvec')


def coo_matvec(data, row, col, v, *, spinfo: COOInfo, transpose=False):
  """Product of COO sparse matrix and a dense vector.

  Parameters
  ----------
  data: ndarray
    An array of shape ``(nse,)``.
  row: ndarray
    An array of shape ``(nse,)``.
  col: ndarray
    An array of shape ``(nse,)`` and dtype ``row.dtype``.
  v: ndarray
    An array of shape ``(shape[0] if transpose else shape[1],)`` and
    dtype ``data.dtype``.
  spinfo: COOInfo
    A tuple representing the matrix shape
  transpose: bool
    A boolean specifying whether to transpose the sparse matrix
    before computing.

  Returns
  -------
  y: ndarray
    An array of shape ``(shape[1] if transpose else shape[0],)`` representing
    the matrix vector product.
  """
  return coo_matvec_p.bind(data, row, col, v, spinfo=spinfo, transpose=transpose)


@coo_matvec_p.def_impl
def _coo_matvec_impl(data, row, col, v, *, spinfo, transpose):
  v = jnp.asarray(v)
  if transpose:
    row, col = col, row
  out_shape = spinfo.shape[1] if transpose else spinfo.shape[0]
  dv = data * v[col]
  return jnp.zeros(out_shape, dv.dtype).at[row].add(dv)


@coo_matvec_p.def_abstract_eval
def _coo_matvec_abstract_eval(data, row, col, v, *, spinfo, transpose):
  assert data.shape == row.shape == col.shape
  assert data.dtype == v.dtype
  assert row.dtype == col.dtype
  assert len(spinfo.shape) == 2
  assert v.ndim == 1
  assert v.shape[0] == (spinfo.shape[0] if transpose else spinfo.shape[1])
  out_shape = spinfo.shape[1] if transpose else spinfo.shape[0]
  return core.ShapedArray((out_shape,), data.dtype)


_coo_matvec_lowering = mlir.lower_fun(
  _coo_matvec_impl, multiple_results=False)


def _coo_matvec_gpu_lowering(coo_matvec_mhlo, ctx, data, row, col, v, *, spinfo,
                             transpose):
  data_aval, row_aval, _, x_aval = ctx.avals_in
  dtype = data_aval.dtype
  if dtype not in [np.float32, np.float64, np.complex64, np.complex128]:
    warnings.warn(f"coo_matvec cusparse/hipsparse lowering not available for dtype={dtype}. "
                  "Falling back to default implementation.", UserWarning)
    return _coo_matvec_lowering(ctx, data, row, col, v, spinfo=spinfo,
                                transpose=transpose)

  if spinfo.rows_sorted:
    shape = spinfo.shape
  elif spinfo.cols_sorted:
    row, col = col, row
    transpose = not transpose
    shape = spinfo.shape[::-1]
  else:
    warnings.warn("coo_matvec GPU lowering requires matrices with sorted rows or sorted cols. "
                  "To sort the rows in your matrix, use e.g. mat = mat._sort_rows(). Falling "
                  "back to the default implementation.", UserWarning)
    return _coo_matvec_lowering(ctx, data, row, col, v, spinfo=spinfo,
                                transpose=transpose)

  return [coo_matvec_mhlo(
    data, row, col, v, shape=shape, transpose=transpose,
    index_dtype=row_aval.dtype, data_dtype=dtype, x_dtype=x_aval.dtype)]


def _coo_matvec_jvp_mat(data_dot, data, row, col, v, *, spinfo, transpose):
  return coo_matvec(data_dot, row, col, v, spinfo=spinfo, transpose=transpose)


def _coo_matvec_jvp_vec(v_dot, data, row, col, v, *, spinfo, transpose):
  return coo_matvec(data, row, col, v_dot, spinfo=spinfo, transpose=transpose)


def _coo_matvec_transpose(ct, data, row, col, v, *, spinfo, transpose):
  assert not ad.is_undefined_primal(row)
  assert not ad.is_undefined_primal(col)

  if ad.is_undefined_primal(v):
    return data, row, col, coo_matvec(data, row, col, ct, spinfo=spinfo, transpose=not transpose)
  else:
    v = jnp.asarray(v)
    return ct[row] * v[col], row, col, v


register_general_batching(coo_matvec_p)
ad.defjvp(coo_matvec_p, _coo_matvec_jvp_mat, None, None, _coo_matvec_jvp_vec)
ad.primitive_transposes[coo_matvec_p] = _coo_matvec_transpose
mlir.register_lowering(coo_matvec_p, _coo_matvec_lowering)
if gpu_sparse.cuda_is_supported:
  mlir.register_lowering(
    coo_matvec_p,
    partial(_coo_matvec_gpu_lowering, gpu_sparse.cuda_coo_matvec),
    platform='cuda'
  )
if gpu_sparse.rocm_is_supported:
  mlir.register_lowering(
    coo_matvec_p,
    partial(_coo_matvec_gpu_lowering, gpu_sparse.rocm_coo_matvec),
    platform='rocm'
  )

