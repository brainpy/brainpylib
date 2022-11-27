# -*- coding: utf-8 -*-

__all__ = [
  'event_csr_matvec'
]

from functools import partial
from typing import Union, Tuple

import jax.numpy as jnp
import numba
import numpy as np
from jax import vmap
from jax.lib import xla_client
from jax.core import ShapedArray, Primitive
from jax.interpreters import ad, xla

from brainpylib.op_register import compile_cpu_signature_with_numba
from brainpylib.sparse_ops.cusparse_matvec import cusparse_csr_matvec
from brainpylib.sparse_ops.utils import csr_to_coo

try:
  from brainpylib import gpu_ops
except ImportError:
  gpu_ops = None


def event_csr_matvec(
    values: Union[float, jnp.ndarray],
    indices: jnp.ndarray,
    indptr: jnp.ndarray,
    events: jnp.ndarray,
    *,
    shape: Tuple[int, int],
    transpose: bool = False
) -> jnp.ndarray:
  """Product of a sparse CSR matrix and a dense event vector.

  Parameters
  ----------
  values: ndarray, float
    An array of shape ``(nse,)``.
  indices: ndarray
    An array of shape ``(nse,)``.
  indptr: ndarray
    An array of shape ``(shape[0] + 1,)`` and dtype ``indices.dtype``.
  events: ndarray
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
  values = jnp.atleast_1d(values)
  if np.ndim(values) == 1:
    if values.shape[0] not in [1, indices.shape[0]]:
      raise ValueError('The size of values should be 1 or be consistent with indices.'
                       f'But we got {values.shape} != {indices.shape}, {values.shape} != 1.')
  else:
    raise ValueError('values should be a scalar or 1D vector. '
                     f'But we got {np.ndim(values)}-D array.')
  if np.ndim(indices) != 1:
    raise ValueError('indices should be a 1D vector with integer type.')
  if np.ndim(indptr) != 1:
    raise ValueError('indptr should be a 1D vector with integer type.')
  if not jnp.issubdtype(indices.dtype, jnp.integer):
    raise ValueError('indices should be a 1D vector with integer type.')
  if not jnp.issubdtype(indptr.dtype, jnp.integer):
    raise ValueError('indptr should be a 1D vector with integer type.')
  if np.ndim(events) != 1:
    raise ValueError('events should be a 1D vector.')
  if len(shape) != 2:
    raise ValueError('shape should be a length-2 tuple.')
  if transpose:
    if events.shape[0] != shape[0]:
      raise ValueError(f'Shape mismatch, vec ({events.shape[0]},) @ mat {shape}.')
  else:
    if events.shape[0] != shape[1]:
      raise ValueError(f'Shape mismatch, mat {shape} @ vec ({events.shape[0]},).')
  assert indptr.shape[0] == shape[0] + 1
  assert events.shape[0] == (shape[0] if transpose else shape[1])

  # computing
  return event_csr_matvec_p.bind(values, indices, indptr, events, shape=shape, transpose=transpose)


# ----------------------------------------------------------
# event csr matvec
# ----------------------------------------------------------

# operator for `event_csr_matvec` batching rule
# --------

def _batch_event_csr_matvec_abstract(
    values, indices, indptr, events, *, batch_size, shape, transpose=False
):
  return ShapedArray(dtype=values.dtype, shape=(batch_size, shape[1] if transpose else shape[0]))


@numba.njit(fastmath=True, parallel=True, nogil=True)
def _batch_event_csr_matvec_transpose_numba_imp(outs, ins):
  res_val = outs
  res_val.fill(0)
  values, indices, indptr, events, batch_size, shape, _ = ins
  batch_size = batch_size[()]
  event_batch_dim = events.shape[0]
  indices_batch_dim = indices.shape[0]
  indptr_batch_dim = indptr.shape[0]
  values_batch_dim = values.shape[0]

  if values.shape[1] == 1:  # homogeneous value
    for bi in numba.prange(batch_size):
      event_bi = bi % event_batch_dim
      indptr_bi = bi % indptr_batch_dim
      indices_bi = bi % indices_batch_dim
      values_bi = bi % values_batch_dim
      for row_i in range(shape[0]):
        if events[event_bi, row_i]:
          value = values[values_bi, 0]
          for j in range(indptr[indptr_bi, row_i], indptr[indptr_bi, row_i + 1]):
            col_i = indices[indices_bi, j]
            res_val[bi, col_i] += value

  else:  # heterogeneous values
    for bi in numba.prange(batch_size):
      event_bi = bi % event_batch_dim
      indptr_bi = bi % indptr_batch_dim
      indices_bi = bi % indices_batch_dim
      value_bi = bi % values_batch_dim
      for row_i in range(shape[0]):
        if events[event_bi, row_i]:
          for j in range(indptr[indptr_bi, row_i], indptr[indptr_bi, row_i + 1]):
            col_i = indices[indices_bi, j]
            res_val[bi, col_i] += values[value_bi, j]


@numba.njit(fastmath=True, parallel=True, nogil=True)
def _batch_event_csr_matvec_numba_imp(outs, ins):
  res_val = outs
  res_val.fill(0)
  values, indices, indptr, events, batch_size, shape, transpose = ins
  batch_size = batch_size[()]
  event_batch_dim = events.shape[0]
  indices_batch_dim = indices.shape[0]
  indptr_batch_dim = indptr.shape[0]
  values_batch_dim = values.shape[0]

  if values.shape[1] == 1:  # homogeneous value
    for bi in numba.prange(batch_size):
      event_bi = bi % event_batch_dim
      indptr_bi = bi % indptr_batch_dim
      indices_bi = bi % indices_batch_dim
      value_bi = bi % values_batch_dim
      value = values[value_bi, 0]
      for row_i in numba.prange(shape[0]):
        r = 0.
        for j in range(indptr[indptr_bi, row_i], indptr[indptr_bi, row_i + 1]):
          col_i = indices[indices_bi, j]
          if events[event_bi, col_i]:
            r += value
        res_val[bi, row_i] = r

  else:  # heterogeneous values
    for bi in numba.prange(batch_size):
      event_bi = bi % event_batch_dim
      indptr_bi = bi % indptr_batch_dim
      indices_bi = bi % indices_batch_dim
      value_bi = bi % values_batch_dim
      for row_i in numba.prange(shape[0]):
        r = 0.
        for j in range(indptr[indptr_bi, row_i], indptr[indptr_bi, row_i + 1]):
          col_i = indices[indices_bi, j]
          if events[event_bi, col_i]:
            r += values[value_bi, j]
        res_val[bi, row_i] = r


def _batch_event_csr_matvec_cpu_translation(c, values, indices, indptr, events, *,
                                            batch_size, shape, transpose):
  if transpose:
    name, inputs, in_layouts, out_layouts = compile_cpu_signature_with_numba(
      c,
      _batch_event_csr_matvec_transpose_numba_imp,
      _batch_event_csr_matvec_abstract,
      values, indices, indptr, events,
      batch_size=batch_size,
      shape=shape,
      transpose=transpose
    )
  else:
    name, inputs, in_layouts, out_layouts = compile_cpu_signature_with_numba(
      c,
      _batch_event_csr_matvec_numba_imp,
      _batch_event_csr_matvec_abstract,
      values, indices, indptr, events,
      batch_size=batch_size,
      shape=shape,
      transpose=transpose
    )
  return xla_client.ops.CustomCallWithLayout(
    c,
    name,
    operands=inputs,
    operand_shapes_with_layout=in_layouts,
    shape_with_layout=out_layouts,
  )


def _batch_event_csr_matvec_jvp_values(values_dot, values, indices, indptr, events, *,
                                       batch_size, shape, transpose):
  return event_csr_matvec_batching_p.bind(values_dot, indices, indptr, events,
                                          batch_size=batch_size, shape=shape, transpose=transpose)


def _batch_csr_matvec(values, indices, indptr, vectors, *, shape, transpose):
  f = vmap(partial(cusparse_csr_matvec, shape=shape, transpose=transpose),
           in_axes=(0 if values.shape[0] > 1 else None,
                    0 if indices.shape[0] > 1 else None,
                    0 if indptr.shape[0] > 1 else None,
                    0 if vectors.shape[0] > 1 else None))
  return f(values if values.shape[0] > 1 else values[0],
           indices if indices.shape[0] > 1 else indices[0],
           indptr if indptr.shape[0] > 1 else indptr[0],
           vectors if vectors.shape[0] > 1 else vectors[0])


def _batch_event_csr_matvec_jvp_events(events_dot, values, indices, indptr, events, *,
                                       batch_size, shape, transpose):
  return _batch_csr_matvec(values, indices, indptr, events_dot,
                           shape=shape, transpose=transpose)


def _f(ct, indices, indptr, events, *, transpose):
  row, col = csr_to_coo(indices, indptr)
  ct_values = events[row] * ct[col] if transpose else events[col] * ct[row]
  return ct_values


def _batch_event_csr_matvec_transpose(ct, values, indices, indptr, events, *,
                                      batch_size, shape, transpose):
  if ad.is_undefined_primal(indices) or ad.is_undefined_primal(indptr):
    raise ValueError("Cannot transpose with respect to sparse indices.")

  if ad.is_undefined_primal(events):
    ct_events = (ad.Zero(events.aval) if type(ct) is ad.Zero else
                 _batch_csr_matvec(ct, indices, indptr, values,
                                   shape=shape, transpose=not transpose))
    return values, indices, indptr, ct_events
  else:
    if values.aval.shape[1] == 1:  # scalar
      temp = event_csr_matvec_batching_p.bind(jnp.ones((1, 1)), indices, indptr, events,
                                              batch_size=batch_size, shape=shape,
                                              transpose=transpose)
      ct_values = vmap(jnp.inner)(ct, temp)
    else:  # heterogeneous values
      if type(ct) is ad.Zero:
        ct_values = ad.Zero(values.aval)
      else:
        f = vmap(partial(_f, transpose=transpose),
                 in_axes=(0,
                          0 if indices.shape[0] > 1 else None,
                          0 if indptr.shape[0] > 1 else None,
                          0 if events.shape[0] > 1 else None))
        ct_values = f(ct,
                      indices if indices.shape[0] > 1 else indices[0],
                      indptr if indptr.shape[0] > 1 else indptr[0],
                      events if events.shape[0] > 1 else events[0])
    return ct_values, indices, indptr, events


event_csr_matvec_batching_p = Primitive('event_csr_matvec_batching')
event_csr_matvec_batching_p.def_abstract_eval(_batch_event_csr_matvec_abstract)
event_csr_matvec_batching_p.def_impl(partial(xla.apply_primitive, event_csr_matvec_batching_p))
xla.backend_specific_translations['cpu'][event_csr_matvec_batching_p] = None
ad.defjvp(event_csr_matvec_batching_p, _batch_event_csr_matvec_jvp_values,
          None, None, _batch_event_csr_matvec_jvp_events)
ad.primitive_transposes[event_csr_matvec_batching_p] = _batch_event_csr_matvec_transpose


# operator for `event_csr_matvec` #
# ------------------------------- #


def _event_csr_matvec_abstract(values, indices, indptr, events, *, shape, transpose=False):
  return ShapedArray(dtype=values.dtype, shape=(shape[1] if transpose else shape[0],))


@numba.njit(fastmath=True)
def _event_csr_matvec_transpose_numba_imp(outs, ins):
  res_val = outs
  res_val.fill(0)
  values, indices, indptr, events, shape, _ = ins
  if values.shape[0] > 1:  # heter
    for row_i in range(shape[0]):
      if events[row_i]:
        for j in range(indptr[row_i], indptr[row_i + 1]):
          col_i = indices[j]
          res_val[col_i] += values[j]

  else:  # homo
    values = values[0]
    for row_i in range(shape[0]):
      if events[row_i]:
        for j in range(indptr[row_i], indptr[row_i + 1]):
          col_i = indices[j]
          res_val[col_i] += values


@numba.njit(fastmath=True, parallel=True, nogil=True)
def _event_csr_matvec_numba_imp(outs, ins):
  res_val = outs
  res_val.fill(0)
  values, indices, indptr, events, shape, _ = ins

  if values.shape[0] > 1:  # heter
    for row_i in range(shape[0]):
      r = 0.
      for j in range(indptr[row_i], indptr[row_i + 1]):
        col_i = indices[j]
        if events[col_i]:
          r += values[j]
      res_val[row_i] = r

  else:  # homo
    values = values[0]
    for row_i in numba.prange(shape[0]):
      r = 0.
      for j in range(indptr[row_i], indptr[row_i + 1]):
        col_i = indices[j]
        if events[col_i]:
          r += values
      res_val[row_i] = r


def _event_csr_matvec_cpu_translation(c, values, indices, indptr, events, *, shape, transpose):
  if transpose:
    name, inputs, in_layouts, out_layouts = compile_cpu_signature_with_numba(
      c,
      _event_csr_matvec_transpose_numba_imp,
      _event_csr_matvec_abstract,
      values, indices, indptr, events,
      shape=shape, transpose=transpose
    )
  else:
    name, inputs, in_layouts, out_layouts = compile_cpu_signature_with_numba(
      c,
      _event_csr_matvec_numba_imp,
      _event_csr_matvec_abstract,
      values, indices, indptr, events,
      shape=shape, transpose=transpose
    )
  return xla_client.ops.CustomCallWithLayout(
    c, name,
    operands=inputs,
    operand_shapes_with_layout=in_layouts,
    shape_with_layout=out_layouts,
  )


def _event_csr_matvec_batching_rule(args, axes, *, shape, transpose):
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

  r = event_csr_matvec_batching_p.bind(*args_processed,
                                       batch_size=batch_size,
                                       shape=shape,
                                       transpose=transpose)
  return r, 0


def _event_csr_matvec_jvp_values(values_dot, values, indices, indptr, events, *, shape, transpose):
  return event_csr_matvec(values_dot, indices, indptr, events, shape=shape, transpose=transpose)


def _event_csr_matvec_jvp_events(events_dot, values, indices, indptr, events, *, shape, transpose):
  return cusparse_csr_matvec(values, indices, indptr, events_dot, shape=shape, transpose=transpose)


def _event_csr_matvec_transpose_events(ct, values, indices, indptr, events, *, shape, transpose):
  ct_events = (ad.Zero(events) if type(ct) is ad.Zero else
               cusparse_csr_matvec(ct, indices, indptr, values, shape=shape, transpose=not transpose))
  return values, indices, indptr, ct_events


def _event_csr_matvec_transpose_values(ct, values, indices, indptr, events, *, shape, transpose):
  if values.shape[0] == 1:  # scalar
    ct_values = jnp.inner(ct, event_csr_matvec(jnp.ones(1), indices, indptr, events, shape=shape, transpose=transpose))
  else:  # heterogeneous values
    if type(ct) is ad.Zero:
      ct_values = ad.Zero(values)
    else:
      row, col = csr_to_coo(indices, indptr)
      ct_values = events[row] * ct[col] if transpose else events[col] * ct[row]
  return ct_values, indices, indptr, events


def _event_csr_matvec_transpose(ct, values, indices, indptr, events, *, shape, transpose):
  if ad.is_undefined_primal(indices) or ad.is_undefined_primal(indptr):
    raise ValueError("Cannot transpose with respect to sparse indices.")
  if ad.is_undefined_primal(events):
    return _event_csr_matvec_transpose_events(ct, values, indices, indptr, events.aval,
                                              shape=shape, transpose=transpose)
  else:
    return _event_csr_matvec_transpose_values(ct, values.aval, indices, indptr, events,
                                              shape=shape, transpose=transpose)


event_csr_matvec_p = Primitive('event_csr_matvec')
event_csr_matvec_p.def_abstract_eval(_event_csr_matvec_abstract)
event_csr_matvec_p.def_impl(partial(xla.apply_primitive, event_csr_matvec_p))
xla.backend_specific_translations['cpu'][event_csr_matvec_p] = _event_csr_matvec_cpu_translation
ad.defjvp(event_csr_matvec_p, _event_csr_matvec_jvp_values, None, None, _event_csr_matvec_jvp_events)
ad.primitive_transposes[event_csr_matvec_p] = _event_csr_matvec_transpose
