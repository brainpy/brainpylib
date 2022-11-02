# -*- coding: utf-8 -*-

__all__ = [
  'event_csr_matvec'
]

from functools import partial
import jax.numpy as jnp
import numba
import numpy as np
from jax import vmap
from jax.core import ShapedArray
from jax.interpreters import ad

from .custom_op import register_op_with_numba
from .sparse_matmul import csr_matvec, csr_to_coo

try:
  from . import gpu_ops
except ImportError:
  gpu_ops = None


# ----------------------------------------------------------
# event csr matvec
# ----------------------------------------------------------

# operator for `event_csr_matvec` batching rule
# --------

def _event_csr_matvec_batching_abstract(
    values, indices, indptr, events, *, batch_size, shape, transpose=False
):
  return ShapedArray(dtype=values.dtype, shape=(batch_size, shape[1] if transpose else shape[0]))


@numba.njit(fastmath=True)
def _event_csr_matvec_batching(outs, ins):
  res_val = outs
  res_val.fill(0)
  values, indices, indptr, events, _, _, transpose = ins
  batch_size = res_val.shape[0]
  event_batch_dim = events.shape[0]
  indices_batch_dim = indices.shape[0]
  indptr_batch_dim = indptr.shape[0]
  values_batch_dim = values.shape[0]
  transpose = transpose[()]

  if transpose:
    num_pre = events.shape[1]
    if values.shape[1] == 1:  # homogeneous value
      for bi in numba.prange(batch_size):
        event_bi = bi % event_batch_dim
        indptr_bi = bi % indptr_batch_dim
        indices_bi = bi % indices_batch_dim
        values_bi = bi % values_batch_dim
        for pre_i in range(num_pre):
          if events[event_bi, pre_i]:
            value = values[values_bi, 0]
            for syn_i in range(indptr[indptr_bi, pre_i], indptr[indptr_bi, pre_i + 1]):
              post_i = indices[indices_bi, syn_i]
              res_val[bi, post_i] += value

    else:  # heterogeneous values
      for bi in numba.prange(batch_size):
        event_bi = bi % event_batch_dim
        indptr_bi = bi % indptr_batch_dim
        indices_bi = bi % indices_batch_dim
        value_bi = bi % values_batch_dim
        for pre_i in range(num_pre):
          if events[event_bi, pre_i]:
            for syn_i in range(indptr[indptr_bi, pre_i], indptr[indptr_bi, pre_i + 1]):
              post_i = indices[indices_bi, syn_i]
              res_val[bi, post_i] += values[value_bi, post_i]

  else:
    num_pre = indptr.shape[1] - 1
    if values.shape[1] == 1:  # homogeneous value
      for bi in numba.prange(batch_size):
        event_bi = bi % event_batch_dim
        indptr_bi = bi % indptr_batch_dim
        indices_bi = bi % indices_batch_dim
        value_bi = bi % values_batch_dim
        for pre_i in range(num_pre):
          value = values[value_bi, 0]
          for syn_i in range(indptr[indptr_bi, pre_i], indptr[indptr_bi, pre_i + 1]):
            post_i = indices[indices_bi, syn_i]
            if events[event_bi, post_i]:
              res_val[bi, pre_i] += value

    else:  # heterogeneous values
      for bi in numba.prange(batch_size):
        event_bi = bi % event_batch_dim
        indptr_bi = bi % indptr_batch_dim
        indices_bi = bi % indices_batch_dim
        value_bi = bi % values_batch_dim
        for pre_i in range(num_pre):
          for syn_i in range(indptr[indptr_bi, pre_i], indptr[indptr_bi, pre_i + 1]):
            post_i = indices[indices_bi, syn_i]
            if events[event_bi, post_i]:
              res_val[bi, pre_i] += values[value_bi, post_i]


event_csr_matvec_batching_p = register_op_with_numba(
  op_name='event_csr_matvec_batching',
  cpu_func=_event_csr_matvec_batching,
  out_shapes=_event_csr_matvec_batching_abstract,
  apply_cpu_func_to_gpu=True if gpu_ops is None else False,
)


def _event_csr_matvec_batching_jvp_values(values_dot, values, indices, indptr, events, *,
                                          batch_size, shape, transpose):
  return event_csr_matvec_batching_p.bind(values_dot, indices, indptr, events,
                                          batch_size=batch_size, shape=shape, transpose=transpose)


def _batch_csr_matvec(values, indices, indptr, vectors, *, shape, transpose):
  f = vmap(partial(csr_matvec, shape=shape, transpose=transpose),
           in_axes=(0 if values.shape[0] > 1 else None,
                    0 if indices.shape[0] > 1 else None,
                    0 if indptr.shape[0] > 1 else None,
                    0 if vectors.shape[0] > 1 else None))
  return f(values if values.shape[0] > 1 else values[0],
           indices if indices.shape[0] > 1 else indices[0],
           indptr if indptr.shape[0] > 1 else indptr[0],
           vectors if vectors.shape[0] > 1 else vectors[0])


def _event_csr_matvec_batching_jvp_events(events_dot, values, indices, indptr, events, *,
                                          batch_size, shape, transpose):
  return _batch_csr_matvec(values, indices, indptr, events_dot,
                           shape=shape, transpose=transpose)


def _f(ct, indices, indptr, events, *, transpose):
  row, col = csr_to_coo(indices, indptr)
  ct_values = events[row] * ct[col] if transpose else events[col] * ct[row]
  return ct_values


def _event_csr_matvec_batching_transpose(ct, values, indices, indptr, events, *,
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


ad.defjvp(event_csr_matvec_batching_p,
          _event_csr_matvec_batching_jvp_values,
          None, None,
          _event_csr_matvec_batching_jvp_events)
ad.primitive_transposes[event_csr_matvec_batching_p] = _event_csr_matvec_batching_transpose


# operator for `event_csr_matvec` #
# ------------------------------- #

def _event_csr_matvec_abstract(values, indices, indptr, events, *, shape, transpose=False):
  return ShapedArray(dtype=values.dtype, shape=(shape[1] if transpose else shape[0],))


@numba.njit(fastmath=True)
def _event_csr_matvec(outs, ins):
  res_val = outs
  res_val.fill(0)
  values, indices, indptr, events, shape, transpose = ins
  transpose = transpose[()]

  if transpose:
    if values.shape[0] > 1:  # heter
      for pre_i in range(events.shape[0]):
        if events[pre_i]:
          for syn_i in numba.prange(indptr[pre_i], indptr[pre_i + 1]):
            post_i = indices[syn_i]
            res_val[post_i] += values[post_i]

    else:  # homo
      values = values[0]
      for pre_i in range(events.shape[0]):
        if events[pre_i]:
          for syn_i in numba.prange(indptr[pre_i], indptr[pre_i + 1]):
            post_i = indices[syn_i]
            res_val[post_i] += values

  else:
    if values.shape[0] > 1:  # heter
      for pre_i in range(shape[0]):
        for syn_i in numba.prange(indptr[pre_i], indptr[pre_i + 1]):
          post_i = indices[syn_i]
          if events[post_i]:
            res_val[pre_i] += values[post_i]

    else:  # homo
      values = values[0]
      for pre_i in range(events.shape[0]):
        for syn_i in numba.prange(indptr[pre_i], indptr[pre_i + 1]):
          post_i = indices[syn_i]
          if events[post_i]:
            res_val[pre_i] += values


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

  return event_csr_matvec_batching_p.bind(
    *args_processed, batch_size=batch_size, shape=shape, transpose=transpose
  ), 0


event_csr_matvec_p = register_op_with_numba(
  op_name='event_csr_matvec',
  cpu_func=_event_csr_matvec,
  out_shapes=_event_csr_matvec_abstract,
  batching_translation=_event_csr_matvec_batching_rule,
  apply_cpu_func_to_gpu=True if gpu_ops is None else False,
)


def _event_csr_matvec_jvp_values(values_dot, values, indices, indptr, events, *, shape, transpose):
  return event_csr_matvec(values_dot, indices, indptr, events, shape=shape, transpose=transpose)


def _event_csr_matvec_jvp_events(events_dot, values, indices, indptr, events, *, shape, transpose):
  return csr_matvec(values, indices, indptr, events_dot, shape=shape, transpose=transpose)


def _event_csr_matvec_transpose_events(ct, values, indices, indptr, events, *, shape, transpose):
  ct_events = (ad.Zero(events) if type(ct) is ad.Zero else
               csr_matvec(ct, indices, indptr, values, shape=shape, transpose=not transpose))
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


# autodiff
ad.defjvp(event_csr_matvec_p,
          _event_csr_matvec_jvp_values,
          None,
          None,
          _event_csr_matvec_jvp_events)
ad.primitive_transposes[event_csr_matvec_p] = _event_csr_matvec_transpose


def event_csr_matvec(values, indices, indptr, events, *, shape, transpose=False):
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
