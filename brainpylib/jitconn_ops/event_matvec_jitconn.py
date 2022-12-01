# -*- coding: utf-8 -*-


from functools import partial
from typing import Tuple, Optional

import numpy as np
from jax import numpy as jnp, dtypes
from jax.core import ShapedArray, Primitive
from jax.interpreters import xla
from jax.lib import xla_client

from brainpylib.errors import GPUOperatorNotFound
from brainpylib.event_ops.event_info_collection import event_info
from brainpylib.op_register import (register_general_batching)

try:
  from brainpylib import gpu_ops
except ImportError:
  gpu_ops = None

__all__ = [
  'event_matvec_prob_conn_homo_weight',
  'event_matvec_prob_conn_uniform_weight',
  'event_matvec_prob_conn_normal_weight',
]


def event_matvec_prob_conn_homo_weight(
    events: jnp.ndarray,
    weight: float,
    *,
    conn_prob: float,
    shape: Tuple[int, int],
    seed: Optional[int] = None,
    transpose: bool = False
) -> jnp.ndarray:
  # checking
  weight = jnp.atleast_1d(weight)
  if weight.size != 1:
    raise ValueError()
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

  # computing
  event_ids, event_num = event_info(events)
  return event_matvec_prob_conn_homo_weight_p.bind(
    event_ids, event_num, weight,
    conn_prob=conn_prob, shape=shape, seed=seed, transpose=transpose
  )


def event_matvec_prob_conn_uniform_weight(
    events: jnp.ndarray,
    *,
    weight: Tuple[float, float],
    conn_prob: float,
    shape: Tuple[int, int],
    seed: Optional[int] = None,
    transpose: bool = False
) -> jnp.ndarray:
  # checking
  if not isinstance(weight, (tuple, list)) and len(weight) != 2:
    raise TypeError('Must be a tuple/list with two elements.')
  weight_low, weight_high = weight
  assert weight_high > weight_low
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

  # computing
  event_ids, event_num = event_info(events)
  return event_matvec_prob_conn_uniform_weight_p.bind(
    event_ids, event_num,
    weight_low=weight_low, weight_range=weight_high - weight_low,
    conn_prob=conn_prob, shape=shape, seed=seed, transpose=transpose
  )


def event_matvec_prob_conn_normal_weight(
    events: jnp.ndarray,
    *,
    weight: Tuple[float, float],
    conn_prob: float,
    shape: Tuple[int, int],
    seed: Optional[int] = None,
    transpose: bool = False
) -> jnp.ndarray:
  # checking
  if not isinstance(weight, (tuple, list)) and len(weight) != 2:
    raise TypeError('Must be a tuple/list with two elements.')
  weight_mu, weight_sigma = weight
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

  # computing
  event_ids, event_num = event_info(events)
  return event_matvec_prob_conn_normal_weight_p.bind(
    event_ids, event_num,
    weight_mu=weight_mu, weight_sigma=weight_sigma,
    conn_prob=conn_prob, shape=shape, seed=seed, transpose=transpose
  )


def _event_matvec_prob_conn_homo_weight_abstract(
    event_ids, event_num, weight, *, conn_prob, shape, seed, transpose
):
  return ShapedArray(dtype=weight.dtype, shape=(shape[1] if transpose else shape[0],))


def _event_matvec_prob_conn_homo_weight_gpu_translation(
    c, event_ids, event_num, weight, *, conn_prob, shape, seed, transpose
):
  if gpu_ops is None:
    raise GPUOperatorNotFound(event_matvec_prob_conn_homo_weight_p.name)

  data_shape = c.get_shape(weight)
  type_name = b'_float' if data_shape.element_type() == jnp.float32 else b'_double'

  opaque = gpu_ops.build_jitconn_prob_homo_descriptor(shape[0], shape[1], seed, conn_prob, transpose)
  return xla_client.ops.CustomCallWithLayout(
    c,
    b'event_matvec_jitconn_prob_homo' + type_name,
    operands=(event_ids, event_num, weight),
    operand_shapes_with_layout=(c.get_shape(event_ids),
                                c.get_shape(event_num),
                                c.get_shape(weight)),
    shape_with_layout=xla_client.Shape.array_shape(data_shape.element_type(),
                                                   (shape[1] if transpose else shape[0],),
                                                   (0,)),
    opaque=opaque,
  )


event_matvec_prob_conn_homo_weight_p = Primitive('event_matvec_prob_conn_homo_weight')
event_matvec_prob_conn_homo_weight_p.def_abstract_eval(_event_matvec_prob_conn_homo_weight_abstract)
event_matvec_prob_conn_homo_weight_p.def_impl(partial(xla.apply_primitive, event_matvec_prob_conn_homo_weight_p))
xla.backend_specific_translations['gpu'][event_matvec_prob_conn_homo_weight_p] = \
  _event_matvec_prob_conn_homo_weight_gpu_translation
register_general_batching(event_matvec_prob_conn_homo_weight_p)


def _event_matvec_prob_conn_uniform_weight_abstract(
    event_ids, event_num, *,
    weight_low, weight_range, conn_prob, shape, seed, transpose
):
  return ShapedArray(dtype=dtypes.canonicalize_dtype(float),
                     shape=(shape[1] if transpose else shape[0],))


def _event_matvec_prob_conn_uniform_weight_gpu_translation(
    c, event_ids, event_num, *,
    weight_low, weight_range, conn_prob, shape, seed, transpose
):
  if gpu_ops is None:
    raise GPUOperatorNotFound(event_matvec_prob_conn_homo_weight_p.name)

  out_dtype = dtypes.canonicalize_dtype(float)
  type_name = b'_float' if out_dtype == jnp.float32 else b'_double'

  opaque = gpu_ops.build_jitconn_prob_uniform_descriptor(shape[0], shape[1], seed,
                                                         conn_prob,
                                                         weight_low,
                                                         weight_range,
                                                         transpose)
  return xla_client.ops.CustomCallWithLayout(
    c,
    b'event_matvec_jitconn_prob_uniform' + type_name,
    operands=(event_ids, event_num),
    operand_shapes_with_layout=(c.get_shape(event_ids),
                                c.get_shape(event_num)),
    shape_with_layout=xla_client.Shape.array_shape(out_dtype,
                                                   (shape[1] if transpose else shape[0],),
                                                   (0,)),
    opaque=opaque,
  )


event_matvec_prob_conn_uniform_weight_p = Primitive('event_matvec_prob_conn_uniform_weight')
event_matvec_prob_conn_uniform_weight_p.def_abstract_eval(_event_matvec_prob_conn_uniform_weight_abstract)
event_matvec_prob_conn_uniform_weight_p.def_impl(partial(xla.apply_primitive, event_matvec_prob_conn_uniform_weight_p))
xla.backend_specific_translations['gpu'][event_matvec_prob_conn_uniform_weight_p] = \
  _event_matvec_prob_conn_uniform_weight_gpu_translation
register_general_batching(event_matvec_prob_conn_uniform_weight_p)


def _event_matvec_prob_conn_normal_weight_abstract(
    event_ids, event_num, *,
    weight_mu, weight_sigma, conn_prob, shape, seed, transpose
):
  return ShapedArray(dtype=dtypes.canonicalize_dtype(float),
                     shape=(shape[1] if transpose else shape[0],))


def _event_matvec_prob_conn_normal_weight_gpu_translation(
    c, event_ids, event_num, *,
    weight_mu, weight_sigma, conn_prob, shape, seed, transpose
):
  if gpu_ops is None:
    raise GPUOperatorNotFound(event_matvec_prob_conn_homo_weight_p.name)

  out_dtype = dtypes.canonicalize_dtype(float)
  type_name = b'_float' if out_dtype == jnp.float32 else b'_double'

  opaque = gpu_ops.build_jitconn_prob_normal_descriptor(shape[0], shape[1], seed,
                                                        conn_prob,
                                                        weight_mu,
                                                        weight_sigma,
                                                        transpose)
  return xla_client.ops.CustomCallWithLayout(
    c,
    b'event_matvec_jitconn_prob_normal' + type_name,
    operands=(event_ids, event_num),
    operand_shapes_with_layout=(c.get_shape(event_ids),
                                c.get_shape(event_num)),
    shape_with_layout=xla_client.Shape.array_shape(out_dtype,
                                                   (shape[1] if transpose else shape[0],),
                                                   (0,)),
    opaque=opaque,
  )


event_matvec_prob_conn_normal_weight_p = Primitive('event_matvec_prob_conn_normal_weight')
event_matvec_prob_conn_normal_weight_p.def_abstract_eval(_event_matvec_prob_conn_normal_weight_abstract)
event_matvec_prob_conn_normal_weight_p.def_impl(partial(xla.apply_primitive, event_matvec_prob_conn_normal_weight_p))
xla.backend_specific_translations['gpu'][event_matvec_prob_conn_normal_weight_p] = \
  _event_matvec_prob_conn_normal_weight_gpu_translation
register_general_batching(event_matvec_prob_conn_normal_weight_p)
