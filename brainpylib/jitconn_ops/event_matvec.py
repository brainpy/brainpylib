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
  'event_matvec_prob_conn_homo_weight_v1',
  'event_matvec_prob_conn_uniform_weight_v1',
  'event_matvec_prob_conn_normal_weight_v1',
]


def event_matvec_prob_conn_homo_weight(
    events: jnp.ndarray,
    weight: float,
    *,
    conn_prob: float,
    shape: Tuple[int, int],
    seed: Optional[int] = None,
    transpose: bool = False,
) -> jnp.ndarray:
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
  if seed is None:
    seed = np.random.randint(0, int(1e10))
  r = event_matvec_prob_conn_homo_weight_p.bind(events,
                                                conn_prob=conn_prob,
                                                shape=shape,
                                                seed=seed,
                                                transpose=transpose)
  return r * weight


def _event_matvec_prob_conn_homo_weight_abstract(
    events, *, conn_prob, shape, seed, transpose
):
  return ShapedArray(dtype=(dtypes.canonicalize_dtype(float)
                            if events.dtype == jnp.bool_ else events.dtype),
                     shape=(shape[1] if transpose else shape[0],))


def _event_matvec_prob_conn_homo_weight_gpu_translation(
    c, events, *, conn_prob, shape, seed, transpose
):
  if gpu_ops is None:
    raise GPUOperatorNotFound(event_matvec_prob_conn_homo_weight_p.name)

  event_shape = c.get_shape(events)
  if event_shape.element_type() == jnp.bool_:
    event_type = b'_bool'
    out_dtype = dtypes.canonicalize_dtype(float)
    type_name = b'_float' if out_dtype == jnp.float32 else b'_double'
  else:
    out_dtype = event_shape.element_type()
    event_type = b'_float' if out_dtype == jnp.float32 else b'_double'
    type_name = event_type

  opaque = gpu_ops.build_jitconn_prob_homo_descriptor(shape[1] if transpose else shape[0],
                                                      shape[0] if transpose else shape[1],
                                                      seed,
                                                      float(np.log((1 - conn_prob) if conn_prob < 1 else 1e-40)), )
  return xla_client.ops.CustomCallWithLayout(
    c,
    b'event_matvec_jitconn_prob_homo_v2' + type_name + event_type,
    operands=(events,),
    operand_shapes_with_layout=(c.get_shape(events),),
    shape_with_layout=xla_client.Shape.array_shape(out_dtype,
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


def event_matvec_prob_conn_uniform_weight(
    events: jnp.ndarray,
    *,
    w_low: float,
    w_high: float,
    conn_prob: float,
    shape: Tuple[int, int],
    seed: Optional[int] = None,
    transpose: bool = False,
) -> jnp.ndarray:
  assert w_high > w_low
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
  if seed is None:
    seed = np.random.randint(0, int(1e10))
  return event_matvec_prob_conn_uniform_weight_p.bind(events,
                                                      w_low=w_low,
                                                      weight_range=w_high - w_low,
                                                      conn_prob=conn_prob,
                                                      shape=shape,
                                                      seed=seed,
                                                      transpose=transpose)


def _event_matvec_prob_conn_uniform_weight_abstract(
    events, *, w_low, weight_range, conn_prob, shape, seed, transpose
):
  return ShapedArray(dtype=(dtypes.canonicalize_dtype(float)
                            if events.dtype == jnp.bool_ else events.dtype),
                     shape=(shape[1] if transpose else shape[0],))


def _event_matvec_prob_conn_uniform_weight_gpu_translation(
    c, events, *, w_low, weight_range, conn_prob, shape, seed, transpose
):
  if gpu_ops is None:
    raise GPUOperatorNotFound(event_matvec_prob_conn_homo_weight_p.name)

  event_shape = c.get_shape(events)
  if event_shape.element_type() == jnp.bool_:
    event_type = b'_bool'
    out_dtype = dtypes.canonicalize_dtype(float)
    type_name = b'_float' if out_dtype == jnp.float32 else b'_double'
  else:
    out_dtype = event_shape.element_type()
    event_type = b'_float' if out_dtype == jnp.float32 else b'_double'
    type_name = event_type

  opaque = gpu_ops.build_jitconn_prob_uniform_descriptor(shape[1] if transpose else shape[0],
                                                         shape[0] if transpose else shape[1],
                                                         seed,
                                                         float(np.log((1 - conn_prob) if conn_prob < 1 else 1e-40)),
                                                         w_low,
                                                         weight_range)
  return xla_client.ops.CustomCallWithLayout(
    c,
    b'event_matvec_jitconn_prob_uniform_v2' + type_name + event_type,
    operands=(events,),
    operand_shapes_with_layout=(c.get_shape(events),),
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


def event_matvec_prob_conn_normal_weight(
    events: jnp.ndarray,
    *,
    w_mu: float,
    w_sigma: float,
    conn_prob: float,
    shape: Tuple[int, int],
    seed: Optional[int] = None,
    transpose: bool = False,
) -> jnp.ndarray:
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
  if seed is None:
    seed = np.random.randint(0, int(1e10))
  return event_matvec_prob_conn_normal_weight_p.bind(events,
                                                     w_mu=w_mu,
                                                     w_sigma=w_sigma,
                                                     conn_prob=conn_prob,
                                                     shape=shape,
                                                     seed=seed,
                                                     transpose=transpose)


def _event_matvec_prob_conn_normal_weight_abstract(
    events, *, w_mu, w_sigma, conn_prob, shape, seed, transpose
):
  return ShapedArray(dtype=(dtypes.canonicalize_dtype(float)
                            if events.dtype == jnp.bool_ else events.dtype),
                     shape=(shape[1] if transpose else shape[0],))


def _event_matvec_prob_conn_normal_weight_gpu_translation(
    c, events, *, w_mu, w_sigma, conn_prob, shape, seed, transpose
):
  if gpu_ops is None:
    raise GPUOperatorNotFound(event_matvec_prob_conn_homo_weight_p.name)

  event_shape = c.get_shape(events)
  if event_shape.element_type() == jnp.bool_:
    event_type = b'_bool'
    out_dtype = dtypes.canonicalize_dtype(float)
    type_name = b'_float' if out_dtype == jnp.float32 else b'_double'
  else:
    out_dtype = event_shape.element_type()
    event_type = b'_float' if out_dtype == jnp.float32 else b'_double'
    type_name = event_type
  opaque = gpu_ops.build_jitconn_prob_normal_descriptor(shape[1] if transpose else shape[0],
                                                        shape[0] if transpose else shape[1],
                                                        seed,
                                                        float(np.log((1 - conn_prob) if conn_prob < 1 else 1e-40)),
                                                        w_mu,
                                                        w_sigma)
  return xla_client.ops.CustomCallWithLayout(
    c,
    b'event_matvec_jitconn_prob_normal_v2' + type_name + event_type,
    operands=(events,),
    operand_shapes_with_layout=(c.get_shape(events),),
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


def event_matvec_prob_conn_homo_weight_v1(
    events: jnp.ndarray,
    weight: float,
    *,
    conn_prob: float,
    shape: Tuple[int, int],
    seed: Optional[int] = None,
    transpose: bool = False,
) -> jnp.ndarray:
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
  if seed is None:
    seed = np.random.randint(0, int(1e10))
  event_ids, event_num = event_info(events)
  r = event_matvec_prob_conn_homo_weight_v1_p.bind(event_ids,
                                                   event_num,
                                                   conn_prob=conn_prob,
                                                   shape=shape,
                                                   seed=seed,
                                                   transpose=transpose)
  return r * weight


def _event_matvec_prob_conn_homo_weight_v1_abstract(
    event_ids, event_num, *, conn_prob, shape, seed, transpose
):
  return ShapedArray(dtype=dtypes.canonicalize_dtype(float),
                     shape=(shape[1] if transpose else shape[0],))


def _event_matvec_prob_conn_homo_weight_v1_gpu_translation(
    c, event_ids, event_num, *, conn_prob, shape, seed, transpose
):
  if gpu_ops is None:
    raise GPUOperatorNotFound(event_matvec_prob_conn_homo_weight_v1_p.name)

  out_dtype = dtypes.canonicalize_dtype(float)
  type_name = b'_float' if out_dtype == jnp.float32 else b'_double'

  opaque = gpu_ops.build_jitconn_prob_homo_descriptor(shape[1] if transpose else shape[0],
                                                      shape[0] if transpose else shape[1],
                                                      seed,
                                                      conn_prob)
  return xla_client.ops.CustomCallWithLayout(
    c,
    b'event_matvec_jitconn_prob_homo' + type_name,
    operands=(event_ids, event_num),
    operand_shapes_with_layout=(c.get_shape(event_ids),
                                c.get_shape(event_num)),
    shape_with_layout=xla_client.Shape.array_shape(out_dtype,
                                                   (shape[1] if transpose else shape[0],),
                                                   (0,)),
    opaque=opaque,
  )


event_matvec_prob_conn_homo_weight_v1_p = Primitive('event_matvec_prob_conn_homo_weight_v1')
event_matvec_prob_conn_homo_weight_v1_p.def_abstract_eval(_event_matvec_prob_conn_homo_weight_v1_abstract)
event_matvec_prob_conn_homo_weight_v1_p.def_impl(partial(xla.apply_primitive, event_matvec_prob_conn_homo_weight_v1_p))
xla.backend_specific_translations['gpu'][event_matvec_prob_conn_homo_weight_v1_p] = \
  _event_matvec_prob_conn_homo_weight_v1_gpu_translation
register_general_batching(event_matvec_prob_conn_homo_weight_v1_p)


def event_matvec_prob_conn_uniform_weight_v1(
    events: jnp.ndarray,
    *,
    w_low: float,
    w_high: float,
    conn_prob: float,
    shape: Tuple[int, int],
    seed: Optional[int] = None,
    transpose: bool = False,
) -> jnp.ndarray:
  assert w_high > w_low
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
  if seed is None:
    seed = np.random.randint(0, int(1e10))
  event_ids, event_num = event_info(events)
  return event_matvec_prob_conn_uniform_weight_v1_p.bind(event_ids,
                                                         event_num,
                                                         w_low=w_low,
                                                         weight_range=w_high - w_low,
                                                         conn_prob=conn_prob,
                                                         shape=shape,
                                                         seed=seed,
                                                         transpose=transpose)


def _event_matvec_prob_conn_uniform_weight_v1_abstract(
    event_ids, event_num, *,
    w_low, weight_range, conn_prob, shape, seed, transpose
):
  return ShapedArray(dtype=dtypes.canonicalize_dtype(float),
                     shape=(shape[1] if transpose else shape[0],))


def _event_matvec_prob_conn_uniform_weight_v1_gpu_translation(
    c, event_ids, event_num, *, w_low, weight_range, conn_prob, shape, seed, transpose
):
  if gpu_ops is None:
    raise GPUOperatorNotFound(event_matvec_prob_conn_homo_weight_v1_p.name)

  out_dtype = dtypes.canonicalize_dtype(float)
  type_name = b'_float' if out_dtype == jnp.float32 else b'_double'

  opaque = gpu_ops.build_jitconn_prob_uniform_descriptor(shape[1] if transpose else shape[0],
                                                         shape[0] if transpose else shape[1],
                                                         seed,
                                                         conn_prob,
                                                         w_low,
                                                         weight_range)
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


event_matvec_prob_conn_uniform_weight_v1_p = Primitive('event_matvec_prob_conn_uniform_weight_v1')
event_matvec_prob_conn_uniform_weight_v1_p.def_abstract_eval(_event_matvec_prob_conn_uniform_weight_v1_abstract)
event_matvec_prob_conn_uniform_weight_v1_p.def_impl(
  partial(xla.apply_primitive, event_matvec_prob_conn_uniform_weight_v1_p))
xla.backend_specific_translations['gpu'][event_matvec_prob_conn_uniform_weight_v1_p] = \
  _event_matvec_prob_conn_uniform_weight_v1_gpu_translation
register_general_batching(event_matvec_prob_conn_uniform_weight_v1_p)


def event_matvec_prob_conn_normal_weight_v1(
    events: jnp.ndarray,
    *,
    w_mu: float,
    w_sigma: float,
    conn_prob: float,
    shape: Tuple[int, int],
    seed: Optional[int] = None,
    transpose: bool = False,
) -> jnp.ndarray:
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
  if seed is None:
    seed = np.random.randint(0, int(1e10))
  event_ids, event_num = event_info(events)
  return event_matvec_prob_conn_normal_weight_v1_p.bind(event_ids,
                                                        event_num,
                                                        w_mu=w_mu,
                                                        w_sigma=w_sigma,
                                                        conn_prob=conn_prob,
                                                        shape=shape,
                                                        seed=seed,
                                                        transpose=transpose)


def _event_matvec_prob_conn_normal_weight_v1_abstract(
    event_ids, event_num, *, w_mu, w_sigma, conn_prob, shape, seed, transpose
):
  return ShapedArray(dtype=dtypes.canonicalize_dtype(float),
                     shape=(shape[1] if transpose else shape[0],))


def _event_matvec_prob_conn_normal_weight_v1_gpu_translation(
    c, event_ids, event_num, *, w_mu, w_sigma, conn_prob, shape, seed, transpose
):
  if gpu_ops is None:
    raise GPUOperatorNotFound(event_matvec_prob_conn_homo_weight_v1_p.name)

  out_dtype = dtypes.canonicalize_dtype(float)
  type_name = b'_float' if out_dtype == jnp.float32 else b'_double'

  opaque = gpu_ops.build_jitconn_prob_normal_descriptor(shape[1] if transpose else shape[0],
                                                        shape[0] if transpose else shape[1],
                                                        seed,
                                                        conn_prob,
                                                        w_mu,
                                                        w_sigma)
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


event_matvec_prob_conn_normal_weight_v1_p = Primitive('event_matvec_prob_conn_normal_weight_v1')
event_matvec_prob_conn_normal_weight_v1_p.def_abstract_eval(_event_matvec_prob_conn_normal_weight_v1_abstract)
event_matvec_prob_conn_normal_weight_v1_p.def_impl(
  partial(xla.apply_primitive, event_matvec_prob_conn_normal_weight_v1_p))
xla.backend_specific_translations['gpu'][event_matvec_prob_conn_normal_weight_v1_p] = \
  _event_matvec_prob_conn_normal_weight_v1_gpu_translation
register_general_batching(event_matvec_prob_conn_normal_weight_v1_p)
