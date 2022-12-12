# -*- coding: utf-8 -*-


from functools import partial
from typing import Tuple, Optional

import numpy as np
from jax import numpy as jnp, dtypes
from jax.core import ShapedArray, Primitive
from jax.interpreters import xla, ad
from jax.lib import xla_client

from brainpylib.errors import GPUOperatorNotFound
from brainpylib.op_register import (register_general_batching)
from .matvec import (matvec_prob_homo_p,
                     matvec_prob_uniform_p,
                     matvec_prob_normal_p,
                     vecmat_prob_homo_p,
                     vecmat_prob_uniform_p,
                     vecmat_prob_normal_p, )

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
    seed = int(np.random.randint(0, int(1e10)))
  r = event_matvec_prob_homo_p.bind(events,
                                    conn_prob=conn_prob,
                                    shape=shape,
                                    seed=seed,
                                    transpose=transpose)[0]
  weight = jnp.asarray(weight, dtype=r.dtype)
  return r * weight


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
    seed = int(np.random.randint(0, int(1e10)))
  return event_matvec_prob_uniform_p.bind(events,
                                          w_low=w_low,
                                          w_high=w_high,
                                          conn_prob=conn_prob,
                                          shape=shape,
                                          seed=seed,
                                          transpose=transpose)[0]


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
    seed = int(np.random.randint(0, int(1e10)))
  return event_matvec_prob_normal_p.bind(events,
                                         w_mu=w_mu,
                                         w_sigma=w_sigma,
                                         conn_prob=conn_prob,
                                         shape=shape,
                                         seed=seed,
                                         transpose=transpose)[0]


def _event_matvec_prob_homo_abstract(
    events, *, conn_prob, shape, seed, transpose
):
  out = ShapedArray(dtype=(dtypes.canonicalize_dtype(float)
                           if events.dtype == jnp.bool_ else events.dtype),
                    shape=(shape[1] if transpose else shape[0],))
  return [out]


def _event_matvec_prob_homo_cpu_translation(
    c, events, *, conn_prob, shape, seed, transpose
):
  log_p = float(np.log((1 - conn_prob) if (conn_prob < 1) else 1e-40))
  n_row, n_col = (shape[1], shape[0]) if transpose else shape

  event_shape = c.get_shape(events)
  if event_shape.element_type() == jnp.bool_:
    event_type = b'_bool'
    out_dtype = dtypes.canonicalize_dtype(float)
    type_name = b'_float' if out_dtype == jnp.float32 else b'_double'
  else:
    out_dtype = event_shape.element_type()
    event_type = b'_float' if out_dtype == jnp.float32 else b'_double'
    type_name = event_type

  return xla_client.ops.CustomCallWithLayout(
    c,
    b'cpu_event_matvec_prob_homo' + type_name + event_type,
    operands=(events,
              xla_client.ops.ConstantLiteral(c, log_p),
              xla_client.ops.ConstantLiteral(c, seed),
              xla_client.ops.ConstantLiteral(c, n_row),
              xla_client.ops.ConstantLiteral(c, n_col)),
    operand_shapes_with_layout=(c.get_shape(events),
                                xla_client.Shape.array_shape(np.dtype(np.float64), (), ()),
                                xla_client.Shape.array_shape(np.dtype(np.uint32), (), ()),
                                xla_client.Shape.array_shape(np.dtype(np.uint32), (), ()),
                                xla_client.Shape.array_shape(np.dtype(np.uint32), (), ())),
    shape_with_layout=xla_client.Shape.tuple_shape(
      (
        xla_client.Shape.array_shape(out_dtype, (shape[1] if transpose else shape[0],), (0,)),
      )
    ),
  )


def _event_matvec_prob_homo_gpu_translation(
    c, events, *, conn_prob, shape, seed, transpose
):
  if gpu_ops is None:
    raise GPUOperatorNotFound(event_matvec_prob_homo_p.name)

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
    shape_with_layout=xla_client.Shape.tuple_shape(
      (
        xla_client.Shape.array_shape(out_dtype, (shape[1] if transpose else shape[0],), (0,)),
      )
    ),
    opaque=opaque,
  )


def _event_matvec_prob_homo_jvp(
    primals, tangents, *, conn_prob, shape, seed, transpose
):
  events, = primals
  event_dot, = tangents
  r = event_matvec_prob_homo_p.bind(events,
                                    conn_prob=conn_prob,
                                    shape=shape,
                                    seed=seed,
                                    transpose=transpose, )
  dr = matvec_prob_homo_p.bind(event_dot,
                               conn_prob=conn_prob,
                               shape=shape,
                               seed=seed,
                               transpose=transpose,
                               version='v2')
  return r, dr


def _event_matvec_prob_homo_transpose(
    ct, events, *, conn_prob, shape, seed, transpose
):
  ct_event = vecmat_prob_homo_p.bind(ct,
                                     conn_prob=conn_prob,
                                     seed=seed,
                                     shape=shape,
                                     transpose=transpose,
                                     version='v2')
  return ct_event


event_matvec_prob_homo_p = Primitive('event_matvec_prob_conn_homo_weight')
event_matvec_prob_homo_p.multiple_results = True
event_matvec_prob_homo_p.def_abstract_eval(_event_matvec_prob_homo_abstract)
event_matvec_prob_homo_p.def_impl(partial(xla.apply_primitive, event_matvec_prob_homo_p))
xla.backend_specific_translations['cpu'][event_matvec_prob_homo_p] = _event_matvec_prob_homo_cpu_translation
xla.backend_specific_translations['gpu'][event_matvec_prob_homo_p] = _event_matvec_prob_homo_gpu_translation
ad.primitive_jvps[event_matvec_prob_homo_p] = _event_matvec_prob_homo_jvp
ad.primitive_transposes[event_matvec_prob_homo_p] = _event_matvec_prob_homo_transpose
register_general_batching(event_matvec_prob_homo_p)


def _event_matvec_prob_uniform_abstract(
    events, *, w_low, w_high, conn_prob, shape, seed, transpose
):
  out = ShapedArray(dtype=(dtypes.canonicalize_dtype(float)
                           if events.dtype == jnp.bool_ else events.dtype),
                    shape=(shape[1] if transpose else shape[0],))
  return [out]


def _event_matvec_prob_uniform_cpu_translation(
    c, events, *, w_low, w_high, conn_prob, shape, seed, transpose
):
  log_p = np.log((1 - conn_prob) if (conn_prob < 1) else 1e-40)
  n_row, n_col = (shape[1], shape[0]) if transpose else shape

  event_shape = c.get_shape(events)
  if event_shape.element_type() == jnp.bool_:
    event_type = b'_bool'
    out_dtype = dtypes.canonicalize_dtype(float)
    type_name = b'_float' if out_dtype == jnp.float32 else b'_double'
  else:
    out_dtype = event_shape.element_type()
    event_type = b'_float' if out_dtype == jnp.float32 else b'_double'
    type_name = event_type
  w_low = jnp.asarray(w_low, dtype=out_dtype)
  w_high = jnp.asarray(w_high, dtype=out_dtype)

  return xla_client.ops.CustomCallWithLayout(
    c,
    b'cpu_event_matvec_prob_uniform' + type_name + event_type,
    operands=(events,
              xla_client.ops.ConstantLiteral(c, log_p),
              xla_client.ops.ConstantLiteral(c, w_low),
              xla_client.ops.ConstantLiteral(c, w_high),
              xla_client.ops.ConstantLiteral(c, seed),
              xla_client.ops.ConstantLiteral(c, n_row),
              xla_client.ops.ConstantLiteral(c, n_col)),
    operand_shapes_with_layout=(c.get_shape(events),
                                xla_client.Shape.array_shape(np.dtype(np.float64), (), ()),
                                xla_client.Shape.array_shape(out_dtype, (), ()),
                                xla_client.Shape.array_shape(out_dtype, (), ()),
                                xla_client.Shape.array_shape(np.dtype(np.uint32), (), ()),
                                xla_client.Shape.array_shape(np.dtype(np.uint32), (), ()),
                                xla_client.Shape.array_shape(np.dtype(np.uint32), (), ())),
    shape_with_layout=xla_client.Shape.tuple_shape(
      (
        xla_client.Shape.array_shape(out_dtype, (shape[1] if transpose else shape[0],), (0,)),
      )
    ),
  )


def _event_matvec_prob_uniform_gpu_translation(
    c, events, *, w_low, w_high, conn_prob, shape, seed, transpose
):
  if gpu_ops is None:
    raise GPUOperatorNotFound(event_matvec_prob_homo_p.name)

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
                                                         w_high - w_low)
  return xla_client.ops.CustomCallWithLayout(
    c,
    b'event_matvec_jitconn_prob_uniform_v2' + type_name + event_type,
    operands=(events,),
    operand_shapes_with_layout=(c.get_shape(events),),
    shape_with_layout=xla_client.Shape.tuple_shape(
      (
        xla_client.Shape.array_shape(out_dtype, (shape[1] if transpose else shape[0],), (0,)),
      )
    ),
    opaque=opaque,
  )


def _event_matvec_prob_uniform_jvp(
    primals, tangents, *, w_low, w_high, conn_prob, shape, seed, transpose
):
  events, = primals
  events_dot, = tangents
  r = event_matvec_prob_uniform_p.bind(events,
                                       w_low=w_low,
                                       w_high=w_high,
                                       conn_prob=conn_prob,
                                       shape=shape,
                                       seed=seed,
                                       transpose=transpose)
  r_dot = matvec_prob_uniform_p.bind(events_dot,
                                     w_low=w_low,
                                     w_high=w_high,
                                     conn_prob=conn_prob,
                                     shape=shape,
                                     seed=seed,
                                     transpose=transpose,
                                     version='v2')
  return r, r_dot


def _event_matvec_prob_uniform_transpose(
    ct, events, *, w_low, w_high, conn_prob, shape, seed, transpose
):
  return vecmat_prob_uniform_p.bind(ct,
                                    w_low=w_low,
                                    w_high=w_high,
                                    conn_prob=conn_prob,
                                    seed=seed,
                                    shape=shape,
                                    transpose=transpose,
                                    version='v2')


event_matvec_prob_uniform_p = Primitive('event_matvec_prob_uniform')
event_matvec_prob_uniform_p.multiple_results = True
event_matvec_prob_uniform_p.def_abstract_eval(_event_matvec_prob_uniform_abstract)
event_matvec_prob_uniform_p.def_impl(partial(xla.apply_primitive, event_matvec_prob_uniform_p))
xla.backend_specific_translations['cpu'][event_matvec_prob_uniform_p] = _event_matvec_prob_uniform_cpu_translation
xla.backend_specific_translations['gpu'][event_matvec_prob_uniform_p] = _event_matvec_prob_uniform_gpu_translation
register_general_batching(event_matvec_prob_uniform_p)
ad.primitive_jvps[event_matvec_prob_uniform_p] = _event_matvec_prob_uniform_jvp
ad.primitive_transposes[event_matvec_prob_uniform_p] = _event_matvec_prob_uniform_transpose


def _event_matvec_prob_normal_abstract(
    events, *, w_mu, w_sigma, conn_prob, shape, seed, transpose
):
  out = ShapedArray(dtype=(dtypes.canonicalize_dtype(float)
                           if events.dtype == jnp.bool_ else events.dtype),
                    shape=(shape[1] if transpose else shape[0],))
  return [out]


def _event_matvec_prob_normal_cpu_translation(
    c, events, *, w_mu, w_sigma, conn_prob, shape, seed, transpose
):
  log_p = np.log((1 - conn_prob) if (conn_prob < 1) else 1e-40)
  n_row, n_col = (shape[1], shape[0]) if transpose else shape

  event_shape = c.get_shape(events)
  if event_shape.element_type() == jnp.bool_:
    event_type = b'_bool'
    out_dtype = dtypes.canonicalize_dtype(float)
    type_name = b'_float' if out_dtype == jnp.float32 else b'_double'
  else:
    out_dtype = event_shape.element_type()
    event_type = b'_float' if out_dtype == jnp.float32 else b'_double'
    type_name = event_type
  w_mu = jnp.asarray(w_mu, dtype=out_dtype)
  w_sigma = jnp.asarray(w_sigma, dtype=out_dtype)

  return xla_client.ops.CustomCallWithLayout(
    c,
    b'cpu_event_matvec_prob_normal' + type_name + event_type,
    operands=(events,
              xla_client.ops.ConstantLiteral(c, log_p),
              xla_client.ops.ConstantLiteral(c, w_mu),
              xla_client.ops.ConstantLiteral(c, w_sigma),
              xla_client.ops.ConstantLiteral(c, seed),
              xla_client.ops.ConstantLiteral(c, n_row),
              xla_client.ops.ConstantLiteral(c, n_col)),
    operand_shapes_with_layout=(c.get_shape(events),
                                xla_client.Shape.array_shape(np.dtype(np.float64), (), ()),
                                xla_client.Shape.array_shape(out_dtype, (), ()),
                                xla_client.Shape.array_shape(out_dtype, (), ()),
                                xla_client.Shape.array_shape(np.dtype(np.uint32), (), ()),
                                xla_client.Shape.array_shape(np.dtype(np.uint32), (), ()),
                                xla_client.Shape.array_shape(np.dtype(np.uint32), (), ())),
    shape_with_layout=xla_client.Shape.tuple_shape(
      (
        xla_client.Shape.array_shape(out_dtype, (shape[1] if transpose else shape[0],), (0,)),
      )
    ),
  )


def _event_matvec_prob_normal_gpu_translation(
    c, events, *, w_mu, w_sigma, conn_prob, shape, seed, transpose
):
  if gpu_ops is None:
    raise GPUOperatorNotFound(event_matvec_prob_homo_p.name)

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
    shape_with_layout=xla_client.Shape.tuple_shape(
      (
        xla_client.Shape.array_shape(out_dtype, (shape[1] if transpose else shape[0],), (0,)),
      )
    ),
    opaque=opaque,
  )


def _event_matvec_prob_normal_jvp(
    primals, tangents, *, w_mu, w_sigma, conn_prob, shape, seed, transpose
):
  events, = primals
  events_dot, = tangents
  r = event_matvec_prob_normal_p.bind(events,
                                      w_mu=w_mu,
                                      w_sigma=w_sigma,
                                      conn_prob=conn_prob,
                                      shape=shape,
                                      seed=seed,
                                      transpose=transpose)
  r_dot = matvec_prob_normal_p.bind(events_dot,
                                    w_mu=w_mu,
                                    w_sigma=w_sigma,
                                    conn_prob=conn_prob,
                                    shape=shape,
                                    seed=seed,
                                    transpose=transpose,
                                    version='v2')
  return r, r_dot


def _event_matvec_prob_normal_transpose(
    ct, events, *, w_mu, w_sigma, conn_prob, shape, seed, transpose
):
  return vecmat_prob_normal_p.bind(ct,
                                   w_mu=w_mu,
                                   w_sigma=w_sigma,
                                   conn_prob=conn_prob,
                                   seed=seed,
                                   shape=shape,
                                   transpose=transpose,
                                   version='v2')


event_matvec_prob_normal_p = Primitive('event_matvec_prob_normal')
event_matvec_prob_normal_p.multiple_results = True
event_matvec_prob_normal_p.def_abstract_eval(_event_matvec_prob_normal_abstract)
event_matvec_prob_normal_p.def_impl(partial(xla.apply_primitive, event_matvec_prob_normal_p))
xla.backend_specific_translations['cpu'][event_matvec_prob_normal_p] = _event_matvec_prob_normal_cpu_translation
xla.backend_specific_translations['gpu'][event_matvec_prob_normal_p] = _event_matvec_prob_normal_gpu_translation
register_general_batching(event_matvec_prob_normal_p)
ad.primitive_jvps[event_matvec_prob_normal_p] = _event_matvec_prob_normal_jvp
ad.primitive_transposes[event_matvec_prob_normal_p] = _event_matvec_prob_normal_transpose
