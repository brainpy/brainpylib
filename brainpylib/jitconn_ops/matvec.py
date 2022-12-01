# -*- coding: utf-8 -*-


from functools import partial
from typing import Tuple, Optional

import numpy as np
from jax import numpy as jnp, dtypes
from jax.core import ShapedArray, Primitive
from jax.interpreters import xla
from jax.lib import xla_client

from brainpylib.errors import GPUOperatorNotFound
from brainpylib.op_register import (register_general_batching)

try:
  from brainpylib import gpu_ops
except ImportError:
  gpu_ops = None

__all__ = [
  'matvec_prob_conn_homo_weight',
  'matvec_prob_conn_uniform_weight',
  'matvec_prob_conn_normal_weight',
]


def matvec_prob_conn_homo_weight(
    vector: jnp.ndarray,
    weight: float,
    *,
    conn_prob: float,
    shape: Tuple[int, int],
    seed: Optional[int] = None,
    transpose: bool = False,
    version: str = 'v1'
) -> jnp.ndarray:
  if np.ndim(vector) != 1:
    raise ValueError('vector should be a 1D vector.')
  if len(shape) != 2:
    raise ValueError('shape should be a length-2 tuple.')
  if transpose:
    if vector.shape[0] != shape[0]:
      raise ValueError(f'Shape mismatch, vec ({vector.shape[0]},) @ mat {shape}.')
  else:
    if vector.shape[0] != shape[1]:
      raise ValueError(f'Shape mismatch, mat {shape} @ vec ({vector.shape[0]},).')
  if seed is None:
    seed = np.random.randint(0, int(1e10))
  assert version in ['v1', 'v2', 'v3']
  r = matvec_prob_conn_homo_weight_p.bind(vector,
                                          conn_prob=conn_prob,
                                          shape=shape,
                                          seed=seed,
                                          transpose=transpose,
                                          version=version)
  return r * weight


def matvec_prob_conn_uniform_weight(
    vector: jnp.ndarray,
    *,
    w_low: float,
    w_high: float,
    conn_prob: float,
    shape: Tuple[int, int],
    seed: Optional[int] = None,
    transpose: bool = False,
    version: str = 'v1'
) -> jnp.ndarray:
  assert w_high > w_low
  if np.ndim(vector) != 1:
    raise ValueError('vector should be a 1D vector.')
  if len(shape) != 2:
    raise ValueError('shape should be a length-2 tuple.')
  if transpose:
    if vector.shape[0] != shape[0]:
      raise ValueError(f'Shape mismatch, vec ({vector.shape[0]},) @ mat {shape}.')
  else:
    if vector.shape[0] != shape[1]:
      raise ValueError(f'Shape mismatch, mat {shape} @ vec ({vector.shape[0]},).')
  if seed is None:
    seed = np.random.randint(0, int(1e10))
  return matvec_prob_conn_uniform_weight_p.bind(vector,
                                                w_low=w_low,
                                                w_range=w_high - w_low,
                                                conn_prob=conn_prob,
                                                shape=shape,
                                                seed=seed,
                                                transpose=transpose,
                                                version=version)


def matvec_prob_conn_normal_weight(
    vector: jnp.ndarray,
    *,
    w_mu: float,
    w_sigma: float,
    conn_prob: float,
    shape: Tuple[int, int],
    seed: Optional[int] = None,
    transpose: bool = False,
    version: str = 'v1'
) -> jnp.ndarray:
  if np.ndim(vector) != 1:
    raise ValueError('vector should be a 1D vector.')
  if len(shape) != 2:
    raise ValueError('shape should be a length-2 tuple.')
  if transpose:
    if vector.shape[0] != shape[0]:
      raise ValueError(f'Shape mismatch, vec ({vector.shape[0]},) @ mat {shape}.')
  else:
    if vector.shape[0] != shape[1]:
      raise ValueError(f'Shape mismatch, mat {shape} @ vec ({vector.shape[0]},).')
  if seed is None:
    seed = np.random.randint(0, int(1e10))
  return matvec_prob_conn_normal_weight_p.bind(vector,
                                               w_mu=w_mu,
                                               w_sigma=w_sigma,
                                               conn_prob=conn_prob,
                                               shape=shape,
                                               seed=seed,
                                               transpose=transpose,
                                               version=version)


def _matvec_prob_conn_homo_weight_abstract(
    vector, *, conn_prob, shape, seed, transpose, version
):
  return ShapedArray(dtype=dtypes.canonicalize_dtype(float),
                     shape=(shape[1] if transpose else shape[0],))


def _matvec_prob_conn_homo_weight_gpu_translation(
    c, vector, *, conn_prob, shape, seed, transpose, version
):
  if gpu_ops is None:
    raise GPUOperatorNotFound(matvec_prob_conn_homo_weight_p.name)

  out_dtype = dtypes.canonicalize_dtype(float)
  type_name = b'_float' if out_dtype == jnp.float32 else b'_double'
  if version == 'v1':
    version_name = b''
    opaque = gpu_ops.build_jitconn_prob_homo_descriptor(shape[1] if transpose else shape[0],
                                                        shape[0] if transpose else shape[1],
                                                        seed,
                                                        conn_prob)
  elif version == 'v2':
    version_name = b'_v2'
    opaque = gpu_ops.build_jitconn_prob_homo_descriptor(shape[1] if transpose else shape[0],
                                                        shape[0] if transpose else shape[1],
                                                        seed,
                                                        float(np.log((1 - conn_prob) if conn_prob < 1 else 1e-40)))
  elif version == 'v3':
    version_name = b'_v3'
    opaque = gpu_ops.build_jitconn_prob_homo_descriptor(shape[1] if transpose else shape[0],
                                                        shape[0] if transpose else shape[1],
                                                        seed,
                                                        float(np.log((1 - conn_prob) if conn_prob < 1 else 1e-40)))
  else:
    raise ValueError

  return xla_client.ops.CustomCallWithLayout(
    c,
    b'matvec_jitconn_prob_homo' + version_name + type_name,
    operands=(vector,),
    operand_shapes_with_layout=(c.get_shape(vector),),
    shape_with_layout=xla_client.Shape.array_shape(out_dtype,
                                                   (shape[1] if transpose else shape[0],),
                                                   (0,)),
    opaque=opaque,
  )


matvec_prob_conn_homo_weight_p = Primitive('matvec_prob_conn_homo_weight')
matvec_prob_conn_homo_weight_p.def_abstract_eval(_matvec_prob_conn_homo_weight_abstract)
matvec_prob_conn_homo_weight_p.def_impl(partial(xla.apply_primitive, matvec_prob_conn_homo_weight_p))
xla.backend_specific_translations['gpu'][matvec_prob_conn_homo_weight_p] = \
  _matvec_prob_conn_homo_weight_gpu_translation
register_general_batching(matvec_prob_conn_homo_weight_p)


def _matvec_prob_conn_uniform_weight_abstract(
    vector, *, w_low, w_range, conn_prob, shape, seed, transpose, version
):
  return ShapedArray(dtype=dtypes.canonicalize_dtype(float),
                     shape=(shape[1] if transpose else shape[0],))


def _matvec_prob_conn_uniform_weight_gpu_translation(
    c, vector, *, w_low, w_range, conn_prob, shape, seed, transpose, version
):
  if gpu_ops is None:
    raise GPUOperatorNotFound(matvec_prob_conn_homo_weight_p.name)

  out_dtype = dtypes.canonicalize_dtype(float)
  type_name = b'_float' if out_dtype == jnp.float32 else b'_double'

  if version == 'v1':
    version_name = b''
    opaque = gpu_ops.build_jitconn_prob_uniform_descriptor(shape[1] if transpose else shape[0],
                                                           shape[0] if transpose else shape[1],
                                                           seed,
                                                           conn_prob,
                                                           w_low,
                                                           w_range)
  elif version == 'v2':
    version_name = b'_v2'
    opaque = gpu_ops.build_jitconn_prob_uniform_descriptor(shape[1] if transpose else shape[0],
                                                           shape[0] if transpose else shape[1],
                                                           seed,
                                                           float(np.log((1 - conn_prob) if conn_prob < 1 else 1e-40)),
                                                           w_low,
                                                           w_range)
  else:
    raise ValueError

  return xla_client.ops.CustomCallWithLayout(
    c,
    b'matvec_jitconn_prob_uniform' + version_name + type_name,
    operands=(vector,),
    operand_shapes_with_layout=(c.get_shape(vector),),
    shape_with_layout=xla_client.Shape.array_shape(out_dtype,
                                                   (shape[1] if transpose else shape[0],),
                                                   (0,)),
    opaque=opaque,
  )


matvec_prob_conn_uniform_weight_p = Primitive('matvec_prob_conn_uniform_weight')
matvec_prob_conn_uniform_weight_p.def_abstract_eval(_matvec_prob_conn_uniform_weight_abstract)
matvec_prob_conn_uniform_weight_p.def_impl(partial(xla.apply_primitive, matvec_prob_conn_uniform_weight_p))
xla.backend_specific_translations['gpu'][matvec_prob_conn_uniform_weight_p] = \
  _matvec_prob_conn_uniform_weight_gpu_translation
register_general_batching(matvec_prob_conn_uniform_weight_p)


def _matvec_prob_conn_normal_weight_abstract(
    vector, *, w_mu, w_sigma, conn_prob, shape, seed, transpose, version
):
  return ShapedArray(dtype=dtypes.canonicalize_dtype(float),
                     shape=(shape[1] if transpose else shape[0],))


def _matvec_prob_conn_normal_weight_gpu_translation(
    c, vector, *, w_mu, w_sigma, conn_prob, shape, seed, transpose, version
):
  if gpu_ops is None:
    raise GPUOperatorNotFound(matvec_prob_conn_homo_weight_p.name)

  out_dtype = dtypes.canonicalize_dtype(float)
  type_name = b'_float' if out_dtype == jnp.float32 else b'_double'

  if version == 'v1':
    version_name = b''
    opaque = gpu_ops.build_jitconn_prob_normal_descriptor(shape[1] if transpose else shape[0],
                                                          shape[0] if transpose else shape[1],
                                                          seed,
                                                          conn_prob,
                                                          w_mu,
                                                          w_sigma)
  elif version == 'v2':
    version_name = b'_v2'
    opaque = gpu_ops.build_jitconn_prob_uniform_descriptor(shape[1] if transpose else shape[0],
                                                           shape[0] if transpose else shape[1],
                                                           seed,
                                                           float(np.log((1 - conn_prob) if conn_prob < 1 else 1e-40)),
                                                           w_mu,
                                                           w_sigma)
  else:
    raise ValueError

  return xla_client.ops.CustomCallWithLayout(
    c,
    b'matvec_jitconn_prob_normal' + version_name + type_name,
    operands=(vector,),
    operand_shapes_with_layout=(c.get_shape(vector),),
    shape_with_layout=xla_client.Shape.array_shape(out_dtype,
                                                   (shape[1] if transpose else shape[0],),
                                                   (0,)),
    opaque=opaque,
  )


matvec_prob_conn_normal_weight_p = Primitive('matvec_prob_conn_normal_weight')
matvec_prob_conn_normal_weight_p.def_abstract_eval(_matvec_prob_conn_normal_weight_abstract)
matvec_prob_conn_normal_weight_p.def_impl(partial(xla.apply_primitive, matvec_prob_conn_normal_weight_p))
xla.backend_specific_translations['gpu'][matvec_prob_conn_normal_weight_p] = \
  _matvec_prob_conn_normal_weight_gpu_translation
register_general_batching(matvec_prob_conn_normal_weight_p)
