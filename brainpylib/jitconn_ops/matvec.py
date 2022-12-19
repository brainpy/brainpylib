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
from brainpylib.tools import transform_brainpy_array

try:
  from brainpylib import gpu_ops
except ImportError:
  gpu_ops = None

vecmat_prob_homo_p = None
vecmat_prob_uniform_p = None
vecmat_prob_normal_p = None

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
    keep_same_mat: bool = False,
    version: str = 'v2'
) -> jnp.ndarray:
  r"""Performing the :math:`y=M@v` operation, where :math:`M` is just-in-time generated.

  This operator support ``jit()``, ``vmap()``, ``grad()`` and ``pmap()`` etc. transformations.

  .. warning::

     This API may change in the future.

  In this operation, :math:`M` is the random matrix with a connection probability
  `conn_prob`, and at each connection the value is the same scalar `weight`.

  When ``transpose=True``, we perform an operation of :math:`y=M^T@v`.

  .. note::

     Note that the just-in-time generated :math:`M` (`transpose=False`) is
     different from the generated :math:`M^T` (`transpose=True`).

     If you pursue the same :math:`M` and :math:`M^T` when performing the just-in-time
     matrix generation, you should set ``keep_same_mat=True``, with the sacrifice of
     the speed compared with ``keep_same_mat=False``.

  Parameters
  ----------
  vector: Array, ndarray
    The vector.
  weight: float
    The value of the random matrix.
  conn_prob: float
    The connection probability.
  shape: tuple of int
    The matrix shape.
  seed: int
    The random number generation seed.
  transpose: bool
    Transpose the random matrix or not.
  keep_same_mat: bool
    Keep the just-in-time generated :math:M^T: is the same
    as the just-in-time generated :math:`M`.
  version: str
    The api version.

  Returns
  -------
  out: Array, ndarray
    The output of :math:`y = M @ v`.
  """
  vector = transform_brainpy_array(vector)
  weight = transform_brainpy_array(weight)
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
    seed = int(np.random.randint(0, int(1e10)))
  assert version in ['v1', 'v2']
  r = matvec_prob_homo_p.bind(vector,
                              conn_prob=conn_prob,
                              shape=shape,
                              seed=seed,
                              transpose=transpose,
                              keep_same_mat=keep_same_mat,
                              version=version)[0]
  weight = jnp.asarray(weight, dtype=r.dtype)
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
    version: str = 'v2'
) -> jnp.ndarray:
  vector = transform_brainpy_array(vector)
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
    seed = int(np.random.randint(0, int(1e10)))
  return matvec_prob_uniform_p.bind(vector,
                                    w_low=w_low,
                                    w_high=w_high,
                                    conn_prob=conn_prob,
                                    shape=shape,
                                    seed=seed,
                                    transpose=transpose,
                                    version=version)[0]


def matvec_prob_conn_normal_weight(
    vector: jnp.ndarray,
    *,
    w_mu: float,
    w_sigma: float,
    conn_prob: float,
    shape: Tuple[int, int],
    seed: Optional[int] = None,
    transpose: bool = False,
    version: str = 'v2'
) -> jnp.ndarray:
  vector = transform_brainpy_array(vector)
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
    seed = int(np.random.randint(0, int(1e10)))
  return matvec_prob_normal_p.bind(vector,
                                   w_mu=w_mu,
                                   w_sigma=w_sigma,
                                   conn_prob=conn_prob,
                                   shape=shape,
                                   seed=seed,
                                   transpose=transpose,
                                   version=version)[0]


def _matvec_prob_homo_abstract(
    vector, *, conn_prob, shape, seed, transpose, keep_same_mat, version='v2'
):
  out = ShapedArray(dtype=dtypes.canonicalize_dtype(float),
                    shape=(shape[1] if transpose else shape[0],))
  return [out]


def _matvec_prob_homo_cpu_translation(
    c, vector, *, conn_prob, shape, seed, transpose, keep_same_mat, version
):
  log_p = float(np.log((1 - conn_prob) if (conn_prob < 1) else 1e-40))
  n_row, n_col = (shape[1], shape[0]) if transpose else shape

  vec_shape = c.get_shape(vector)
  out_dtype = vec_shape.element_type()
  out_type = b'_float' if out_dtype == jnp.float32 else b'_double'

  return xla_client.ops.CustomCallWithLayout(
    c,
    b'cpu_matvec_prob_homo' + out_type,
    operands=(vector,
              xla_client.ops.ConstantLiteral(c, log_p),
              xla_client.ops.ConstantLiteral(c, seed),
              xla_client.ops.ConstantLiteral(c, n_row),
              xla_client.ops.ConstantLiteral(c, n_col)),
    operand_shapes_with_layout=(c.get_shape(vector),
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


def _matvec_prob_homo_gpu_translation(
    c, vector, *, conn_prob, shape, seed, transpose, keep_same_mat, version
):
  if gpu_ops is None:
    raise GPUOperatorNotFound(matvec_prob_homo_p.name)

  out_dtype = dtypes.canonicalize_dtype(float)
  type_name = b'_float' if out_dtype == jnp.float32 else b'_double'
  if version == 'v1':
    version_name = b'_v1'
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
  else:
    raise ValueError

  return xla_client.ops.CustomCallWithLayout(
    c,
    b'gpu_matvec_prob_homo' + version_name + type_name,
    operands=(vector,),
    operand_shapes_with_layout=(c.get_shape(vector),),
    shape_with_layout=xla_client.Shape.tuple_shape(
      (
        xla_client.Shape.array_shape(out_dtype, (shape[1] if transpose else shape[0],), (0,)),
      )
    ),
    opaque=opaque,
  )


def _matvec_prob_homo_jvp(
    primals, tangents, *, conn_prob, shape, seed, transpose, keep_same_mat, version
):
  vector, = primals
  vector_dot, = tangents
  r = matvec_prob_homo_p.bind(vector,
                              conn_prob=conn_prob,
                              shape=shape,
                              seed=seed,
                              transpose=transpose,
                              version=version)
  r_dot = matvec_prob_homo_p.bind(vector_dot,
                                  conn_prob=conn_prob,
                                  shape=shape,
                                  seed=seed,
                                  transpose=transpose,
                                  version=version)
  return r, r_dot


def _matvec_prob_homo_transpose(
    ct, vector, *, conn_prob, shape, seed, transpose, keep_same_mat, version
):
  global vecmat_prob_homo_p
  if vecmat_prob_homo_p is None:
    from . import vecmat
    vecmat_prob_homo_p = vecmat.vecmat_prob_homo_p
  return vecmat_prob_homo_p.bind(ct[0],
                                 conn_prob=conn_prob,
                                 seed=seed,
                                 shape=shape,
                                 transpose=transpose,
                                 version=version)


matvec_prob_homo_p = Primitive('matvec_prob_homo')
matvec_prob_homo_p.multiple_results = True
matvec_prob_homo_p.def_abstract_eval(_matvec_prob_homo_abstract)
matvec_prob_homo_p.def_impl(partial(xla.apply_primitive, matvec_prob_homo_p))
xla.backend_specific_translations['cpu'][matvec_prob_homo_p] = _matvec_prob_homo_cpu_translation
xla.backend_specific_translations['gpu'][matvec_prob_homo_p] = _matvec_prob_homo_gpu_translation
register_general_batching(matvec_prob_homo_p)
ad.primitive_jvps[matvec_prob_homo_p] = _matvec_prob_homo_jvp
ad.primitive_transposes[matvec_prob_homo_p] = _matvec_prob_homo_transpose


def _matvec_prob_uniform_abstract(
    vector, *, w_low, w_high, conn_prob, shape, seed, transpose, version
):
  out = ShapedArray(dtype=dtypes.canonicalize_dtype(float),
                    shape=(shape[1] if transpose else shape[0],))
  return [out]


def _matvec_prob_uniform_cpu_translation(
    c, vector, *, w_low, w_high, conn_prob, shape, seed, transpose, version
):
  log_p = np.log((1 - conn_prob) if (conn_prob < 1) else 1e-40)
  n_row, n_col = (shape[1], shape[0]) if transpose else shape

  vec_shape = c.get_shape(vector)
  out_dtype = vec_shape.element_type()
  type_name = b'_float' if out_dtype == jnp.float32 else b'_double'
  w_low = jnp.asarray(w_low, dtype=out_dtype)
  w_high = jnp.asarray(w_high, dtype=out_dtype)

  return xla_client.ops.CustomCallWithLayout(
    c,
    b'cpu_matvec_prob_uniform' + type_name,
    operands=(vector,
              xla_client.ops.ConstantLiteral(c, log_p),
              xla_client.ops.ConstantLiteral(c, w_low),
              xla_client.ops.ConstantLiteral(c, w_high),
              xla_client.ops.ConstantLiteral(c, seed),
              xla_client.ops.ConstantLiteral(c, n_row),
              xla_client.ops.ConstantLiteral(c, n_col)),
    operand_shapes_with_layout=(c.get_shape(vector),
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


def _matvec_prob_uniform_gpu_translation(
    c, vector, *, w_low, w_high, conn_prob, shape, seed, transpose, version
):
  if gpu_ops is None:
    raise GPUOperatorNotFound(matvec_prob_homo_p.name)

  out_dtype = dtypes.canonicalize_dtype(float)
  type_name = b'_float' if out_dtype == jnp.float32 else b'_double'

  if version == 'v1':
    version_name = b'_v1'
    opaque = gpu_ops.build_jitconn_prob_uniform_descriptor(shape[1] if transpose else shape[0],
                                                           shape[0] if transpose else shape[1],
                                                           seed,
                                                           conn_prob,
                                                           w_low,
                                                           w_high - w_low)
  elif version == 'v2':
    version_name = b'_v2'
    opaque = gpu_ops.build_jitconn_prob_uniform_descriptor(shape[1] if transpose else shape[0],
                                                           shape[0] if transpose else shape[1],
                                                           seed,
                                                           float(np.log((1 - conn_prob) if conn_prob < 1 else 1e-40)),
                                                           w_low,
                                                           w_high - w_low)
  else:
    raise ValueError

  return xla_client.ops.CustomCallWithLayout(
    c,
    b'gpu_matvec_prob_uniform' + version_name + type_name,
    operands=(vector,),
    operand_shapes_with_layout=(c.get_shape(vector),),
    shape_with_layout=xla_client.Shape.tuple_shape(
      (
        xla_client.Shape.array_shape(out_dtype, (shape[1] if transpose else shape[0],), (0,)),
      )
    ),
    opaque=opaque,
  )


def _matvec_prob_uniform_jvp(
    primals, tangents, *, w_low, w_high, conn_prob, shape, seed, transpose, version
):
  vector, = primals
  vector_dot, = tangents
  r = matvec_prob_uniform_p.bind(vector,
                                 w_low=w_low,
                                 w_high=w_high,
                                 conn_prob=conn_prob,
                                 shape=shape,
                                 seed=seed,
                                 transpose=transpose,
                                 version=version)
  r_dot = matvec_prob_uniform_p.bind(vector_dot,
                                     w_low=w_low,
                                     w_high=w_high,
                                     conn_prob=conn_prob,
                                     shape=shape,
                                     seed=seed,
                                     transpose=transpose,
                                     version=version)
  return r, r_dot


def _matvec_prob_uniform_transpose(
    ct, events, *, w_low, w_high, conn_prob, shape, seed, transpose, version
):
  global vecmat_prob_uniform_p
  if vecmat_prob_uniform_p is None:
    from . import vecmat
    vecmat_prob_uniform_p = vecmat.vecmat_prob_uniform_p
  return vecmat_prob_uniform_p.bind(ct[0],
                                    w_low=w_low,
                                    w_high=w_high,
                                    conn_prob=conn_prob,
                                    seed=seed,
                                    shape=shape,
                                    transpose=transpose,
                                    version=version)


matvec_prob_uniform_p = Primitive('matvec_prob_uniform')
matvec_prob_uniform_p.multiple_results = True
matvec_prob_uniform_p.def_abstract_eval(_matvec_prob_uniform_abstract)
matvec_prob_uniform_p.def_impl(partial(xla.apply_primitive, matvec_prob_uniform_p))
xla.backend_specific_translations['cpu'][matvec_prob_uniform_p] = _matvec_prob_uniform_cpu_translation
xla.backend_specific_translations['gpu'][matvec_prob_uniform_p] = _matvec_prob_uniform_gpu_translation
register_general_batching(matvec_prob_uniform_p)
ad.primitive_jvps[matvec_prob_uniform_p] = _matvec_prob_uniform_jvp
ad.primitive_transposes[matvec_prob_uniform_p] = _matvec_prob_uniform_transpose


def _matvec_prob_normal_abstract(
    vector, *, w_mu, w_sigma, conn_prob, shape, seed, transpose, version
):
  out = ShapedArray(dtype=dtypes.canonicalize_dtype(float),
                    shape=(shape[1] if transpose else shape[0],))
  return [out]


def _matvec_prob_normal_cpu_translation(
    c, events, *, w_mu, w_sigma, conn_prob, shape, seed, transpose, version
):
  log_p = np.log((1 - conn_prob) if (conn_prob < 1) else 1e-40)
  n_row, n_col = (shape[1], shape[0]) if transpose else shape

  vec_shape = c.get_shape(events)
  out_dtype = vec_shape.element_type()
  type_name = b'_float' if out_dtype == jnp.float32 else b'_double'
  w_mu = jnp.asarray(w_mu, dtype=out_dtype)
  w_sigma = jnp.asarray(w_sigma, dtype=out_dtype)

  return xla_client.ops.CustomCallWithLayout(
    c,
    b'cpu_matvec_prob_normal' + type_name,
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


def _matvec_prob_normal_gpu_translation(
    c, vector, *, w_mu, w_sigma, conn_prob, shape, seed, transpose, version
):
  if gpu_ops is None:
    raise GPUOperatorNotFound(matvec_prob_homo_p.name)

  out_dtype = dtypes.canonicalize_dtype(float)
  type_name = b'_float' if out_dtype == jnp.float32 else b'_double'

  if version == 'v1':
    version_name = b'_v1'
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
    b'gpu_matvec_prob_normal' + version_name + type_name,
    operands=(vector,),
    operand_shapes_with_layout=(c.get_shape(vector),),
    shape_with_layout=xla_client.Shape.tuple_shape(
      (
        xla_client.Shape.array_shape(out_dtype, (shape[1] if transpose else shape[0],), (0,)),
      )
    ),
    opaque=opaque,
  )


def _matvec_prob_normal_jvp(
    primals, tangents, *, w_mu, w_sigma, conn_prob, shape, seed, transpose, version
):
  vector, = primals
  vector_dot, = tangents
  r = matvec_prob_normal_p.bind(vector,
                                w_mu=w_mu,
                                w_sigma=w_sigma,
                                conn_prob=conn_prob,
                                shape=shape,
                                seed=seed,
                                transpose=transpose,
                                version=version)
  r_dot = matvec_prob_normal_p.bind(vector_dot,
                                    w_mu=w_mu,
                                    w_sigma=w_sigma,
                                    conn_prob=conn_prob,
                                    shape=shape,
                                    seed=seed,
                                    transpose=transpose,
                                    version=version)
  return r, r_dot


def _matvec_prob_normal_transpose(
    ct, events, *, w_mu, w_sigma, conn_prob, shape, seed, transpose, version
):
  global vecmat_prob_normal_p
  if vecmat_prob_normal_p is None:
    from . import vecmat
    vecmat_prob_normal_p = vecmat.vecmat_prob_normal_p
  return vecmat_prob_normal_p.bind(ct[0],
                                   w_mu=w_mu,
                                   w_sigma=w_sigma,
                                   conn_prob=conn_prob,
                                   seed=seed,
                                   shape=shape,
                                   transpose=transpose,
                                   version=version)


matvec_prob_normal_p = Primitive('matvec_prob_normal')
matvec_prob_normal_p.multiple_results = True
matvec_prob_normal_p.def_abstract_eval(_matvec_prob_normal_abstract)
matvec_prob_normal_p.def_impl(partial(xla.apply_primitive, matvec_prob_normal_p))
xla.backend_specific_translations['cpu'][matvec_prob_normal_p] = _matvec_prob_normal_cpu_translation
xla.backend_specific_translations['gpu'][matvec_prob_normal_p] = _matvec_prob_normal_gpu_translation
register_general_batching(matvec_prob_normal_p)
ad.primitive_jvps[matvec_prob_normal_p] = _matvec_prob_normal_jvp
ad.primitive_transposes[matvec_prob_normal_p] = _matvec_prob_normal_transpose
