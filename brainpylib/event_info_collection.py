# -*- coding: utf-8 -*-

import jax.numpy as jnp
import numba
from jax import dtypes
from jax.core import ShapedArray
from jax.interpreters import xla, batching
from jax.lib import xla_client

from brainpylib.custom_op import register_op_with_numba
from brainpylib.utils import GPUOperatorNotFound

try:
  from . import gpu_ops
except ImportError:
  gpu_ops = None

__all__ = [
  'event_info'
]


def event_info(events):
  """Collect
  """
  if events.dtype != jnp.bool_:
    raise TypeError('Only support bool.')
  if events.ndim != 1:
    raise TypeError('Only support 1D boolean vector.')
  return event_info_p.bind(events)


def _batch_event_info_abstract(events):
  assert events.ndim == 2
  assert events.dtype == jnp.bool_
  event_ids = ShapedArray(dtype=dtypes.canonicalize_dtype(int), shape=events.shape)
  event_num = ShapedArray(dtype=dtypes.canonicalize_dtype(int), shape=(events.shape[0],))
  return event_ids, event_num


@numba.njit(fastmath=True, parallel=True, nogil=True)
def _batch_event_info(outs, ins):
  event_ids, event_num = outs
  event_num.fill(0)
  event_ids.fill(-1)
  events = ins
  for batch_idx in range(event_ids.shape[0]):
    num = 0
    for i in range(event_ids.shape[1]):
      if events[batch_idx, i]:
        event_ids[batch_idx, num] = i
        num += 1
    event_num[batch_idx] = num


def _batch_event_info_batching_rule(args, axes):
  arg = jnp.moveaxis(args[0], axes[0], 0)
  shape = arg.shape
  arg = jnp.reshape(arg, (shape[0] * shape[1], shape[2]))
  event_ids, event_num = batch_event_info_p.bind(arg)
  return ((jnp.reshape(event_ids, shape), jnp.reshape(event_num, shape[:2])),
          (0, 0))


def _event_info_translation(c, events):
  if gpu_ops is None:
    raise GPUOperatorNotFound(event_info_p.name)

  e_shape = c.get_shape(events).dimensions()
  if len(e_shape) == 1:
    event_size = e_shape[0]
    batch_size = 1
    event_ids_shape = xla_client.Shape.array_shape(dtypes.canonicalize_dtype(int),
                                                   (event_size,),
                                                   (0,))
  else:
    batch_size, event_size = e_shape
    event_ids_shape = xla_client.Shape.array_shape(dtypes.canonicalize_dtype(int),
                                                   (batch_size, event_size),
                                                   (1, 0))
  event_num_shape = xla_client.Shape.array_shape(dtypes.canonicalize_dtype(int),
                                                 (batch_size,),
                                                 (0,))
  opaque = gpu_ops.build_nonzero_descriptor(event_size, batch_size)
  return xla_client.ops.CustomCallWithLayout(
    c,
    b'nonzero_64',
    operands=(events,),
    operand_shapes_with_layout=(c.get_shape(events),),
    shape_with_layout=xla_client.Shape.tuple_shape((event_ids_shape, event_num_shape)),
    opaque=opaque,
  )


batch_event_info_p = register_op_with_numba(
  op_name='event_info',
  cpu_func=_batch_event_info,
  out_shapes=_batch_event_info_abstract,
  apply_cpu_func_to_gpu=True if gpu_ops is None else False,
  multiple_results=True
)
batching.primitive_batchers[batch_event_info_p] = _batch_event_info_batching_rule
xla.backend_specific_translations["gpu"][batch_event_info_p] = _event_info_translation


def _event_info_abstract(events):
  assert events.ndim == 1
  assert events.dtype == jnp.bool_
  event_ids = ShapedArray(dtype=dtypes.canonicalize_dtype(int), shape=events.shape)
  event_num = ShapedArray(dtype=dtypes.canonicalize_dtype(int), shape=(1,))
  return event_ids, event_num


@numba.njit(fastmath=True)
def _event_info(outs, ins):
  event_ids, event_num = outs
  event_num.fill(0)
  event_ids.fill(-1)
  events = ins
  num = 0
  for i in range(event_ids.shape[0]):
    if events[i]:
      event_ids[num] = i
      num += 1
  event_num[0] = num


def _event_info_batching_rule(args, axes):
  arg = jnp.moveaxis(args[0], axes[0], 0)
  return (batch_event_info_p.bind(arg), (0, 0))


event_info_p = register_op_with_numba(
  op_name='event_info',
  cpu_func=_event_info,
  out_shapes=_event_info_abstract,
  apply_cpu_func_to_gpu=True if gpu_ops is None else False,
  multiple_results=True
)
batching.primitive_batchers[event_info_p] = _event_info_batching_rule
xla.backend_specific_translations["gpu"][event_info_p] = _event_info_translation