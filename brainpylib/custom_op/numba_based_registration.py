# -*- coding: utf-8 -*-

import collections.abc
from functools import partial
from typing import Callable, Union, Sequence

import numba
import numpy as np
from jax import core
from jax.abstract_arrays import ShapedArray
from jax.interpreters import xla, batching, ad
from numba.core.dispatcher import Dispatcher

from .numba_cpu_translation import func_cpu_translation
from .numba_gpu_translation import func_gpu_translation


__all__ = [
  'register_op_by_nbjit',
  'register_op',
]


def register_op_by_nbjit(
    op_name: str,
    cpu_func: Callable,
    out_shapes: Union[Callable, ShapedArray, Sequence[ShapedArray]],
    gpu_func_translation: Callable = None,
    batching_translation: Callable = None,
    jvp_translation: Callable = None,
    transpose_translation: Callable = None,
    apply_cpu_func_to_gpu: bool = False,
):
  """
  Converting the numba-jitted function in a Jax/XLA compatible primitive.

  Parameters
  ----------
  op_name: str
    Name of the operators.

  cpu_func: Callable
    A callable numba-jitted function or pure function (can be lambda function) running on CPU.

  out_shapes: Callable, ShapedArray, Sequence[ShapedArray], default = None
    Outputs shapes of target function. `out_shapes` can be a `ShapedArray` or
    a sequence of `ShapedArray`. If it is a function, it takes as input the argument
    shapes and dtypes and should return correct output shapes of `ShapedArray`.

  gpu_func_translation: Callable
    A callable cuda-jitted kernel running on GPU.

  batching_translation

  jvp_translation
    The forward autodiff translation rule.

  transpose_translation

  apply_cpu_func_to_gpu: bool,
    True when gpu_func is implemented on CPU and other logics(data transfer) is implemented on GPU.
    Default is True.

  Returns
  -------
  op: callable
    A jitable JAX function.
  """
  if gpu_func_translation is not None:
    raise RuntimeError('Currently cuda.jit function is not supported to convert into a Jax/XLA compatible primitive.'
                       ' Please wait for future version to use gpu_func. Now we support to set '
                       'apply_cpu_func_to_gpu = True for a alternative method to run on GPU.')

  if out_shapes is None:
    raise RuntimeError('out_shapes cannot be None. It can be a `ShapedArray` or '
                       'a sequence of `ShapedArray`. If it is a function, it takes as input the argument '
                       'shapes and dtypes and should return correct output shapes of `ShapedArray`.')

  if (gpu_func_translation is not None) and apply_cpu_func_to_gpu:
    raise RuntimeError("apply_cpu_func_to_gpu cannot be true if gpu_func is not None.")

  prim = core.Primitive(op_name)
  prim.multiple_results = True

  # user defined function
  if not isinstance(cpu_func, Dispatcher):
    cpu_func = numba.jit(fastmath=True, nopython=True)(cpu_func)

  # output shape evaluation function
  def abs_eval_rule(*input_shapes, **info):
    if callable(out_shapes):
      shapes = out_shapes(*input_shapes, **info)
    elif isinstance(out_shapes, ShapedArray):
      shapes = [out_shapes]
    elif isinstance(out_shapes, (tuple, list)):
      shapes = out_shapes
      for elem in out_shapes:
        if not isinstance(elem, ShapedArray):
          raise ValueError(f'Elements in "out_shapes" must be instances of '
                           f'jax.abstract_arrays.ShapedArray, but we got '
                           f'{type(elem)}: {elem}')
    else:
      raise ValueError(f'Unknown type {type(out_shapes)}, only '
                       f'supports function, ShapedArray or '
                       f'list/tuple of ShapedArray.')

    # output shapes
    if not isinstance(shapes, collections.abc.Collection):
      return [shapes]
    else:
      return shapes

  # output evaluation function
  def eval_rule(*inputs, **info):
    # compute the output shapes
    output_shapes = abs_eval_rule(*inputs, **info)
    # Preallocate the outputs
    outputs = tuple(np.zeros(shape.shape, dtype=shape.dtype) for shape in output_shapes)
    # convert inputs to a tuple
    inputs = tuple(np.asarray(arg) for arg in inputs)
    inputs += tuple(np.asarray(i) for i in info.values())
    # call the kernel
    cpu_func(outputs, inputs)
    # Return the outputs
    return outputs[0] if len(outputs) == 1 else tuple(outputs)

  # cpu function
  prim.def_abstract_eval(abs_eval_rule)
  prim.def_impl(eval_rule)
  xla.backend_specific_translations['cpu'][prim] = partial(func_cpu_translation, cpu_func, abs_eval_rule)
  if apply_cpu_func_to_gpu:
    xla.backend_specific_translations['gpu'][prim] = partial(func_gpu_translation, cpu_func, abs_eval_rule)

  # gpu function
  if gpu_func_translation is not None:
    xla.backend_specific_translations['gpu'][prim] = gpu_func_translation

  # batching
  if batching_translation is not None:
    batching.primitive_batchers[prim] = batching_translation

  # jvp
  if jvp_translation is not None:
    ad.primitive_jvps[prim] = jvp_translation

  # transpose
  if transpose_translation is not None:
    ad.primitive_transposes[prim] = transpose_translation

  return prim


register_op = register_op_by_nbjit
