# -*- coding: utf-8 -*-

from functools import partial
from typing import Callable, Union, Sequence

import numba
import numpy as np
from jax import core, numpy as jnp
from jax.abstract_arrays import ShapedArray
from jax.interpreters import xla, batching, ad
from numba.core.dispatcher import Dispatcher

from .cpu_translation import cpu_translation
from .gpu2cpu_translation import gpu2cpu_translation


__all__ = [
  'register_op_with_numba',
]


def register_op_with_numba(
    op_name: str,
    cpu_func: Callable,
    out_shapes: Union[Callable, ShapedArray, Sequence[ShapedArray]],
    gpu_func_translation: Callable = None,
    batching_translation: Callable = None,
    jvp_translation: Callable = None,
    transpose_translation: Callable = None,
    apply_cpu_func_to_gpu: bool = False,
    multiple_results: bool = False,
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

  batching_translation: Callable
    The batching translation for the primitive.

  jvp_translation: Callable
    The forward autodiff translation rule.

  transpose_translation: Callable
    The backward autodiff translation rule.

  apply_cpu_func_to_gpu: bool
    True when gpu_func is implemented on CPU and other logics(data transfer) is implemented on GPU.
    Default is False.

  multiple_results: bool
    Whether the primitive returns multiple results. Default is False.

  Returns
  -------
  op: core.Primitive
    A JAX Primitive object.
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
  prim.multiple_results = multiple_results

  # user defined function
  if not isinstance(cpu_func, Dispatcher):
    cpu_func = numba.jit(fastmath=True, nopython=True)(cpu_func)

  # output shape evaluation function
  def abs_eval_rule(*input_shapes, **info):
    if callable(out_shapes):
      shapes = out_shapes(*input_shapes, **info)
    else:
      shapes = out_shapes

    if isinstance(shapes, ShapedArray):
      pass
    elif isinstance(shapes, (tuple, list)):
      for elem in shapes:
        if not isinstance(elem, ShapedArray):
          raise ValueError(f'Elements in "out_shapes" must be instances of '
                           f'jax.abstract_arrays.ShapedArray, but we got '
                           f'{type(elem)}: {elem}')
    else:
      raise ValueError(f'Unknown type {type(shapes)}, only '
                       f'supports function, ShapedArray or '
                       f'list/tuple of ShapedArray.')
    return shapes

  # output evaluation function
  def eval_rule(*inputs, **info):
    # compute the output shapes
    output_shapes = abs_eval_rule(*inputs, **info)
    # Preallocate the outputs
    if isinstance(output_shapes, ShapedArray):
      outputs = np.zeros(output_shapes.shape, dtype=output_shapes.dtype)
      assert not multiple_results
    else:
      assert multiple_results
      outputs = tuple(np.zeros(shape.shape, dtype=shape.dtype) for shape in output_shapes)
    # convert inputs to a tuple
    inputs = tuple(np.asarray(arg) for arg in inputs)
    inputs += tuple(np.asarray(i) for i in info.values())
    # call the kernel
    cpu_func(outputs, inputs)
    # Return the outputs
    return tuple(jnp.asarray(out) for out in outputs) if multiple_results else jnp.asarray(outputs)

  # cpu function
  prim.def_abstract_eval(abs_eval_rule)
  prim.def_impl(eval_rule)
  xla.backend_specific_translations['cpu'][prim] = partial(cpu_translation,
                                                           cpu_func,
                                                           abs_eval_rule,
                                                           multiple_results)
  if apply_cpu_func_to_gpu:
    xla.backend_specific_translations['gpu'][prim] = partial(gpu2cpu_translation,
                                                             cpu_func,
                                                             abs_eval_rule,
                                                             multiple_results)

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


