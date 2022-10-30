# -*- coding: utf-8 -*-


from functools import partial

import jax.numpy as jnp
from jax import lax
from jax.interpreters import batching

__all__ = [
  'GPUOperatorNotFound',
  'set_op_setting',
  'get_op_setting',

  'register_general_batching',
]


class GPUOperatorNotFound(Exception):
  def __init__(self, name):
    super(GPUOperatorNotFound, self).__init__(f'''
GPU operator for "{name}" does not found. 

Please compile brainpylib GPU operators with the guidance in the following link:

https://brainpy.readthedocs.io/en/latest/tutorial_advanced/compile_brainpylib.html
    ''')


DEFAULT_SETTING = dict(PARALLEL=False, NOGIL=False)

op_numba_setting = dict(
)


def set_op_setting(op_name, **settings):
  op_numba_setting[op_name] = settings


def get_op_setting(op_name):
  return op_numba_setting[op_name]


def _general_batching_rule(prim, args, axes, **kwargs):
  batch_axes, batch_args, non_batch_args = [], {}, {}
  for ax_i, ax in enumerate(axes):
    if ax is None:
      non_batch_args[f'ax{ax_i}'] = args[ax_i]
    else:
      batch_args[f'ax{ax_i}'] = args[ax_i] if ax == 0 else jnp.moveaxis(args[ax_i], ax, 0)
      batch_axes.append(ax_i)

  def f(_, x):
    pars = tuple([(x[f'ax{i}'] if i in batch_axes else non_batch_args[f'ax{i}'])
                  for i in range(len(axes))])
    return 0, prim.bind(*pars, **kwargs)

  _, outs = lax.scan(f, 0, batch_args)
  return outs, 0


def register_general_batching(prim):
  batching.primitive_batchers[prim] = partial(_general_batching_rule, prim)


