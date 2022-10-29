# -*- coding: utf-8 -*-


__all__ = [
  'GPUOperatorNotFound',
  'set_op_setting',
  'get_op_setting',
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


