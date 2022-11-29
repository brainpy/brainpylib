# -*- coding: utf-8 -*-


from functools import partial

import jax.numpy as jnp
import numpy as np
from jax import core
from jax.abstract_arrays import ShapedArray
from jax.interpreters import xla
from jax.lib import xla_client

try:
  from . import gpu_ops
except ImportError:
  gpu_ops = None

__all__ = [
  'Event_MaskedLR_FixedProb',
]


class Event_MaskedLR_FixedProb(object):
  """An operator computes :math:`event @ [(L @ R) * M]`,
  where :math:`L` and :math:`R` are decomposed matrices,
  and :math:`M` are boolean valued matrix with the fixed
  connection probability.

  """

  _op_names = {
    'v1': {'fp': {'k=4': b'event_mmm_fp_v1_4x64x256',
                  'k=8': b'event_mmm_fp_v1_8x32x256',
                  'k=16': b'event_mmm_fp_v1_16x16x256',
                  'k=32': b'event_mmm_fp_v1_32x8x256'}},
    'v2': {'fp': {'k=4': b'event_mmm_fp_v2_4x64x256',
                  'k=8': b'event_mmm_fp_v2_8x32x256',
                  'k=16': b'event_mmm_fp_v2_16x16x256',
                  'k=32': b'event_mmm_fp_v2_32x8x256', }},
  }

  @staticmethod
  def _get_k(k):
    all_ks = np.asarray([4, 8, 16, 32])
    ids = np.where((all_ks - k) >= 0)[0]
    if len(ids) > 0:
      return all_ks[ids[0]]
    raise ValueError

  def __init__(self, seed, n, p, k, version='v1'):
    self.seed = seed
    self.n = n
    self.k = k
    if version == 'v1':
      self.p = float(np.log((1 - p) if p < 1 else 1e-40).astype(np.float32))
    elif version == 'v2':
      self.p = p
    else:
      raise ValueError
    self.fn = self._op_names[version]['fp'][f'k={self._get_k(k)}']
    self.ones = jnp.ones((self.k,), dtype=jnp.float32)

  def __call__(self, events, L, R):
    if L.dtype != jnp.float32:
      raise TypeError(f'"L" must be a matrix of float32, while we got {L.dtype}.')
    if R.dtype != jnp.float32:
      raise TypeError(f'"R" must be a matrix of float32, while we got {R.dtype}.')
    assert L.ndim == R.ndim == 2, '"L" and "R" must be a matrix.'
    assert L.shape[1] == self.k == R.shape[0]
    if events.dtype != jnp.bool_:
      raise TypeError(f'"events" must be a vector of bool, while we got {events.dtype}')
    assert events.shape[0] == L.shape[0]
    assert R.shape[1] == self.n
    temp = event_mmm_prim.bind(events,
                               L.T,
                               p=self.p,
                               k=self.k,
                               m=L.shape[0],
                               n=self.n,
                               fn=self.fn,
                               seed=self.seed)  # R^(k, n)
    return self.ones @ (temp * R)  # R^n


def event_mmm_abstract(events, mat, *, p, k, m, n, seed, fn):
  return ShapedArray(shape=(k, n), dtype=mat.dtype)


def event_mmm_translation_gpu(c, events, mat, *, p, k, m, n, seed, fn):
  if gpu_ops is None:
    raise ValueError('Cannot find compiled gpu wheels.')

  opaque = gpu_ops.build_matmul_descriptor(m, k, n, seed, p)
  return xla_client.ops.CustomCallWithLayout(
    c,
    fn,
    operands=(events, mat),
    operand_shapes_with_layout=(c.get_shape(events), c.get_shape(mat)),
    shape_with_layout=xla_client.Shape.array_shape(np.dtype(c.get_shape(mat).element_type()), (k, n), (1, 0)),
    opaque=opaque,
  )


event_mmm_prim = core.Primitive("event_mat_mtp_mask")
event_mmm_prim.def_abstract_eval(event_mmm_abstract)
event_mmm_prim.def_impl(partial(xla.apply_primitive, event_mmm_prim))
xla.backend_specific_translations["gpu"][event_mmm_prim] = event_mmm_translation_gpu
