# -*- coding: utf-8 -*-

import jax.numpy as jnp

__all__ = [
  'coo_to_csr',
  'csr_to_coo',
]


def coo_to_csr(pre_ids, post_ids, num_pre):
  """convert pre_ids, post_ids to (indices, indptr)."""
  # sorting
  sort_ids = jnp.argsort(pre_ids, kind='stable')
  post_ids = post_ids[sort_ids]

  indices = post_ids
  unique_pre_ids, pre_count = jnp.unique(pre_ids, return_counts=True)
  final_pre_count = jnp.zeros(num_pre)
  final_pre_count[unique_pre_ids] = pre_count
  indptr = final_pre_count.cumsum()
  indptr = jnp.insert(indptr, 0, 0)
  return indices, indptr


def csr_to_coo(indices, indptr):
  """Given CSR (indices, indptr) return COO (row, col)"""
  return jnp.cumsum(jnp.zeros_like(indices).at[indptr].add(1)) - 1, indices




