# -*- coding: utf-8 -*-


import unittest
from functools import partial

import brainpy as bp
import brainpy.math as bm
import jax
import pytest

from brainpylib import event_csr_matvec

if jax.default_backend() != 'gpu':
  pytest.skip("No gpu available.", allow_module_level=True)


class TestEventCSRMatVec(unittest.TestCase):
  def __init__(self, *args, **kwargs):
    super(TestEventCSRMatVec, self).__init__(*args, **kwargs)
    self.rng = bm.random.RandomState(123)
    print()

  def test_homo(self):
    shape = (100, 500)

    indices, indptr = bp.conn.FixedProb(0.4, seed=123)(*shape).require('pre2post')
    indices = bm.as_jax(indices)
    indptr = bm.as_jax(indptr)
    values = 1.
    events = self.rng.random(shape[1]).value < 0.1

    f = partial(event_csr_matvec, shape=shape)

    r1 = f(values, indices, indptr, events)
    r2 = jax.jit(f)(values, indices, indptr, events)
    print(r1)
    print(r2)
    self.assertTrue(bm.allclose(r1, r2))

  def test_homo_grad(self):
    shape = (100, 500)

    indices, indptr = bp.conn.FixedProb(0.4, seed=123)(*shape).require('pre2post')
    indices = bm.as_jax(indices)
    indptr = bm.as_jax(indptr)
    values = 1.
    events = self.rng.random(shape[1]).value < 0.1

    f = partial(event_csr_matvec, shape=shape)

    grad_f1 = jax.grad(lambda d, e: f(d, indices, indptr, e).sum(), argnums=0)
    print('grad_f1:')
    print(grad_f1(values, events))

    grad_f2 = jax.grad(lambda d, e: f(d, indices, indptr, e).sum(), argnums=1)
    print('grad_f2:')
    print(grad_f2(values, events.astype(float)))

    grad_f3 = jax.grad(lambda d, e: f(d, indices, indptr, e).mean(), argnums=(0, 1))
    print('grad_f3:')
    print(grad_f3(values, events.astype(float)))

  def test_homo_transpose(self):
    events = self.rng.random(10).value < 0.1
    indices, indptr = bp.conn.FixedProb(0.4, seed=123)(10, 5).require('pre2post')
    indices = bm.as_jax(indices)
    indptr = bm.as_jax(indptr)

    values = 1.

    f = partial(event_csr_matvec, shape=(10, 5), transpose=True)
    print('f:')
    print(f(values, indices, indptr, events))

    jit_f = jax.jit(f)
    print('jit_f:')
    print(jit_f(values, indices, indptr, events))

  def test_homo_transpose_grad(self):
    events = self.rng.random(10).value < 0.1
    indices, indptr = bp.conn.FixedProb(0.4, seed=123)(10, 5).require('pre2post')
    indices = bm.as_jax(indices)
    indptr = bm.as_jax(indptr)
    f = partial(event_csr_matvec, shape=(10, 5), transpose=True)

    values = 1.

    grad_f1 = jax.grad(lambda d, e: f(d, indices, indptr, e).sum(), argnums=0)
    print('grad_f1:')
    print(grad_f1(values, events))

    grad_f2 = jax.grad(lambda d, e: f(d, indices, indptr, e).sum(), argnums=1)
    print('grad_f2:')
    print(grad_f2(values, events.astype(float)))

    grad_f3 = jax.grad(lambda d, e: f(d, indices, indptr, e).mean(), argnums=(0, 1))
    print('grad_f3:')
    print(grad_f3(values, events.astype(float)))

    # vmap_grad_f1 = jax.vmap(grad_f1, in_axes=(0, None))
    # print('vmap_grad_f1:')
    # print(vmap_grad_f1(bm.ones(5).value * values, events))
    #
    # vmap_grad_f2 = jax.jit(jax.vmap(grad_f2, in_axes=(None, 0)))
    # print('vmap_grad_f2:')
    # print(vmap_grad_f2(values, bm.as_jax([events.astype(float),
    #                                       events.astype(float),
    #                                       events.astype(float)])))


