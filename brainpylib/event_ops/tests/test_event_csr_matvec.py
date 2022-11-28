# -*- coding: utf-8 -*-


import unittest
from functools import partial

import brainpy as bp
import brainpy.math as bm
import brainpylib
import jax

bm.set_platform('cpu')


class Test_event_csr_matvec(unittest.TestCase):
  def __init__(self, *args, **kwargs):
    super(Test_event_csr_matvec, self).__init__(*args, **kwargs)
    self.rng = bm.random.RandomState(123)

  def test_homo(self):
    shape = (10, 50)

    indices, indptr = bp.conn.FixedProb(0.4, seed=123)(*shape).require('pre2post')
    indices = bm.as_jax(indices)
    indptr = bm.as_jax(indptr)
    values = 1.
    events = self.rng.random(shape[1]).value < 0.1

    f = partial(brainpylib.event_csr_matvec, shape=shape)
    r1 = f(values, indices, indptr, events)
    r2 = jax.jit(f)(values, indices, indptr, events)
    self.assertTrue(bm.allclose(r1, r2))

    r3 = brainpylib.cusparse_csr_matvec(values, indices, indptr, events.astype(float), shape=shape)
    self.assertTrue(bm.allclose(r1, r3))

  def test_homo_grad(self):
    shape = (10, 5)

    indices, indptr = bp.conn.FixedProb(0.4, seed=123)(*shape).require('pre2post')
    indices = bm.as_jax(indices)
    indptr = bm.as_jax(indptr)
    values = 1.
    events = self.rng.random(shape[1]).value < 0.1

    f1 = partial(brainpylib.event_csr_matvec, indices=indices, indptr=indptr, events=events, shape=shape)
    f2 = partial(brainpylib.cusparse_csr_matvec, indices=indices, indptr=indptr, vector=events.astype(float), shape=shape)
    r1 = jax.grad(lambda a: f1(a).sum())(values)
    r2 = jax.grad(lambda a: f2(a).sum())(values)
    print(r1, r2)





    # print('grad_f1:')
    # print(grad_f1(values, events))
    #
    # grad_f2 = jax.grad(lambda d, e: f(d, indices, indptr, e).sum(), argnums=1)
    # print('grad_f2:')
    # print(grad_f2(values, events.astype(float)))
    #
    # grad_f3 = jax.grad(lambda d, e: f(d, indices, indptr, e).mean(), argnums=(0, 1))
    # print('grad_f3:')
    # print(grad_f3(values, events.astype(float)))
    #
    # vmap_grad_f1 = jax.vmap(grad_f1, in_axes=(0, None))
    # print('vmap_grad_f1:')
    # print(vmap_grad_f1(bm.ones(5).value * values, events))
    #
    # vmap_grad_f2 = jax.jit(jax.vmap(grad_f2, in_axes=(None, 0)))
    # print('vmap_grad_f2:')
    # print(vmap_grad_f2(values, bm.as_jax([events.astype(float),
    #                                       events.astype(float),
    #                                       events.astype(float)])))

  def test_homo_vmap(self):
    shape = (10, 5)

    events = self.rng.random(shape[1]).value < 0.1
    indices, indptr = bp.conn.FixedProb(0.4, seed=123)(*shape).require('pre2post')
    indices = bm.as_jax(indices)
    indptr = bm.as_jax(indptr)
    f = partial(brainpylib.event_csr_matvec, indices=indices, indptr=indptr, events=events, shape=shape)
    vmap_f1 = jax.vmap(f)

    values = 1.

    print('vmap_f1:')
    print(vmap_f1(bm.ones(5).value * values, ))

    grad_vmap_f1 = jax.grad(lambda dd: vmap_f1(dd).sum())
    print('grad_vmap_f1:')
    print(grad_vmap_f1(bm.ones(5).value * values))

  def test_homo_vmap_vmap(self):
    shape = (10, 50)

    indices, indptr = bp.conn.FixedProb(0.4, seed=123)(*shape).require('pre2post')
    indices = bm.as_jax(indices)
    indptr = bm.as_jax(indptr)
    events = self.rng.random(shape[1]).value < 0.1

    f = partial(brainpylib.event_csr_matvec, indices=indices, indptr=indptr, events=events, shape=shape)
    vmap_f1 = jax.vmap(jax.vmap(f))

    print('vmap_vmap_f1:')
    print(vmap_f1(bm.ones((5, 5)).value))

  def test_transpose_homo(self):
    events = self.rng.random(10).value < 0.1
    indices, indptr = bp.conn.FixedProb(0.4, seed=123)(10, 5).require('pre2post')
    indices = bm.as_jax(indices)
    indptr = bm.as_jax(indptr)

    values = 1.

    f = partial(brainpylib.event_csr_matvec, shape=(10, 5), transpose=True)
    print('f:')
    print(f(values, indices, indptr, events))

    jit_f = jax.jit(f)
    print('jit_f:')
    print(jit_f(values, indices, indptr, events))

  def test_transpose_homo_grad(self):
    events = self.rng.random(10).value < 0.1
    indices, indptr = bp.conn.FixedProb(0.4, seed=123)(10, 5).require('pre2post')
    indices = bm.as_jax(indices)
    indptr = bm.as_jax(indptr)
    f = partial(brainpylib.event_csr_matvec, shape=(10, 5), transpose=True)

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

    vmap_grad_f1 = jax.vmap(grad_f1, in_axes=(0, None))
    print('vmap_grad_f1:')
    print(vmap_grad_f1(bm.ones(5).value * values, events))

    vmap_grad_f2 = jax.jit(jax.vmap(grad_f2, in_axes=(None, 0)))
    print('vmap_grad_f2:')
    print(vmap_grad_f2(values, bm.as_jax([events.astype(float),
                                          events.astype(float),
                                          events.astype(float)])))

  def test_transpose_homo_vmap(self):
    events = self.rng.random(10).value < 0.1
    indices, indptr = bp.conn.FixedProb(0.4, seed=123)(10, 5).require('pre2post')
    indices = bm.as_jax(indices)
    indptr = bm.as_jax(indptr)
    f = partial(brainpylib.event_csr_matvec, shape=(10, 5), transpose=True)

    values = 1.

    vmap_f1 = jax.vmap(f, in_axes=(0, None, None, None))
    print('vmap_f1:')
    print(vmap_f1(bm.ones(5).value * values, indices, indptr, events))

    grad_vmap_f1 = jax.grad(
      lambda dd: (jax.vmap(lambda d: f(d, indices, indptr, events))(dd)).sum(),
      argnums=0
    )
    print('grad_vmap_f1:')
    print(grad_vmap_f1(bm.ones(5).value * values))

  def test_transpose_homo_vmap_vmap(self):
    shape = (10, 50)

    indices, indptr = bp.conn.FixedProb(0.4, seed=123)(*shape).require('pre2post')
    indices = bm.as_jax(indices)
    indptr = bm.as_jax(indptr)
    events = self.rng.random(shape[0]).value < 0.1

    f = partial(brainpylib.event_csr_matvec, indices=indices, indptr=indptr, events=events,
                shape=shape, transpose=True)
    vmap_f1 = jax.vmap(jax.vmap(f))

    print('vmap_vmap_f1:')
    print(vmap_f1(bm.ones((5, 5)).value))

