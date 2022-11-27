# -*- coding: utf-8 -*-


import unittest
from brainpylib import event_csr_matvec
import brainpy as bp
import brainpy.math as bm
import jax
from functools import partial


bm.set_platform('cpu')


class TestEventSCRMatVec(unittest.TestCase):
  def test_homo_value(self):
    rng = bm.random.RandomState(123)
    events = rng.random(10) < 0.1
    indices, indptr = bp.conn.FixedProb(0.4, seed=123)(10, 5).require('pre2post')

    values = 1.

    f = partial(event_csr_matvec, shape=(10, 5), transpose=True)
    print('f:')
    print(f(values, bm.as_jax(indices), bm.as_jax(indptr), events.value))

    jit_f = jax.jit(f)
    print('jit_f:')
    print(jit_f(values, bm.as_jax(indices), bm.as_jax(indptr), events.value))

    grad_f1 = jax.grad(lambda d, e: f(d, bm.as_jax(indices), bm.as_jax(indptr), e).sum(), argnums=0)
    print('grad_f1:')
    print(grad_f1(values, events.value))

    grad_f2 = jax.grad(lambda d, e: f(d, bm.as_jax(indices), bm.as_jax(indptr), e).sum(), argnums=1)
    print('grad_f2:')
    print(grad_f2(values, events.value.astype(float)))

    grad_f3 = jax.grad(lambda d, e: f(d, bm.as_jax(indices), bm.as_jax(indptr), e).mean(), argnums=(0, 1))
    print('grad_f3:')
    print(grad_f3(values, events.value.astype(float)))

    vmap_f1 = jax.vmap(f, in_axes=(0, None, None, None))
    print('vmap_f1:')
    print(vmap_f1(bm.ones(5).value * values, bm.as_jax(indices), bm.as_jax(indptr), events.value))

    vmap_grad_f1 = jax.vmap(grad_f1, in_axes=(0, None))
    print('vmap_grad_f1:')
    print(vmap_grad_f1(bm.ones(5).value * values, events.value))

    vmap_grad_f2 = jax.jit(jax.vmap(grad_f2, in_axes=(None, 0)))
    print('vmap_grad_f2:')
    print(vmap_grad_f2(values, bm.as_jax([events.value.astype(float),
                                          events.value.astype(float),
                                          events.value.astype(float)])))

    grad_vmap_f1 = jax.grad(
      lambda dd: (jax.vmap(lambda d: f(d, bm.as_jax(indices), bm.as_jax(indptr), events.value))(dd)).sum(),
      argnums=0
    )
    print('grad_vmap_f1:')
    print(grad_vmap_f1(bm.ones(5).value * values))



