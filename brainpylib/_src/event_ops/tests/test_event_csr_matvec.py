# -*- coding: utf-8 -*-


import unittest
from functools import partial
import jax.numpy as jnp
import brainpy as bp
import brainpy.math as bm
import brainpylib
import jax


def sum_op(op):
  def func(*args, **kwargs):
    r = op(*args, **kwargs)
    return r.sum()

  return func


class Test_event_csr_matvec(unittest.TestCase):
  def __init__(self, *args, platform='cpu', **kwargs):
    super(Test_event_csr_matvec, self).__init__(*args, **kwargs)
    bm.set_platform(platform)
    print()

  def _test_homo(self, shape, transpose, homo_data):
    print(f'{self._test_homo.__name__}: shape = {shape}, transpose = {transpose}, homo_data = {homo_data}')

    rng = bm.random.RandomState()
    indices, indptr = bp.conn.FixedProb(0.4)(*shape).require('pre2post')
    indices = bm.as_jax(indices)
    indptr = bm.as_jax(indptr)
    events = rng.random(shape[0] if transpose else shape[1]) < 0.1
    heter_data = bm.ones(indices.shape).value * homo_data

    r1 = brainpylib.event_csr_matvec(homo_data, indices, indptr, events,
                                     shape=shape, transpose=transpose)
    r2 = brainpylib.event_csr_matvec(heter_data, indices, indptr, events,
                                     shape=shape, transpose=transpose)
    self.assertTrue(jnp.allclose(r1, r2))

    r3 = brainpylib.cusparse_csr_matvec(homo_data, indices, indptr, events.astype(float),
                                        shape=shape, transpose=transpose)
    self.assertTrue(jnp.allclose(r1, r3))

    dense = brainpylib.csr_to_dense(heter_data, indices, indptr, shape=shape)
    r4 = (events @ dense) if transpose else (dense @ events)
    self.assertTrue(jnp.allclose(r1, r4))

    r5 = brainpylib.event_csr_matvec(heter_data, indices, indptr, events.astype(float),
                                     shape=shape, transpose=transpose)
    self.assertTrue(jnp.allclose(r1, r5))

    bm.clear_buffer_memory()

  def _test_homo_vamp(self, shape, transpose, homo_data):
    print(f'{self._test_homo_vamp.__name__}: shape = {shape}, transpose = {transpose}, homo_data = {homo_data}')

    rng = bm.random.RandomState()
    indices, indptr = bp.conn.FixedProb(0.4)(*shape).require('pre2post')
    indices = bm.as_jax(indices)
    indptr = bm.as_jax(indptr)

    # vmap 'data'
    events = bm.as_jax(rng.random(shape[0] if transpose else shape[1])) < 0.1
    f1 = jax.vmap(partial(brainpylib.event_csr_matvec, indices=indices, indptr=indptr, events=events,
                          shape=shape, transpose=transpose))
    f2 = jax.vmap(partial(brainpylib.cusparse_csr_matvec, indices=indices, indptr=indptr, vector=events.astype(float),
                          shape=shape, transpose=transpose))
    vmap_data = bm.as_jax([homo_data] * 10)
    self.assertTrue(jnp.allclose(f1(vmap_data), f2(vmap_data)))

    # vmap 'events'
    f3 = jax.vmap(partial(brainpylib.event_csr_matvec, homo_data, indices, indptr,
                          shape=shape, transpose=transpose))
    f4 = jax.vmap(partial(brainpylib.cusparse_csr_matvec, homo_data, indices, indptr,
                          shape=shape, transpose=transpose))
    vmap_data = bm.as_jax(rng.random((10, shape[0] if transpose else shape[1]))) < 0.1
    self.assertTrue(jnp.allclose(f3(vmap_data), f4(vmap_data.astype(float))))

    # vmap 'data' and 'events'
    f5 = jax.vmap(lambda dd, ee: brainpylib.event_csr_matvec(dd, indices, indptr, ee,
                                                             shape=shape, transpose=transpose))
    f6 = jax.vmap(lambda dd, ee: brainpylib.cusparse_csr_matvec(dd, indices, indptr, ee,
                                                                shape=shape, transpose=transpose))
    vmap_data1 = bm.as_jax([homo_data] * 10)
    vmap_data2 = bm.as_jax(rng.random((10, shape[0] if transpose else shape[1]))) < 0.2
    self.assertTrue(jnp.allclose(f5(vmap_data1, vmap_data2),
                                f6(vmap_data1, vmap_data2.astype(float))))

    bm.clear_buffer_memory()

  def _test_homo_grad(self, shape, transpose, homo_data):
    print(f'{self._test_homo_grad.__name__}: shape = {shape}, transpose = {transpose}, homo_data = {homo_data}')

    rng = bm.random.RandomState()
    indices, indptr = bp.conn.FixedProb(0.4)(*shape).require('pre2post')
    indices = bm.as_jax(indices)
    indptr = bm.as_jax(indptr)
    events = bm.as_jax(rng.random(shape[0] if transpose else shape[1])) < 0.1
    dense_conn = brainpylib.csr_to_dense(bm.ones(indices.shape).value, indices, indptr, shape=shape)

    # grad 'data'
    r1 = jax.grad(sum_op(brainpylib.event_csr_matvec))(
      homo_data, indices, indptr, events, shape=shape, transpose=transpose)
    r2 = jax.grad(sum_op(brainpylib.cusparse_csr_matvec))(
      homo_data, indices, indptr, events.astype(float), shape=shape, transpose=transpose)
    self.assertTrue(jnp.allclose(r1, r2))
    r3 = jax.grad(sum_op(lambda a: (events @ (dense_conn * a) if transpose else
                                    ((dense_conn * a) @ events))))(homo_data)
    self.assertTrue(jnp.allclose(r1, r3))

    # grad 'events'
    r4 = jax.grad(sum_op(brainpylib.event_csr_matvec), argnums=3)(
      homo_data, indices, indptr, events.astype(float), shape=shape, transpose=transpose)
    r5 = jax.grad(sum_op(brainpylib.cusparse_csr_matvec), argnums=3)(
      homo_data, indices, indptr, events.astype(float), shape=shape, transpose=transpose)
    r6 = jax.grad(sum_op(lambda e: (e @ (dense_conn * homo_data) if transpose else
                                    ((dense_conn * homo_data) @ e))))(events.astype(float))
    self.assertTrue(jnp.allclose(r4, r5))
    self.assertTrue(jnp.allclose(r4, r6))

    bm.clear_buffer_memory()

  def test_homo(self):
    for transpose in [True, False]:
      for shape in [(100, 200),
                    (200, 200),
                    (300, 200),
                    (10, 1000),
                    (2, 10000),
                    (1000, 10),
                    (10000, 2)]:
        for homo_data in [-1., 0., 1.]:
          self._test_homo(shape, transpose, homo_data)

  def test_homo_vmap(self):
    for transpose in [True, False]:
      for shape in [(100, 200),
                    (200, 200),
                    (300, 200),
                    (10, 1000),
                    (2, 10000),
                    (1000, 10),
                    (100000, 2)]:
        for homo_data in [-1., 0., 1.]:
          self._test_homo_vamp(shape, transpose, homo_data)

  def test_homo_grad(self):
    for transpose in [True, False]:
      for shape in [(100, 200),
                    (200, 200),
                    (300, 200),
                    (10, 1000),
                    (2, 10000),
                    (1000, 10),
                    (100000, 2)]:
        for homo_data in [-1., 0., 1.]:
          self._test_homo_grad(shape, transpose, homo_data)

  def _test_heter(self, shape, transpose):
    print(f'{self._test_heter.__name__}: shape = {shape}, transpose = {transpose}')

    rng = bm.random.RandomState()
    indices, indptr = bp.conn.FixedProb(0.4)(*shape).require('pre2post')
    indices = bm.as_jax(indices)
    indptr = bm.as_jax(indptr)
    events = bm.as_jax(rng.random(shape[0] if transpose else shape[1])) < 0.1
    heter_data = bm.as_jax(rng.random(indices.shape))

    r1 = brainpylib.event_csr_matvec(heter_data, indices, indptr, events,
                                     shape=shape, transpose=transpose)
    r2 = brainpylib.cusparse_csr_matvec(heter_data, indices, indptr, events.astype(float),
                                        shape=shape, transpose=transpose)
    self.assertTrue(jnp.allclose(r1, r2))

    dense = brainpylib.csr_to_dense(heter_data, indices, indptr, shape=shape)
    r3 = (events @ dense) if transpose else (dense @ events)
    self.assertTrue(jnp.allclose(r1, r3))

    r4 = brainpylib.event_csr_matvec(heter_data, indices, indptr, events.astype(float),
                                     shape=shape, transpose=transpose)
    self.assertTrue(jnp.allclose(r1, r4))

    bm.clear_buffer_memory()

  def _test_heter_vamp(self, shape, transpose):
    print(f'{self._test_heter_vamp.__name__}: shape = {shape}, transpose = {transpose}')

    rng = bm.random.RandomState()
    indices, indptr = bp.conn.FixedProb(0.4)(*shape).require('pre2post')
    indices = bm.as_jax(indices)
    indptr = bm.as_jax(indptr)

    # vmap 'data'
    events = bm.as_jax(rng.random(shape[0] if transpose else shape[1])) < 0.1
    f1 = jax.vmap(partial(brainpylib.event_csr_matvec, indices=indices, indptr=indptr, events=events,
                          shape=shape, transpose=transpose))
    f2 = jax.vmap(partial(brainpylib.cusparse_csr_matvec, indices=indices, indptr=indptr, vector=events.astype(float),
                          shape=shape, transpose=transpose))
    vmap_data = bm.as_jax(rng.random((10, indices.shape[0])))
    self.assertTrue(jnp.allclose(f1(vmap_data), f2(vmap_data)))

    # vmap 'events'
    data = bm.as_jax(rng.random(indices.shape))
    f3 = jax.vmap(partial(brainpylib.event_csr_matvec, data, indices, indptr,
                          shape=shape, transpose=transpose))
    f4 = jax.vmap(partial(brainpylib.cusparse_csr_matvec, data, indices, indptr,
                          shape=shape, transpose=transpose))
    vmap_data = bm.as_jax(rng.random((10, shape[0] if transpose else shape[1]))) < 0.1
    self.assertTrue(jnp.allclose(f3(vmap_data), f4(vmap_data.astype(float))))

    # vmap 'data' and 'events'
    f5 = jax.vmap(lambda dd, ee: brainpylib.event_csr_matvec(dd, indices, indptr, ee,
                                                             shape=shape, transpose=transpose))
    f6 = jax.vmap(lambda dd, ee: brainpylib.cusparse_csr_matvec(dd, indices, indptr, ee,
                                                                shape=shape, transpose=transpose))
    vmap_data1 = bm.as_jax(rng.random((10, indices.shape[0])))
    vmap_data2 = bm.as_jax(rng.random((10, shape[0] if transpose else shape[1]))) < 0.2
    self.assertTrue(jnp.allclose(f5(vmap_data1, vmap_data2),
                                f6(vmap_data1, vmap_data2.astype(float))))

    bm.clear_buffer_memory()

  def _test_heter_grad(self, shape, transpose):
    print(f'{self._test_homo_grad.__name__}: shape = {shape}, transpose = {transpose}')

    rng = bm.random.RandomState()
    indices, indptr = bp.conn.FixedProb(0.4)(*shape).require('pre2post')
    indices = bm.as_jax(indices)
    indptr = bm.as_jax(indptr)
    events = rng.random(shape[0] if transpose else shape[1]) < 0.1
    dense_conn = brainpylib.csr_to_dense(bm.ones(indices.shape).value, indices, indptr, shape=shape)

    # grad 'data'
    data = rng.random(indices.shape)
    r1 = jax.grad(sum_op(brainpylib.event_csr_matvec))(
      data, indices, indptr, events, shape=shape, transpose=transpose)
    r2 = jax.grad(sum_op(brainpylib.cusparse_csr_matvec))(
      data, indices, indptr, events.astype(float), shape=shape, transpose=transpose)
    self.assertTrue(jnp.allclose(r1, r2))

    dense_data = brainpylib.csr_to_dense(data, indices, indptr, shape=shape)
    r3 = jax.grad(sum_op(lambda a: ((events @ a) if transpose else
                                    (a @ events))))(dense_data)
    rows, cols = brainpylib.csr_to_coo(indices, indptr)
    r3 = r3[rows, cols]
    self.assertTrue(jnp.allclose(r1, r3))

    # grad 'events'
    r4 = jax.grad(sum_op(brainpylib.event_csr_matvec), argnums=3)(
      data, indices, indptr, events.astype(float), shape=shape, transpose=transpose)
    r5 = jax.grad(sum_op(brainpylib.cusparse_csr_matvec), argnums=3)(
      data, indices, indptr, events.astype(float), shape=shape, transpose=transpose)
    r6 = jax.grad(sum_op(lambda e: ((e @ dense_data) if transpose else
                                    (dense_data @ e))))(events.astype(float))
    self.assertTrue(jnp.allclose(r4, r5))
    self.assertTrue(jnp.allclose(r4, r6))

    bm.clear_buffer_memory()

  def test_heter(self):
    for transpose in [True, False]:
      for shape in [(100, 200),
                    (200, 200),
                    (300, 200),
                    (10, 1000),
                    (2, 10000),
                    (1000, 10),
                    (10000, 2)]:
        self._test_heter(shape, transpose)

  def test_heter_vmap(self):
    for transpose in [True, False]:
      for shape in [(100, 200),
                    (200, 200),
                    (300, 200),
                    (10, 1000),
                    (2, 10000),
                    (1000, 10),
                    (100000, 2)]:
        self._test_heter_vamp(shape, transpose)

  def test_heter_grad(self):
    for transpose in [True, False]:
      for shape in [(100, 200),
                    (200, 200),
                    (300, 200),
                    (10, 1000),
                    (2, 10000),
                    (1000, 10),
                    (100000, 2)]:
        self._test_heter_grad(shape, transpose)
