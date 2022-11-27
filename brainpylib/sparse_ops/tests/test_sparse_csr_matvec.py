# -*- coding: utf-8 -*-

import unittest
import brainpylib

import brainpy as bp
import brainpy.math as bm

bm.set_platform('cpu')


class Test_csr_matvec(unittest.TestCase):
  def test_csr_matvec_homo(self):
    conn = bp.conn.FixedProb(0.1)
    for v in [-1., 0., 0.1, 1.]:
      for shape in [(100, 200), (300, 200), (10, 1000)]:
        print(f'shape = {shape}, v = {v}')

        # matrix
        indices, indptr = conn(*shape).require('pre2post')
        indices = bm.as_jax(indices)
        indptr = bm.as_jax(indptr)
        # vector
        rng = bm.random.RandomState(123)
        vector = rng.random(shape[1]).value
        r1 = brainpylib.csr_matvec(v, indices, indptr, vector, shape=shape)
        data = bm.ones(indices.shape).value * v

        r2 = brainpylib.csr_matvec(data, indices, indptr, vector, shape=shape)
        self.assertTrue(bm.allclose(r1, r2))

        r3 = brainpylib.cusparse_csr_matvec(data, indices, indptr, vector, shape=shape)
        self.assertTrue(bm.allclose(r1, r3))

        dense = brainpylib.csr_to_dense(data, indices, indptr, shape=shape)
        r4 = dense @ vector
        self.assertTrue(bm.allclose(r1, r4))

  def test_csr_matvec_heter_1(self):
    conn = bp.conn.FixedProb(0.1)
    rng = bm.random.RandomState(123)

    for shape in [(100, 200), (100, 100), (200, 100)]:
      print(f'shape = {shape}')

      indices, indptr = conn(*shape).require('pre2post')
      indices = bm.as_jax(indices)
      indptr = bm.as_jax(indptr)
      data = rng.random(indices.shape).value
      vector = rng.random(shape[1]).value

      r1 = brainpylib.csr_matvec(data, indices, indptr, vector, shape=shape)

      dense = brainpylib.csr_to_dense(data, indices, indptr, shape=shape)
      r2 = dense @ vector
      self.assertTrue(bm.allclose(r1, r2))

      r3 = brainpylib.cusparse_csr_matvec(data, indices, indptr, vector, shape=shape)
      self.assertTrue(bm.allclose(r1, r3))

