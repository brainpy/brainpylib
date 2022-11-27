# -*- coding: utf-8 -*-

import pytest
import unittest
import brainpylib

from jax.lib import xla_bridge
import brainpy as bp
import brainpy.math as bm

if xla_bridge.get_backend().platform != 'gpu':
  pytest.skip("No gpu available.", allow_module_level=True)


class Test_csr_matvec(unittest.TestCase):
  def test_csr_matvec_homo(self):
    # matrix
    conn = bp.conn.FixedProb(0.1)
    indices, indptr = conn(100, 200).require('pre2post')
    indices = bm.as_jax(indices)
    indptr = bm.as_jax(indptr)
    # vector
    rng = bm.random.RandomState(123)
    vector = rng.random(200).value
    r1 = brainpylib.csr_matvec(1., indices, indptr, vector, shape=(100, 200))
    r2 = brainpylib.csr_matvec(bm.ones(indices.shape).value, indices, indptr, vector,
                               shape=(100, 200))
    self.assertTrue(bm.allclose(r1, r2))

  def test_csr_matvec_heter_1(self):
    conn = bp.conn.FixedProb(0.1)
    shape = (100, 200)

    # matrix
    indices, indptr = conn(*shape).require('pre2post')
    indices = bm.as_jax(indices)
    indptr = bm.as_jax(indptr)
    data = bm.random.random(indices.shape).value

    # vector
    rng = bm.random.RandomState(123)
    vector = rng.random(shape[1]).value
    r1 = brainpylib.csr_matvec(data, indices, indptr, vector, shape=shape)
    dense = brainpylib.csr_to_dense(data, indices, indptr, shape=shape)
    r2 = dense @ vector
    # print(r1)
    # print(r2)
    self.assertTrue(bm.allclose(r1, r2))

  def test_csr_matvec_heter_2(self):
    conn = bp.conn.FixedProb(0.1)
    shape = (300, 200)

    # matrix
    indices, indptr = conn(*shape).require('pre2post')
    indices = bm.as_jax(indices)
    indptr = bm.as_jax(indptr)
    data = bm.random.random(indices.shape).value

    # vector
    rng = bm.random.RandomState(123)
    vector = rng.random(shape[1]).value
    r1 = brainpylib.csr_matvec(data, indices, indptr, vector, shape=shape)
    dense = brainpylib.csr_to_dense(data, indices, indptr, shape=shape)
    r2 = dense @ vector
    # print(r1)
    # print(r2)
    self.assertTrue(bm.allclose(r1, r2))
