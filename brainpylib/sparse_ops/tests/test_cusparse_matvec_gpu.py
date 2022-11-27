# -*- coding: utf-8 -*-

import unittest

import brainpy as bp
import brainpy.math as bm
import jax
import pytest

import brainpylib

if jax.default_backend()!= 'gpu':
  pytest.skip("No gpu available.", allow_module_level=True)


class Test_cuSPARSE_CSRMatVec(unittest.TestCase):

  def test_csr_matvec_homo(self):
    conn = bp.conn.FixedProb(0.1)
    indices, indptr = conn(100, 200).require('pre2post')
    indices = bm.as_jax(indices)
    indptr = bm.as_jax(indptr)

    vector = bm.random.random(200).value
    r1 = brainpylib.cusparse_csr_matvec(1., indices, indptr, vector, shape=(100, 200))
    r2 = brainpylib.cusparse_csr_matvec(bm.ones(indices.shape).value, indices, indptr, vector,
                                        shape=(100, 200))
    self.assertTrue(bm.allclose(r1, r2))

  def test_csr_matvec_homo_transpose(self):
    conn = bp.conn.FixedProb(0.1)
    indices, indptr = conn(100, 200).require('pre2post')
    indices = bm.as_jax(indices)
    indptr = bm.as_jax(indptr)

    vector = bm.random.random(100).value
    r1 = brainpylib.cusparse_csr_matvec(1., indices, indptr, vector,
                                        shape=(100, 200), transpose=True)
    r2 = brainpylib.cusparse_csr_matvec(bm.ones(indices.shape).value, indices, indptr, vector,
                                        shape=(100, 200), transpose=True)
    self.assertTrue(bm.allclose(r1, r2))

  def test_csr_matvec_heter(self):
    conn = bp.conn.FixedProb(0.1)
    indices, indptr = conn(100, 200).require('pre2post')
    indices = bm.as_jax(indices)
    indptr = bm.as_jax(indptr)
    data = bm.random.random(indices.shape).value
    vector = bm.random.random(200).value

    r1 = brainpylib.cusparse_csr_matvec(data, indices, indptr, vector,
                                        shape=(100, 200))

    dense = brainpylib.csr_to_dense(data, indices, indptr, shape=(100, 200))
    r2 = dense @ vector
    self.assertTrue(bm.allclose(r1, r2))

  def test_csr_matvec_heter_transpose(self):
    conn = bp.conn.FixedProb(0.1)
    indices, indptr = conn(100, 200).require('pre2post')
    indices = bm.as_jax(indices)
    indptr = bm.as_jax(indptr)
    data = bm.random.random(indices.shape).value
    vector = bm.random.random(100).value

    r1 = brainpylib.cusparse_csr_matvec(data, indices, indptr, vector,
                                        shape=(100, 200),
                                        transpose=True)

    dense = brainpylib.csr_to_dense(data, indices, indptr, shape=(100, 200))
    r2 = dense.T @ vector
    self.assertTrue(bm.allclose(r1, r2))

    r3 = vector @ dense
    self.assertTrue(bm.allclose(r1, r3))
