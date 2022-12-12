# -*- coding: utf-8 -*-

import unittest

import brainpy.math as bm
import jax

from brainpylib import jitconn_ops


class Test_matvec_prob_conn(unittest.TestCase):
  def __init__(self, *args, platform='cpu', **kwargs):
    super(Test_matvec_prob_conn, self).__init__(*args, **kwargs)
    bm.set_platform(platform)
    print()

    self.shapes = [(100, 200),
                   (200, 200),
                   (10, 1000),
                   (2, 1000),
                   (1000, 10),
                   (1000, 2)]

  def _test_homo(self, shape, transpose, prob, homo_data, seed=None):
    print(f'{self._test_homo.__name__}: '
          f'shape = {shape}, '
          f'transpose = {transpose}, '
          f'prob={prob}, '
          f'homo_data = {homo_data}')

    rng = bm.random.RandomState()
    vector = rng.random(shape[0] if transpose else shape[1]).value

    r1 = jitconn_ops.matvec_prob_conn_homo_weight(vector,
                                                  homo_data,
                                                  conn_prob=prob,
                                                  shape=shape,
                                                  seed=seed,
                                                  transpose=transpose)

    r2 = jitconn_ops.matvec_prob_conn_homo_weight(vector,
                                                  homo_data,
                                                  conn_prob=prob,
                                                  shape=shape,
                                                  seed=seed,
                                                  transpose=transpose)
    self.assertTrue(bm.allclose(r1, r2))

    r2 = jitconn_ops.matvec_prob_conn_homo_weight(vector,
                                                  homo_data,
                                                  conn_prob=prob,
                                                  shape=(shape[1], shape[0]),
                                                  seed=seed,
                                                  transpose=not transpose)
    self.assertTrue(bm.allclose(r1, r2))

    # indices, indptr = bp.conn.FixedProb(prob)(*shape).require('pre2post')
    # indices = bm.as_jax(indices)
    # indptr = bm.as_jax(indptr)
    # r3 = sparse_ops.cusparse_csr_matvec(homo_data, indices, indptr, vector,
    #                                     shape=shape, transpose=transpose)
    # print('Homo difference: ', bm.abs(r1 - r3).sum() / r1.size)

    bm.clear_buffer_memory()

  def _test_homo_vmap(self, shape, transpose, prob, seed=None):
    print(f'{self._test_homo_vmap.__name__}: '
          f'shape = {shape}, '
          f'transpose = {transpose}, '
          f'prob={prob}')

    rng = bm.random.RandomState()
    events = rng.random((10, shape[0] if transpose else shape[1])).value
    weights = rng.random(10).value

    f1 = jax.vmap(lambda event, data: jitconn_ops.matvec_prob_conn_homo_weight(
      event, data, conn_prob=prob, shape=shape, seed=seed, transpose=transpose))
    r1 = f1(events, weights)
    r2 = f1(events, weights)
    self.assertTrue(bm.allclose(r1, r2))

    bm.clear_buffer_memory()

  def _test_homo_grad(self, shape, transpose, prob, seed=None):
    print(f'{self._test_homo_grad.__name__}: '
          f'shape = {shape}, '
          f'transpose = {transpose}, '
          f'prob={prob}')

    rng = bm.random.RandomState()
    events = rng.random(shape[0] if transpose else shape[1]).value < 0.5
    events = events.astype(float)

    f1 = jax.grad(
      lambda event, data: jitconn_ops.matvec_prob_conn_homo_weight(
        event, data, conn_prob=prob, shape=shape, seed=seed, transpose=transpose).sum(),
      argnums=0
    )
    r1 = f1(events, 1.)

    f2 = jax.grad(
      lambda event, data: jitconn_ops.matvec_prob_conn_homo_weight(
        event, data, conn_prob=prob, shape=shape, seed=seed, transpose=transpose).sum(),
      argnums=1
    )
    r2 = f2(events, 1.)

    f3 = jax.grad(
      lambda event, data: jitconn_ops.matvec_prob_conn_homo_weight(
        event, data, conn_prob=prob, shape=shape, seed=seed, transpose=transpose).sum(),
      argnums=(0, 1)
    )
    r3 = f3(events, 1.)

    self.assertTrue(bm.allclose(r1, r3[0]))
    self.assertTrue(bm.allclose(r2, r3[1]))

    # print(r3)
    bm.clear_buffer_memory()

  def _test_uniform(self, shape, transpose, prob, w_low, w_high, seed=None):
    print(f'{self._test_uniform.__name__}: '
          f'shape = {shape}, '
          f'transpose = {transpose}, '
          f'prob={prob}, '
          f'w_low = {w_low}, '
          f'w_high = {w_high}')

    rng = bm.random.RandomState()
    events = rng.random(shape[0] if transpose else shape[1]).value

    r1 = jitconn_ops.matvec_prob_conn_uniform_weight(events,
                                                     w_low=w_low,
                                                     w_high=w_high,
                                                     conn_prob=prob,
                                                     shape=shape,
                                                     seed=seed,
                                                     transpose=transpose)

    r2 = jitconn_ops.matvec_prob_conn_uniform_weight(events,
                                                     w_low=w_low,
                                                     w_high=w_high,
                                                     conn_prob=prob,
                                                     shape=shape,
                                                     seed=seed,
                                                     transpose=transpose)
    self.assertTrue(bm.allclose(r1, r2))

    r2 = jitconn_ops.matvec_prob_conn_uniform_weight(events,
                                                     w_low=w_low,
                                                     w_high=w_high,
                                                     conn_prob=prob,
                                                     shape=(shape[1], shape[0]),
                                                     seed=seed,
                                                     transpose=not transpose)
    self.assertTrue(bm.allclose(r1, r2))

    # indices, indptr = bp.conn.FixedProb(prob)(*shape).require('pre2post')
    # indices = bm.as_jax(indices)
    # indptr = bm.as_jax(indptr)
    # heter_data = rng.uniform(w_low, w_high, size=indices.size).value
    # r3 = sparse_ops.cusparse_csr_matvec(heter_data, indices, indptr, events,
    #                                     shape=shape,
    #                                     transpose=transpose)
    # print('Uniform difference: ', bm.abs(r1 - r3).sum() / r1.size)

    bm.clear_buffer_memory()

  def _test_uniform_vmap(self, shape, transpose, prob, seed=None):
    print(f'{self._test_uniform_vmap.__name__}: '
          f'shape = {shape}, '
          f'transpose = {transpose}, '
          f'prob={prob}')

    rng = bm.random.RandomState()
    events = rng.random((10, shape[0] if transpose else shape[1])).value

    f1 = jax.vmap(lambda e: jitconn_ops.matvec_prob_conn_uniform_weight(e,
                                                                        w_low=0.,
                                                                        w_high=1.,
                                                                        conn_prob=prob,
                                                                        shape=shape,
                                                                        seed=seed,
                                                                        transpose=transpose))

    r1 = f1(events)
    r2 = f1(events)
    self.assertTrue(bm.allclose(r1, r2))

    bm.clear_buffer_memory()

  def _test_uniform_grad(self, shape, transpose, prob, seed=None):
    print(f'{self._test_uniform_grad.__name__}: '
          f'shape = {shape}, '
          f'transpose = {transpose}, '
          f'prob={prob}')

    rng = bm.random.RandomState()
    events = rng.random(shape[0] if transpose else shape[1]).value

    f1 = jax.grad(
      lambda e: jitconn_ops.matvec_prob_conn_uniform_weight(
        e,
        w_low=0.,
        w_high=1.,
        conn_prob=prob,
        shape=shape,
        seed=seed,
        transpose=transpose).sum()
    )

    r1 = f1(events)
    # print(r1)

    bm.clear_buffer_memory()

  def _test_normal(self, shape, transpose, prob, w_mu, w_sigma, seed=None):
    print(f'{self._test_normal.__name__}: '
          f'shape = {shape}, '
          f'transpose = {transpose}, '
          f'prob={prob}, '
          f'w_mu = {w_mu}, '
          f'w_sigma = {w_sigma}')

    rng = bm.random.RandomState()
    events = rng.random(shape[0] if transpose else shape[1]).value

    r1 = jitconn_ops.matvec_prob_conn_normal_weight(events,
                                                    w_mu=w_mu,
                                                    w_sigma=w_sigma,
                                                    conn_prob=prob,
                                                    shape=shape,
                                                    seed=seed,
                                                    transpose=transpose)

    r2 = jitconn_ops.matvec_prob_conn_normal_weight(events,
                                                    w_mu=w_mu,
                                                    w_sigma=w_sigma,
                                                    conn_prob=prob,
                                                    shape=shape,
                                                    seed=seed,
                                                    transpose=transpose)
    self.assertTrue(bm.allclose(r1, r2))

    r2 = jitconn_ops.matvec_prob_conn_normal_weight(events,
                                                    w_mu=w_mu,
                                                    w_sigma=w_sigma,
                                                    conn_prob=prob,
                                                    shape=(shape[1], shape[0]),
                                                    seed=seed,
                                                    transpose=not transpose)
    self.assertTrue(bm.allclose(r1, r2))

    # indices, indptr = bp.conn.FixedProb(prob)(*shape).require('pre2post')
    # indices = bm.as_jax(indices)
    # indptr = bm.as_jax(indptr)
    # heter_data = rng.normal(w_mu, w_sigma, size=indices.size).value
    # r3 = sparse_ops.cusparse_csr_matvec(heter_data, indices, indptr, events,
    #                                     shape=shape, transpose=transpose)
    # print('Normal difference: ', bm.abs(r1 - r3).sum() / r1.size)

    bm.clear_buffer_memory()

  def _test_normal_vmap(self, shape, transpose, prob, seed=None):
    print(f'{self._test_normal_vmap.__name__}: '
          f'shape = {shape}, '
          f'transpose = {transpose}, '
          f'prob={prob}')

    rng = bm.random.RandomState()
    events = rng.random((10, shape[0] if transpose else shape[1])).value

    f1 = jax.vmap(lambda e: jitconn_ops.matvec_prob_conn_normal_weight(e,
                                                                       w_mu=0.,
                                                                       w_sigma=1.,
                                                                       conn_prob=prob,
                                                                       shape=shape,
                                                                       seed=seed,
                                                                       transpose=transpose))
    r1 = f1(events)
    r2 = f1(events)
    self.assertTrue(bm.allclose(r1, r2))

    bm.clear_buffer_memory()

  def _test_normal_grad(self, shape, transpose, prob, seed=None):
    print(f'{self._test_normal_grad.__name__}: '
          f'shape = {shape}, '
          f'transpose = {transpose}, '
          f'prob={prob}')

    rng = bm.random.RandomState()
    events = rng.random(shape[0] if transpose else shape[1]).value < 0.1
    events = events.astype(float)

    f1 = jax.grad(
      lambda e: jitconn_ops.matvec_prob_conn_normal_weight(
        e,
        w_mu=0.,
        w_sigma=1.,
        conn_prob=prob,
        shape=shape,
        seed=seed,
        transpose=transpose).sum()
    )
    r1 = f1(events)
    # print(r1)

    bm.clear_buffer_memory()

  def test_homo(self):
    for transpose in [True, False]:
      for shape in self.shapes:
        for prob in [0.01, 0.02, 0.05, 0.1, 0.2, 0.4]:
          for homo_data in [-1., 1.]:
            self._test_homo(shape, transpose, prob, homo_data, seed=4366363)
    bm.enable_x64()
    for transpose in [True, False]:
      for shape in self.shapes:
        for prob in [0.01, 0.02, 0.05, 0.1, 0.2, 0.4]:
          for homo_data in [-1., 1.]:
            self._test_homo(shape, transpose, prob, homo_data,
                            seed=4366363)
    bm.disable_x64()

  def test_homo_vmap(self):
    for transpose in [True, False]:
      for shape in self.shapes:
        for prob in [0.01, 0.05, 0.1, 0.5]:
          self._test_homo_vmap(shape, transpose, prob, seed=4366363)

    bm.enable_x64()
    for transpose in [True, False]:
      for shape in self.shapes:
        for prob in [0.01, 0.05, 0.1, 0.5]:
          self._test_homo_vmap(shape, transpose, prob, seed=4366363)
    bm.disable_x64()

  def test_homo_grad(self):
    for transpose in [True, False]:
      for shape in self.shapes:
        for prob in [0.01, 0.05, 0.1, 0.5]:
          self._test_homo_grad(shape, transpose, prob, seed=4366363)
    bm.enable_x64()
    for transpose in [True, False]:
      for shape in self.shapes:
        for prob in [0.01, 0.05, 0.1, 0.5]:
          self._test_homo_grad(shape, transpose, prob, seed=4366363)
    bm.disable_x64()

  def test_uniform(self):
    for transpose in [True, False]:
      for shape in self.shapes:
        for prob in [0.01, 0.02, 0.05, 0.1, 0.2, 0.4]:
          for w_low, w_high in [(-1., 0.), (0., 1.), (-1., 1.)]:
            self._test_uniform(shape, transpose, prob,
                               w_low=w_low,
                               w_high=w_high,
                               seed=4366363)
    bm.enable_x64()
    for transpose in [True, False]:
      for shape in self.shapes:
        for prob in [0.01, 0.02, 0.05, 0.1, 0.2, 0.4]:
          for w_low, w_high in [(-1., 0.), (0., 1.), (-1., 1.)]:
            self._test_uniform(shape, transpose, prob,
                               w_low=w_low,
                               w_high=w_high,
                               seed=4366363)
    bm.disable_x64()

  def test_uniform_vmap(self):
    for transpose in [True, False]:
      for shape in self.shapes:
        for prob in [0.01, 0.05, 0.1, 0.4]:
          self._test_uniform_vmap(shape, transpose, prob, seed=4366363)

    bm.enable_x64()
    for transpose in [True, False]:
      for shape in self.shapes:
        for prob in [0.01, 0.05, 0.1, 0.4]:
          self._test_uniform_vmap(shape, transpose, prob, seed=4366363)
    bm.disable_x64()

  def test_uniform_grad(self):
    for transpose in [True, False]:
      for shape in self.shapes:
        for prob in [0.01, 0.05, 0.1, 0.4]:
          self._test_uniform_grad(shape, transpose, prob, seed=4366363)

    bm.enable_x64()
    for transpose in [True, False]:
      for shape in self.shapes:
        for prob in [0.01, 0.05, 0.1, 0.4]:
          self._test_uniform_grad(shape, transpose, prob, seed=4366363)
    bm.disable_x64()

  def test_normal(self):
    for transpose in [True, False]:
      for shape in self.shapes:
        for prob in [0.01, 0.02, 0.05, 0.1, 0.2, 0.4]:
          for w_low, w_high in [(-1., 1.), (0., 0.1), (0., 0.5)]:
            self._test_normal(shape, transpose, prob,
                              w_mu=w_low,
                              w_sigma=w_high,
                              seed=4366363)
    bm.enable_x64()
    for transpose in [True, False]:
      for shape in self.shapes:
        for prob in [0.01, 0.02, 0.05, 0.1, 0.2, 0.4]:
          for w_low, w_high in [(-1., 1.), (0., 0.1), (0., 0.5)]:
            self._test_normal(shape, transpose, prob,
                              w_mu=w_low,
                              w_sigma=w_high,
                              seed=4366363)
    bm.disable_x64()

  def test_normal_vmap(self):
    for transpose in [True, False]:
      for shape in self.shapes:
        for prob in [0.01, 0.05, 0.1, 0.4]:
          self._test_normal_vmap(shape,
                                 transpose,
                                 prob,
                                 seed=4366363)

    bm.enable_x64()
    for transpose in [True, False]:
      for shape in self.shapes:
        for prob in [0.01, 0.05, 0.1, 0.4]:
          self._test_normal_vmap(shape,
                                 transpose,
                                 prob,
                                 seed=4366363)
    bm.disable_x64()

  def test_normal_grad(self):
    for transpose in [True, False]:
      for shape in self.shapes:
        for prob in [0.01, 0.05, 0.1, 0.4]:
          self._test_normal_grad(shape,
                                 transpose,
                                 prob,
                                 seed=4366363)

    bm.enable_x64()
    for transpose in [True, False]:
      for shape in self.shapes:
        for prob in [0.01, 0.05, 0.1, 0.4]:
          self._test_normal_grad(shape,
                                 transpose,
                                 prob,
                                 seed=4366363)
    bm.disable_x64()
