# -*- coding: utf-8 -*-

import unittest

import brainpy.math as bm
from jax import jit, vmap

from brainpylib import event_info

bm.set_platform('cpu')


class TestEventInfo(unittest.TestCase):
  def test_simple_3d(self):
    rng = bm.random.RandomState(123)
    events = rng.random((10, 10, 100)).value < 0.1
    event_ids, event_num = vmap(vmap(event_info))(events)
    print(event_ids)
    print(event_num)
    self.assertTrue(bm.allclose(bm.sum(events, axis=-1), event_num))

  def test_simple_2d(self):
    rng = bm.random.RandomState(123)
    events = rng.random((10, 100)).value < 0.1
    event_ids, event_num = vmap(event_info)(events)
    print(event_ids)
    print(event_num)
    self.assertTrue(bm.allclose(bm.sum(events, axis=1), event_num))

  def test_simple_1d(self):
    rng = bm.random.RandomState(123)
    events = rng.random((100,)).value < 0.1
    event_ids, event_num = jit(event_info)(events)
    print(event_ids)
    print(event_num)
    self.assertTrue(bm.allclose(bm.sum(events, keepdims=True), event_num))

