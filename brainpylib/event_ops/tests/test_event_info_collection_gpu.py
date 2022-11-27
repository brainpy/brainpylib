# -*- coding: utf-8 -*-

import unittest

import brainpy.math as bm
import jax
import pytest
from jax import jit, vmap

from brainpylib import event_info

if jax.default_backend() != 'gpu':
  pytest.skip("No gpu available.", allow_module_level=True)


class TestEventInfo(unittest.TestCase):
  def test_simple_3d(self):
    rng = bm.random.RandomState(123)
    events = rng.random((10, 10, 1000)).value < 0.1
    event_ids, event_num = vmap(vmap(event_info))(events)
    print(event_ids)
    print(event_num)
    self.assertTrue(bm.allclose(bm.sum(events, axis=-1), event_num))

  def test_simple_2d(self):
    rng = bm.random.RandomState(123)
    events = rng.random((10, 1000)).value < 0.1
    event_ids, event_num = vmap(event_info)(events)
    print(event_ids)
    print(event_num)
    self.assertTrue(bm.allclose(bm.sum(events, axis=1), event_num))

  def test_simple_1d(self):
    rng = bm.random.RandomState(123)
    events = rng.random((1000,)).value < 0.1
    event_ids, event_num = jit(event_info)(events)
    print(event_ids)
    print(event_num)
    self.assertTrue(bm.allclose(bm.sum(events, keepdims=True), event_num))
