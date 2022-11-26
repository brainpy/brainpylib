import brainpy.math as bm
from jax import jit, vmap, numpy as jnp, block_until_ready

from time import time
from timeit import timeit
from brainpylib import event_info


def try_event_info():
  rng = bm.random.RandomState(123)
  events = rng.random((100,)).value < 0.1
  event_ids, event_num = jit(event_info)(events)
  print(event_ids)
  print(event_num)
  print(bm.allclose(bm.sum(events, keepdims=True), event_num))

  rng = bm.random.RandomState(123)
  events = rng.random((10, 1000)).value < 0.1
  event_ids, event_num = vmap(event_info)(events)
  print(event_ids)
  print(event_num)


def compare_speed_v1():
  bp_event_info = jit(event_info)
  jax_event_info = jit(lambda events: (jnp.argsort(events), jnp.sum(events)))

  rng = bm.random.RandomState(123)
  events = rng.random((10000000,)).value < 0.1
  event_ids1, event_num1 = bp_event_info(events)
  event_ids2, event_num2 = jax_event_info(events)
  assert jnp.allclose(event_num1, event_num2)

  t0 = time()
  for _ in range(100):
    a, b = jax_event_info(events)
    r = a.block_until_ready()
  print(f'JAX time {time() - t0} s')

  t0 = time()
  for _ in range(100):
    a, b = bp_event_info(events)
    r = a.block_until_ready()
  print(f'BP time {time() - t0} s')


def compare_speed_v2():
  bp_event_info = jit(vmap(event_info))
  jax_event_info = jit(vmap(lambda events: (jnp.argsort(events), jnp.sum(events))))

  rng = bm.random.RandomState(123)
  events = rng.random((100, 1000000)).value < 0.1
  event_ids1, event_num1 = bp_event_info(events)
  event_ids2, event_num2 = jax_event_info(events)
  assert jnp.allclose(event_num1, event_num2)

  t0 = time()
  for _ in range(100):
    a, b = jax_event_info(events)
    r = a.block_until_ready()
  print(f'JAX time {time() - t0} s')

  t0 = time()
  for _ in range(100):
    a, b = bp_event_info(events)
    r = a.block_until_ready()
  print(f'BP time {time() - t0} s')


if __name__ == '__main__':
  compare_speed_v2()
