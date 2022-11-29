# -*- coding: utf-8 -*-


__all__ = [
  'EventLowRankMat'
]


class EventLowRankMat(object):
  """An operator computes: :math:`event @ L @ R`, where :math:`L`
  and :math:`R` are decomposed matrices for the original low-rank matrix.
  """

  def __call__(self, events, L, R):
    return events @ L @ R

