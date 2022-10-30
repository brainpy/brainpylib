# -*- coding: utf-8 -*-


from . import numba_based_registration, taichi_based_registration, triton_based_registration

from .numba_based_registration import *
from .taichi_based_registration import *
from .triton_based_registration import *

__all__ = (numba_based_registration.__all__ +
           taichi_based_registration.__all__ +
           triton_based_registration.__all__)
