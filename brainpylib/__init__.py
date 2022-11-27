# -*- coding: utf-8 -*-

__version__ = "0.1.2"

# IMPORTANT, must import first
from . import register_custom_calls

# operator customization
from .op_custom import *

# event-driven operators
from brainpylib.event_ops.event_sparse_matmul import *

# sparse operators
from .sparse_ops import *

# other operators
from brainpylib.event_ops.event_info_collection import *
from .compat import *
