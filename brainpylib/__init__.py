# -*- coding: utf-8 -*-

__version__ = "0.1.2"

# IMPORTANT, must import first
from . import register_custom_calls

# operator customization
from .custom_op import *

# event-driven operators
from .event_sparse_matmul import *

# sparse operators
from .sparse_matmul import *

# other operators
from .event_info_collection import *

from .compat.event_sum import *
from .compat.event_prod import *
from .compat.atomic_sum import *
from .compat.atomic_prod import *
