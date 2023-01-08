# -*- coding: utf-8 -*-

__version__ = "0.1.5"


# IMPORTANT, must import first
from ._src import check, register_custom_calls
del check, register_custom_calls

# operator customization
from . import op_register
from .op_register import *

# event-driven operators
from . import event_ops
from .event_ops import *

# sparse operators
from . import sparse_ops
from .sparse_ops import *

# jitconn operators
from . import jitconn_ops
from .jitconn_ops import *

# other operators
from . import compat
from .compat import *
