# -*- coding: utf-8 -*-

__version__ = "0.1.4"


from . import check
del check


# IMPORTANT, must import first
from . import register_custom_calls

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


__all__ = (
  ['event_ops', 'sparse_ops', 'jitconn_ops', 'op_register']
  + event_ops.__all__
  + sparse_ops.__all__
  + op_register.__all__
  + compat.__all__
)



