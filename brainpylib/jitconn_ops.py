# -*- coding: utf-8 -*-

from ._src.jitconn_ops.event_matvec import (
  event_matvec_prob_conn_homo_weight as event_matvec_prob_conn_homo_weight,
  event_matvec_prob_conn_uniform_weight as event_matvec_prob_conn_uniform_weight,
  event_matvec_prob_conn_normal_weight as event_matvec_prob_conn_normal_weight,
)

from ._src.jitconn_ops.matvec import (
  matvec_prob_conn_homo_weight as matvec_prob_conn_homo_weight,
  matvec_prob_conn_uniform_weight as matvec_prob_conn_uniform_weight,
  matvec_prob_conn_normal_weight as matvec_prob_conn_normal_weight,
)


from ._src.jitconn_ops.matmat import (
  matmat_prob_conn_uniform_weight as matmat_prob_conn_uniform_weight,
  matmat_prob_conn_normal_weight as matmat_prob_conn_normal_weight,
)

