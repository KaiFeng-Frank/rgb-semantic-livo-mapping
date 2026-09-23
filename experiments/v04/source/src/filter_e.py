#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
filter_e.py -- the frozen filter E, as a function of per-point teacher signal.

Dependency-free on purpose (numpy only): the training side imports it through
distil_ext (which drags in pointcept/spconv), the analysis side imports it directly
from an environment that has no CUDA extensions at all.  ONE definition, both sides.

The CRITERION is fixed by the design (keep only strata whose correction:regression
ratio exceeds 1).  The BOUNDARIES arrive as `spec`, re-derived on seq 08 by
tools/write_filterE.py.  This function decides nothing by itself.
"""
import numpy as np

F_INFRUSTUM, F_VISIBLE, F_DEPTHEDGE, F_RANGE_LT50 = 1, 2, 4, 8


def filter_E_mask(t9, m9, conf, flags, spec):
    keep = t9 >= 0
    if spec.get("require_visible", True):
        keep &= (flags & F_VISIBLE) > 0
    if spec.get("range_lt50", True):
        keep &= (flags & F_RANGE_LT50) > 0
    if spec.get("teachers_agree", True):
        keep &= (t9 == m9)
    thr = spec.get("conf_min", None)
    if thr is not None:
        keep &= np.asarray(conf, dtype=np.float32) >= float(thr)
    if spec.get("drop_depth_edge", True):
        keep &= (flags & F_DEPTHEDGE) == 0
    return keep
