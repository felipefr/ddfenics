#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  6 11:41:16 2026

@author: frocha
"""

# deprecated
# CRITICAL = 50 , ERROR = 40 , WARNING = 30, INFO = 20, PROGRESS = 16, TRACE = 13, DBG = 10
# df.set_log_level(40)

from .dd import *
__version__ = "0.1.0"

from . import utils

from .dd import non_intrusive_mode

# from .utils.estimation_metric import get_estimate_C_method, check_positiveness
