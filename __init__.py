#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  6 11:41:16 2026

@author: frocha
"""

# deprecated
# CRITICAL = 50 , ERROR = 40 , WARNING = 30, INFO = 20, PROGRESS = 16, TRACE = 13, DBG = 10
# df.set_log_level(40)

from .dd.ddmaterial import DDMaterial
from .dd.ddmetric import DDMetric
from .dd.ddsolver import DDSolver
from .dd.ddsolver_nested import DDSolverNested

from .dd.ddfunction import DDFunction
from .dd.ddspace import DDSpace
from .dd.ddproblem_base import DDProblemBase
from .dd.ddproblem_infinitesimalstrain import DDProblemInfinitesimalStrain
from .dd.ddsearch import DDSearch
from .dd.ddstate import DDState
from .dd.ddsearch_nnls import DDSearchNNLS
from .dd.utils.estimation_metric import get_estimate_C_method, check_positiveness


from .utils.postprocessing import (db_mech_scatter_plot,
                                   export_xdmf_db_mech,
                                   callback_get_time_elapsed,
                                   callback_get_errors,
                                   get_errors_DD 
                                   )
#                                 comparison_with_reference_sol)
    
