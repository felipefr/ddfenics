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

from .dd.ddfunction import DDFunction
from .dd.ddspace import DDSpace
from .dd.ddproblem_base import DDProblemBase
from .dd.ddproblem_infinitesimalstrain import DDProblemInfinitesimalStrain
from .dd.ddsearch import DDSearch
from .dd.ddstate import DDState
from .dd.ddsearch_nnls import DDSearchNNLS
from .dd.utils.estimation_metric import get_estimate_C_method, check_positiveness

from .utils.postprocessing_multiscale import (callback_get_errors,
                                              callback_get_DBsize,
                                              callback_get_time_elapsed,
                                              get_errors_DD 
                                              # get_sol_ref, 
                                              # get_sol_ref_with_mesh, 
                                              # get_state_database_from_xdmf,
                                              # export_database_from_simul,                                              
                                              # convergence_plot,
                                              # callback_generate_vtk,
                                              # callback_evolution_state,
                                              # callback_evolution_db,
                                              # callback_get_update_counter_field,
                                              # write_vtk_from_array
                                              )
                                             


from .utils.postprocessing import (db_mech_scatter_plot,
                                   export_xdmf_db_mech)
#                                 comparison_with_reference_sol)
    
