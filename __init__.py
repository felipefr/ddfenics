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
# from .dd.ddbilinear import DDBilinear
from .dd.ddproblem_base import DDProblemBase
# from .dd.ddproblem_generic import DDProblemGeneric as DDProblem
from .dd.ddproblem_infinitesimalstrain import DDProblemInfinitesimalStrain
# from .dd.ddproblem_poisson import DDProblemPoisson # negative flux
from .dd.ddsearch import DDSearch
from .dd.ddstate import DDState
from .dd.utils.estimation_metric import get_estimate_C_method, check_positiveness

# # research-oriented development
# from .ddd.ddproblem_finitestrain import DDProblemFiniteStrain
# from .ddd.ddmaterial_rve import DDMaterial_RVE
# from .ddd.ddmaterial_active import DDMaterialActive
# from .ddd.ddsolver_nested import DDSolverNested
# from .ddd.dd_al_update import *
# # from .ddd.ddproblem_infinitesimalstrain_omega import DDProblemInfinitesimalStrainOmega
# # from .ddd.ddsolver_dropout import DDSolverDropout
# # from .ddd.ddsearch_isotropy import DDSearchIsotropy
# from .ddd.ddsearch_nnls import DDSearchNNLS
# from .ddd.ddproblem_poisson_Hdiv import DDProblemPoissonHdiv
# from .ddd.ddproblem_poisson_H1 import DDProblemPoissonH1
# from .ddd.ddproblem_poisson_Hdiv_negflux import DDProblemPoissonHdivNegFlux


# from .utils.postprocessing import (db_mech_scatter_plot,
#                                    comparison_with_reference_sol)

# from .ddd.generalized_metric import *

