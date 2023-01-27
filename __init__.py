import os
import dolfin as df

# CRITICAL = 50 , ERROR = 40 , WARNING = 30, INFO = 20, PROGRESS = 16, TRACE = 13, DBG = 10
df.set_log_level(40)

from .dd.ddmetric import DDMetric
from .dd.ddsolver import DDSolver
from .dd.ddmaterial import DDMaterial
from .dd.ddfunction import DDFunction
from .dd.ddspace import DDSpace
from .dd.ddbilinear import DDBilinear
from .dd.ddproblem_base import DDProblemBase
from .dd.ddproblem_generic import DDProblemGeneric as DDProblem
from .dd.ddproblem_infinitesimalstrain import DDProblemInfinitesimalStrain
from .dd.ddproblem_poisson import DDProblemPoisson
