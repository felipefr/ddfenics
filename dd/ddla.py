#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  8 23:27:28 2022

@author: felipe
"""

import numpy as np
import dolfin as df

# create different matrices for comp and bal problems
# Aide-memoire options
# ['krylov_solver',
#  'linear_solver',
#  'lu_solver',
#  'preconditioner',
#  'print_matrix',
#  'print_rhs',
#  'symmetric']

class DDla:

    def __init__(self, problem):
        self.b_comp = df.PETScVector()
        self.b_bal = df.PETScVector()        
        self.A = df.PETScMatrix()
        self.problem = problem
        
        df.assemble(self.problem.problem_comp.a_ufl, tensor = self.A)
        [bc.apply(self.A) for bc in self.problem.bcs] # supposing problem.bcs is equal to problem.bcs_eta (at least for A)
        
        self.solver = df.PETScLUSolver(self.A, 'superlu')

    def assembly(self):
        df.assemble(self.problem.problem_comp.L_ufl, tensor = self.b_comp)
        [bc.apply(self.b_comp) for bc in self.problem.bcs]
        
        df.assemble(self.problem.problem_bal.L_ufl, tensor = self.b_bal)
        [bc.apply(self.b_bal) for bc in self.problem.bcs_eta]
        [bc.apply(self.b_bal) for bc in self.problem.bcsPF]
    
    # To now, the order of solving is important, because b_comp depends on eta.
    # It might be rearranged in a way this depedence is drop
    # keep the names f_bal, b_bal, b_comp, f_comp, reserved to match with original formulation
    def solve(self):   
    
        self.assembly()
        self.solver.solve( self.problem.eta.vector(), self.b_bal)
        self.assembly()
        self.solver.solve( self.problem.u.vector(), self.b_comp)
        
        # self.assembly()
        self.problem.update_state_mech()
        
        
class DDla_mixed:

    def __init__(self, problem):
        self.b = df.PETScVector()
        self.A = df.PETScMatrix()
        self.problem = problem
        
        df.assemble(self.problem.problem.a_ufl, tensor = self.A)
        [bc.apply(self.A) for bc in self.problem.bcs]
        
        self.solver = df.PETScLUSolver(self.A, 'superlu')

    def assembly(self):
        df.assemble(self.problem.problem.L_ufl, tensor = self.b)
        [bc.apply(self.b) for bc in self.problem.bcs]
        
    def solve(self): 
        
        self.assembly()
        
        self.solver.solve( self.problem.u.vector(), self.b)
        
        self.problem.update_state_mech()