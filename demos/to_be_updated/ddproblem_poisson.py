#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  11 00:20:39 2023

@author: ffiguere

This file is part of ddfenics, a FEniCs-based (Model-Free) Data-driven 
Computational Mechanics implementation.

Copyright (c) 2022-2023, Felipe Rocha.
See file LICENSE.txt for license information.
Please report all bugs and problems to <felipe.figueredo-rocha@ec-nantes.fr>, or
<f.rocha.felipe@gmail.com>

"""

import dolfinx, ufl
from dolfinx import fem,mesh,plot
import dolfinx.fem.petsc
from petsc4py import PETSc
from mpi4py import MPI
from petsc4py.PETSc import ScalarType  # type: ignore
import ddfenicsx as dd

# Block Solver [K 0; 0 K] u = [F_1 ... F_N]
class BlockSolver:

    def __init__(self, a, b1, b2, u1, u2, bcs1, bcs2):
        
        domain = u1.function_space.mesh
        
        # generate associated matrix
        bilinear_form_dd = fem.form(a)
        A_dd = fem.petsc.assemble_matrix(bilinear_form_dd,bcs=bcs1)
        A_dd.assemble()

        solver_dd = PETSc.KSP().create(domain.comm)
        solver_dd.setOperators(A_dd)
        solver_dd.setType(PETSc.KSP.Type.PREONLY)
        solver_dd.getPC().setType(PETSc.PC.Type.LU)
        
        
        L1 = fem.form(b1)
        L1_ = fem.petsc.create_vector(L1)
        
        L2 = fem.form(b2)
        L2_ = fem.petsc.create_vector(L2)
        


#         self.F = [ df.PETScVector() for i in range(self.n_subproblems) ] 
#         self.A = df.PETScMatrix()
        
#         # supposing lhs and bcs are equal for all problems
#         df.assemble(self.subproblems[0].a_ufl , tensor = self.A)
#         [bc.apply(self.A) for bc in self.subproblems[0].bcs()] 
        
#         self.solver = df.PETScLUSolver(self.A)

#     def assembly_rhs(self):
#         for i in range(self.n_subproblems): 
#             df.assemble(self.subproblems[i].L_ufl, tensor = self.F[i])    
#             [bc.apply(self.F[i]) for bc in self.subproblems[i].bcs()]
            
#     def solve(self):   
#         self.assembly_rhs()
#         for i in range(self.n_subproblems): 
#             self.solver.solve(self.subproblems[i].u_ufl.vector(), self.F[i])

#     def solve_subproblem(self, i):   
#         df.assemble(self.subproblems[i].L_ufl, tensor = self.F[i])    
#         [bc.apply(self.F[i]) for bc in self.subproblems[i].bcs()]
#         self.solver.solve(self.subproblems[i].u_ufl.vector(), self.F[i])

class DDProblemPoisson(dd.DDProblemBase):
    def __init__(self, spaces, grad, L, bcs, metric,  
                 form_compiler_parameters = {}, bcsPF = []):
    
        super().__init__(spaces, grad, L, bcs, metric,  
                     form_compiler_parameters, bcsPF)
        
        
    def create_problem(self):
        
        self.u = fem.Function(self.Uh)
        self.eta = fem.Function(self.Uh)

        bcs_eta = [fem.bcs.DirichletBC(b) for b in self.bcs] # creates a copy
        # [b.homogenize() for b in bcs_eta] # in the case tested its already homogeneous
        
        uh = ufl.TrialFunction(self.Uh)
        vh = ufl.TestFunction(self.Uh)
        
        a = ufl.inner(self.C*self.grad(uh), self.grad(vh))*self.dx
        
        f_comp = ufl.inner(self.C*self.z_db[0] , self.grad(vh))*self.dx     
        f_bal = self.L(vh,self.dx)
        
        
        # self.problem_comp = dolfinx.fem.petsc.LinearProblem(a, f_comp, self.bcs, self.u)
        # self.problem_bal = dolfinx.fem.petsc.LinearProblem(a, f_bal, bcs_eta, self.eta)
        
        # idea of the bug
        # https://fenicsproject.discourse.group/t/how-to-apply-many-forces-when-solving-elasticity-problem/13173/9
        
        blocksolver = BlockSolver(a, f_comp, f_bal, self.u, self.eta, self.bcs, bcs_eta) 
#        blocksolver = None
        z = [self.grad(self.u), self.z_db[1] + self.C*self.grad(self.eta)] # z_mech symbolic

        return blocksolver, z
    

    def get_sol(self):
        return {"state_mech" : self.z_mech ,
                "state_db": self.z_db ,
                "u" : self.u }
    
