#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 12 09:10:41 2026

@author: frocha

This is a condensed version of fetricks (https://github.com/felipefr/fetricks)
"""

import numpy as np
import ufl
from dolfinx import fem
from petsc4py import PETSc
from mpi4py import MPI
from dolfinx.fem.petsc import LinearProblem
from dolfinx import fem, la 


halfsqrt2 = 0.5*np.sqrt(2.)


Id_mandel = ufl.as_vector([1.0, 1.0, 0.0])

def symgrad_mandel(v): # it was shown somehow to have better performance than doing it explicity
    return ufl.as_vector([v[0].dx(0), v[1].dx(1), halfsqrt2*(v[0].dx(1) + v[1].dx(0))])

def tr_mandel(X):
    return X[0] + X[1]

def tensor2mandel(X):
    return ufl.as_vector([X[0,0], X[1,1], halfsqrt2*(X[0,1] + X[1,0])])

def mandel2tensor_np(X):
    return np.array([[X[0], halfsqrt2*X[2]],
                     [halfsqrt2*X[2], X[1]]])

def tensor2mandel_np(X):
    return np.array([X[0,0], X[1,1], halfsqrt2*(X[0,1] + X[1,0])])


def mandel2tensor(X):
    return ufl.as_tensor([[X[0], halfsqrt2*X[2]],
                        [halfsqrt2*X[2], X[1]]])


def L2norm_given_form(form, comm):
    val = fem.assemble_scalar(form)
    return np.sqrt(comm.allreduce(val, op=MPI.SUM))


def get_indexes_cells(mesh):
    map_c = mesh.topology.index_map(mesh.topology.dim)
    num_cells = map_c.size_local + map_c.num_ghosts
    return np.arange(0, num_cells, dtype=np.int32)

def interpolate_quadrature(ufl_expr, u):
    V = u.function_space
    mesh = V.mesh
    cells = get_indexes_cells(mesh)
    quadrature_points = V.element.interpolation_points
    
    if len(ufl_expr.ufl_shape) == 0:
        expr_expr = fem.Expression(ufl_expr, quadrature_points)
        expr_eval = expr_expr.eval(mesh, cells)
        u.x.array[:] = expr_eval.flatten()[:]
    else:
        n = ufl_expr.ufl_shape[0]
        for i in range(n):
            expr_expr = fem.Expression(ufl_expr[i], quadrature_points)
            expr_eval = expr_expr.eval(mesh, cells)
            u.x.array[i::n] = expr_eval.flatten()[:]


# def L2norm(u, dx, comm):
#     return L2norm_given_form(fem.form(ufl.inner(u, u) * dx), comm)


class CustomLinearSolver:
    def __init__(self, lhs, rhs, sol, bcs, solver = None):
        self.sol = sol
        self.bcs = bcs
        domain = self.sol.function_space.mesh
        
        if(solver): #if solver is given
            self.solver = solver
            self.lhs = lhs
        else:
            self.solver = PETSc.KSP().create(domain.comm)
            self.solver.setType(PETSc.KSP.Type.PREONLY)
            self.solver.getPC().setType(PETSc.PC.Type.LU)
            self.solver.getPC().setFactorSolverType("mumps")
            self.lhs = fem.form(lhs)
            self.assembly_lhs()
            
        self.rhs = fem.form(rhs)
        self.b = fem.petsc.create_vector(self.sol.function_space)
    
    def assembly_lhs(self):
        self.A = fem.petsc.assemble_matrix(self.lhs, bcs=self.bcs)
        self.A.assemble()
        self.solver.setOperators(self.A)

    def assembly_rhs(self):
        with self.b.localForm() as b:
            b.set(0.0)
        
        fem.petsc.assemble_vector(self.b, self.rhs)
        fem.petsc.apply_lifting(self.b, [self.lhs], [self.bcs])
        self.b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES,mode=PETSc.ScatterMode.REVERSE)    
        fem.petsc.set_bc(self.b,self.bcs)

    def solve(self):  
       self.assembly_rhs()
       # self.solver.solve(self.b,self.sol.vector)
       self.solver.solve(self.b,self.sol.array)
       self.sol.x.scatter_forward()

class BlockSolver:
    def __init__(self, lhs, rhs, sol, bcs):
        self.n_subproblems = len(rhs)        
        
        if(isinstance(lhs, list)):    
            self.list_solver = [CustomLinearSolver(lhs[i], rhs[i], sol[i], bcs[i]) 
                                for i in range(self.n_subproblems)]
        
        else:
            self.list_solver = [CustomLinearSolver(lhs, rhs[0], sol[0], bcs[0])] 
            self.list_solver += [CustomLinearSolver(self.list_solver[0].lhs, rhs[i], sol[i], 
                                 bcs[i], solver = self.list_solver[0].solver) for i in range(1, self.n_subproblems)]                    

            
         
    def assembly_rhs(self):
       for i in range(self.n_subproblems):
           self.list_solver[i].assembly_rhs()

    def solve(self):  
       self.assembly_rhs()
       for i in range(self.n_subproblems):
           s = self.list_solver[i]
           s.solver.solve(s.b, s.sol.x.petsc_vec)
           s.sol.x.scatter_forward()

