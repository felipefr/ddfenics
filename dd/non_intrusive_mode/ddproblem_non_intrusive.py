#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 12:53:31 2026

@author: frocha
"""

import numpy as np
import scipy.sparse as sp

class DDProblemNonIntrusive:
    
    def __init__(self, Nz, W, B, WBT, L, Kbc, Lbc, metric, acc_op = None):
    
        self.metric = metric
        
        self.Nh = WBT.shape[0]
        self.Nz = Nz
        self.n_strain = int(Nz/2)
        self.Nqp = len(W) 
        
        self.z_mech = np.zeros((self.Nqp, self.Nz))# "strain/stress_mech" 
        self.z_db = np.zeros((self.Nqp, self.Nz)) # "strain/stress_db"

        self.B = B       
        self.WBT = WBT
        self.L = L
        self.W = W
        self.Kbc = Kbc
        self.Lbc = Lbc
        
        self.C = self.metric.C
        self.Cinv = self.metric.Cinv
    
        self.solver_lhs, self.solver_rhs, self.znew = self.create_problem()
        
        self.acc_op = acc_op

        # APE acceleration : https://doi.org/10.1016/j.mechmat.2025.105382
        if self.acc_op == "APE":
            # Pre-compute the operator (transposed because is a right multiplication )
            self.CCacc_right = 0.5 * np.block([
                [np.zeros_like(self.C), self.Cinv],
                [self.C,                np.zeros_like(self.C)]
            ]).T
            self._apply_acceleration = self.APE_update
        else:
            # Point the update handle to a "do nothing" function
            self._apply_acceleration = self._no_op
            
    # Typically, you should instanciate self.u, return the solver, and the symbolic update for z    
    def create_problem(self):
        self.u = np.zeros(self.Nh)
        self.eta = np.zeros(self.Nh)         
        
        C_big = sp.block_diag([self.C] * self.Nqp)
        
        K = self.WBT @ (C_big @ self.B) + self.Kbc
        LU = sp.linalg.splu(K)
        
        rhs = [lambda z_db: self.WBT@(z_db[:, :self.n_strain]@self.C).flatten() + self.Lbc, 
               lambda z_db: self.L - self.WBT@z_db[:, self.n_strain:].flatten() ]
        
        znew = [lambda u : (self.B@u).reshape((-1, self.n_strain)), 
             lambda eta: self.z_db[:, self.n_strain:] + (self.B@self.eta).reshape((-1, self.n_strain))@self.C ] 
        
        return LU,  rhs, znew
    
    def get_sol(self):
        return {"state_mech" : self.z_mech ,
                "state_db": self.z_db ,
                "u" : self.u }
    
    def solve(self):
        self.u = self.solver_lhs.solve(self.solver_rhs[0](self.z_db))
        self.eta = self.solver_lhs.solve(self.solver_rhs[1](self.z_db))
        
        
    def APE_update(self):
        self.z_mech += (self.z_mech - self.z_db) @ self.CCacc_right 
        
    def update_state_mech(self):
        self.z_mech[:, :self.n_strain] = self.znew[0](self.u)
        self.z_mech[:, self.n_strain:] = self.znew[1](self.eta)
        self._apply_acceleration()
            
    def update_state_db(self, state_db):
        self.z_db[:,:] =  state_db[:,:]
        
    def _no_op(self):
        """Does nothing, used when APE is disabled."""
        pass