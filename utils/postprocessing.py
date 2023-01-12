#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 22 12:35:49 2022

@author: felipefr
"""


import sys, os
import dolfin as df 
import matplotlib.pyplot as plt
import numpy as np
import fetricks.fenics.postprocessing.wrapper_io as iofe
from fetricks.fenics.la.wrapper_solvers import local_project_given_sol

from ddfenics.dd.ddfunction import DDFunction
from ddfenics.dd.ddspace import DDSpace

def comparison_with_reference_sol(sol, output_sol_ref, labels = ["uh", "eps_mech", "sig_mech"]): # Comparison
    Uh = sol['u'].function_space()    
    mesh = Uh.mesh()
    dim = sol['state_db'][0].function_space().num_sub_spaces()
    
    Sh0_DG = DDSpace(Uh, dim = dim, representation = 'DG') # for stress  
    sol_ref = {"state" : [DDFunction(Sh0_DG), DDFunction(Sh0_DG)],
                "u" : df.Function(sol['u'].function_space()) }
    
    with df.XDMFFile(output_sol_ref) as f:
        f.read_checkpoint(sol_ref['u'], labels[0])
        f.read_checkpoint(sol_ref['state'][0], labels[1])
        f.read_checkpoint(sol_ref['state'][1], labels[2])
        
    norm = lambda x : np.sqrt( df.assemble(df.inner(x,x)*Sh0_DG.dxm) )
    
    errors = {'u': norm(sol['u'] - sol_ref['u'])/norm(sol_ref['u']), 
               'sig_db': norm(sol['state_db'][1] - sol_ref['state'][1])/norm(sol_ref['state'][1]),
               'eps_db': norm(sol['state_db'][0] - sol_ref['state'][0])/norm(sol_ref['state'][0]),
               'sig_mech': norm(sol['state_mech'][1] - sol_ref['state'][1])/norm(sol_ref['state'][1]),
               'eps_mech': norm(sol['state_mech'][0] - sol_ref['state'][0])/norm(sol_ref['state'][0]) 
    }
    
    for key in errors.keys():
        print("Norm {:s} : {:e}".format(key, errors[key]) ) 
        
    print("{:e}, {:e}, {:e}, {:e}, {:e}".format(errors['u'], errors['sig_db'], errors['eps_db'], errors['sig_mech'], errors['eps_mech'] ) )

    return errors

def generate_vtk_db_mech(sol, output_vtk, output_sol = None, labels = ["uh", "eps", "sig"]):
    sol["u"].rename('uh', '')
    sol["state_mech"][1].rename(labels[2] + '_mech', '')
    sol["state_db"][1].rename(labels[2] + '_db', '')
    sol["state_mech"][0].rename(labels[1] + '_mech', '')
    sol["state_db"][0].rename(labels[2] + '_db', '')
    
    uh = sol["u"]
    state_mech = sol["state_mech"]
    state_db = sol["state_db"]
    
    Sh0 = state_mech[0].function_space()
    mesh = uh.function_space().mesh()
    
    Sh0_DG = df.VectorFunctionSpace(mesh, 'DG', degree = 0 , dim = Sh0.num_sub_spaces()) # for stress    
    
    sm = local_project_given_sol(state_mech[1], Sh0_DG, dxm = state_mech[0].dxm)
    sdb = local_project_given_sol(state_db[1], Sh0_DG, dxm = state_mech[0].dxm)
    em = local_project_given_sol(state_mech[0], Sh0_DG, dxm = state_mech[0].dxm)
    edb = local_project_given_sol(state_db[0], Sh0_DG, dxm = state_mech[0].dxm)
    
    sm.rename(labels[2] + '_mech', '')
    sdb.rename(labels[2] + '_db', '')
    
    em.rename(labels[1] + '_mech', '')
    edb.rename(labels[1] + '_db', '')
    
    fields = {'vertex': [uh], 'cell_vector': [sm, sdb, em, edb] }
    fields_sol = {'vertex': [uh], 'cell': [sm, sdb, em, edb] }
    iofe.exportXDMF_gen(output_vtk  , fields)
    if(type(output_sol) is not type(None)):
        iofe.exportXDMF_checkpoint_gen(output_sol  , fields_sol)

    
def db_mech_scatter_plot(sol, DB, namefig, fig_num = 1, title = ''):
    
    state_mech = sol["state_mech"]
    state_db = sol["state_db"]
    
    fig = plt.figure(fig_num, (6,4))
    plt.suptitle(title)
    
    # fig,(ax1,ax2) = plt.subplots(1,2)
    ax1 = fig.add_subplot(1,2,1)
    
    ax1.set_xlabel(r'$\epsilon_{11}+\epsilon_{22}$')
    ax1.set_ylabel(r'$\sigma_{11}+\sigma_{22}$')
    ax1.scatter(DB[:, 0, 0] + DB[:, 0, 1], DB[:, 1, 0] + DB[:, 1, 1], c='gray', label = 'DB')
    ax1.scatter(state_db[0].data()[:,0]+state_db[0].data()[:,1],state_db[1].data()[:,0]+state_db[1].data()[:,1], c='blue', label = 'D')
    ax1.scatter(state_mech[0].data()[:,0]+state_mech[0].data()[:,1],state_mech[1].data()[:,0]+state_mech[1].data()[:,1], marker = 'x', c='black' , label = 'E')
    ax1.legend(loc = 'best')
    ax1.grid()

    ax2 = fig.add_subplot(1,2,2)
    
    ax2.set_xlabel(r'$\epsilon_{12}$')
    ax2.set_ylabel(r'$\sigma_{12}$')
    ax2.scatter(DB[:, 0, 2], DB[:, 1, 2], c='gray', label = 'DB')
    ax2.scatter(state_db[0].data()[:,2], state_db[1].data()[:,2], c='blue', label = 'D')
    ax2.scatter(state_mech[0].data()[:,2], state_mech[1].data()[:,2], marker = 'x', c='black', label = 'E')
    ax2.legend(loc = 'best')
    ax2.grid()

    plt.tight_layout()
    plt.savefig(namefig)
    

def convergence_plot(hist, namefig, threshold = None, fig_num = 1):
    
    fig = plt.figure(fig_num)
    plt.plot(hist['relative_energy'], 'o-', label = "relative_energy")

    # plt.plot(hist['classical_relative_energy'], 'o-' , label = "classical_relative_energy")
    
    if(threshold is not None): 
         plt.plot([0,len(hist['relative_energy'])],[threshold,threshold])
    
    plt.yscale('log')
    plt.xlabel('iterations')
    plt.ylabel('distance')
    plt.legend(loc = 'best')
    plt.grid()
    
    plt.savefig(namefig)
        