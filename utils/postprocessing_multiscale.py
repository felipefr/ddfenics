#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 22 12:35:49 2022

@author: felipefr
"""


import sys, os
# import dolfin as df 
import ufl
from dolfinx import fem
import matplotlib.pyplot as plt
import numpy as np
import copy 
from timeit import default_timer as timer

import fetricksx as ft
import ddfenicsx as dd

def get_errors_DD(sol, sol_ref, norm):
    gdim = sol_ref['u'].function_space().mesh().geometric_dimension()
    norm_max = lambda X: np.max(np.linalg.norm(X, axis = 1))
    dist_rel = lambda x, x0, norm : norm(x - x0)/norm(x0)
    
    errors = {}
    errors['e_u_L2'] = dist_rel(sol['u'], sol_ref['u'], norm)
    errors['e_eps_L2'] = dist_rel(sol['state_mech'][0], sol_ref['state'][0], norm)
    errors['e_sig_L2'] = dist_rel(sol['state_mech'][1], sol_ref['state'][1], norm)
    errors['e_eps_dd_L2'] = dist_rel(sol['state_db'][0], sol['state_mech'][0], norm)
    errors['e_sig_dd_L2'] = dist_rel(sol['state_db'][1], sol['state_mech'][1], norm)

    errors['e_u_max'] = dist_rel(sol['u'].vector().get_local().reshape((-1,gdim)), 
                       sol_ref['u'].vector().get_local().reshape((-1,gdim)), norm_max)
    errors['e_eps_max'] = dist_rel(sol['state_mech'][0].data(), sol_ref['state'][0].data(), norm_max)
    errors['e_sig_max'] = dist_rel(sol['state_mech'][1].data(), sol_ref['state'][1].data(), norm_max)
    errors['e_eps_dd_max'] = dist_rel(sol['state_db'][0].data(), sol['state_mech'][0].data(), norm_max)
    errors['e_sig_dd_max'] = dist_rel(sol['state_db'][1].data(), sol['state_mech'][1].data(), norm_max)

    return errors


def callback_get_errors(k, solver, sol_ref, error_list, norm):
    sol = solver.get_sol()
    error_list.append(np.array(list(get_errors_DD(sol, sol_ref, norm).values())))
    
def callback_get_DBsize(k, solver, list_DBsize):
    ddmat = solver.ddmat
    list_DBsize.append(len(ddmat.DB))
    
def callback_get_time_elapsed(k, solver, list_time, start_time = 0.0):
    list_time.append(timer() - start_time)
    
def generate_vtk_db_mech(sol, output_vtk, output_sol = None):
    sol["u"].rename('uh', '')
    sol["state_mech"][1].rename('sigma_mech', '')
    sol["state_db"][1].rename('sigma_db', '')
    sol["state_mech"][0].rename('eps_mech', '')
    sol["state_db"][0].rename('eps_db', '')
    
    uh = sol["u"]
    state_mech = sol["state_mech"]
    state_db = sol["state_db"]
    
    Sh0 = state_mech[0].function_space()
    mesh = uh.function_space().mesh()
    
    Sh0_DG = fem.functionspace(mesh, ('DG', 0, (Sh0.num_sub_spaces(),)) ) # for stress    
    
    sm = ft.local_project_given_sol(state_mech[1], Sh0_DG, dxm = state_mech[0].dxm)
    sdb = ft.local_project_given_sol(state_db[1], Sh0_DG, dxm = state_mech[0].dxm)
    em = ft.local_project_given_sol(state_mech[0], Sh0_DG, dxm = state_mech[0].dxm)
    edb = ft.local_project_given_sol(state_db[0], Sh0_DG, dxm = state_mech[0].dxm)
    
    sm.rename("sigma_mech", '')
    sdb.rename("sigma_db", '')
    
    em.rename("eps_mech", '')
    edb.rename("eps_db", '')
    
    fields = {'vertex': [uh], 'cell_vector': [sm, sdb, em, edb] }
    fields_sol = {'vertex': [uh], 'cell': [sm, sdb, em, edb] }
    ft.exportXDMF_gen(output_vtk  , fields)
    if(type(output_sol) is not type(None)):
        ft.exportXDMF_checkpoint_gen(output_sol  , fields_sol)

    
def db_mech_scatter_plot(sol, DB, namefig, fig_num = 1, title = ''):
    
    import fetricks.plotting.misc as feplt
    
    feplt.loadLatexOptions()


    state_mech = sol["state_mech"]
    state_db = sol["state_db"]
    
    fig = plt.figure(fig_num, (6,4))
    plt.suptitle(title)
    
    # fig,(ax1,ax2) = plt.subplots(1,2)
    ax1 = fig.add_subplot(1,2,1)
    
    ax1.set_xlabel(r'$\Large \epsilon_{11}+\epsilon_{22}$')
    ax1.set_ylabel(r'\Large $\sigma_{11}+\sigma_{22}$')
    ax1.scatter(DB[:, 0, 0] + DB[:, 0, 1], DB[:, 1, 0] + DB[:, 1, 1], c='gray', label = 'DB')
    ax1.scatter(state_db[0].data()[:,0]+state_db[0].data()[:,1],state_db[1].data()[:,0]+state_db[1].data()[:,1], c='blue', label = 'D')
    ax1.scatter(state_mech[0].data()[:,0]+state_mech[0].data()[:,1],state_mech[1].data()[:,0]+state_mech[1].data()[:,1], marker = 'x', c='black' , label = 'E')
    ax1.legend(loc = 'best')
    ax1.grid()

    ax2 = fig.add_subplot(1,2,2)
    
    ax2.set_xlabel(r'\Large $\epsilon_{12}$')
    ax2.set_ylabel(r'\Large $\sigma_{12}$')
    ax2.scatter(DB[:, 0, 2], DB[:, 1, 2], c='gray', label = 'DB')
    ax2.scatter(state_db[0].data()[:,2], state_db[1].data()[:,2], c='blue', label = 'D')
    ax2.scatter(state_mech[0].data()[:,2], state_mech[1].data()[:,2], marker = 'x', c='black', label = 'E')
    ax2.legend(loc = 'best')
    ax2.grid()

    plt.tight_layout()
    plt.savefig(namefig)
    

    


# def get_sol_ref_with_mesh(output_sol_ref, mesh_macro_name,  deg_u = 1, Sdim = 3):
#     mesh = ft.Mesh(mesh_macro_name)
#     Uh = ufl.VectorFunctionSpace(mesh, "CG", deg_u)
#     Sh0_DG = dd.DDSpace(Uh, Sdim, 'DG')    
#     sol_ref = dd.get_sol_ref(output_sol_ref, Sh0_DG, Uh)
    
#     return sol_ref

# def get_sol_ref(output_sol_ref, Sh0_DG, Uh):
       
#     sol_ref = {"state" : [dd.DDFunction(Sh0_DG, Sh0_DG.dxm), dd.DDFunction(Sh0_DG, Sh0_DG.dxm)],
#                 "u" : ufl.Function(Uh) }
    
#     with ufl.XDMFFile(output_sol_ref) as f:
#         f.read_checkpoint(sol_ref['state'][1], 'sig_mech', 0)
#         f.read_checkpoint(sol_ref['state'][0], 'eps_mech', 0)
#         f.read_checkpoint(sol_ref['u'], 'u')
        
#     return sol_ref


# def get_state_database_from_xdmf(output_sol, mesh_name,  deg_u = 1, Sdim = 3):
    
#     mesh = ft.Mesh(mesh_name)
#     Uh = ufl.VectorFunctionSpace(mesh, "CG", deg_u)
#     Sh0_DG = dd.DDSpace(Uh, Sdim, 'DG')    
#     sol_ref = get_sol_ref(output_sol, Sh0_DG, Uh)
    
    
#     sol_ref = {"state" : [dd.DDFunction(Sh0_DG, Sh0_DG.dxm), dd.DDFunction(Sh0_DG, Sh0_DG.dxm)],
#                 "u" : ufl.Function(Uh) }
    
#     data = np.concatenate((sol_ref["state"][0].vector().get_local().reshape((-1,Sdim)), 
#                            sol_ref["state"][1].vector().get_local().reshape((-1,Sdim))), axis = 1)
    
        
#     return data


# def export_database_from_simul(output_sol, mesh_name, database_name, deg_u = 1, Sdim = 3):
#     sol_ref = get_sol_ref_with_mesh(output_sol, mesh_name, deg_u, Sdim)
#     data = np.concatenate((sol_ref["state"][0].data(), sol_ref["state"][1].data()), axis = 1)
#     np.savetxt(database_name, data, header = '1.0 \n%d 2 %d %d'%(data.shape[0], Sdim, Sdim), comments = '', fmt='%.8e')
#     return data



# def comparisonWithReference(sol, output_sol_ref): # Comparison
#     Uh = sol['u'].function_space()    
#     dim = sol['state_db'][0].function_space().num_sub_spaces()
    
#     Sh0_DG = dd.DDSpace(Uh, dim = dim, representation = 'DG') # for stress  
#     sol_ref = {"state" : [dd.DDFunction(Sh0_DG), dd.DDFunction(Sh0_DG)],
#                 "u" : ufl.Function(sol['u'].function_space()) }
    
#     with ufl.XDMFFile(output_sol_ref) as f:
#         f.read_checkpoint(sol_ref['state'][1], "sig_mech")
#         f.read_checkpoint(sol_ref['state'][0], "eps_mech")
#         f.read_checkpoint(sol_ref['u'], "uh")
        
#     norm = lambda x : np.sqrt( ufl.assemble(ufl.inner(x,x)*Sh0_DG.dxm) )
    
#     errors = {'u': norm(sol['u'] - sol_ref['u'])/norm(sol_ref['u']), 
#                'sig_db': norm(sol['state_db'][1] - sol_ref['state'][1])/norm(sol_ref['state'][1]),
#                'eps_db': norm(sol['state_db'][0] - sol_ref['state'][0])/norm(sol_ref['state'][0]),
#                'sig_mech': norm(sol['state_mech'][1] - sol_ref['state'][1])/norm(sol_ref['state'][1]),
#                'eps_mech': norm(sol['state_mech'][0] - sol_ref['state'][0])/norm(sol_ref['state'][0]) 
#     }
    
#     for key in errors.keys():
#         print("Norm {:s} : {:e}".format(key, errors[key]) ) 
        
#     print("{:e}, {:e}, {:e}, {:e}, {:e}".format(errors['u'], errors['sig_db'], errors['eps_db'], errors['sig_mech'], errors['eps_mech'] ) )
 


# def convergence_plot(hist, namefig, threshold = None, fig_num = 1):

#     plt.plot(hist['relative_energy'], 'o-', label = "relative_energy")

#     # plt.plot(hist['classical_relative_energy'], 'o-' , label = "classical_relative_energy")
    
#     if(threshold is not None): 
#          plt.plot([0,len(hist['relative_energy'])],[threshold,threshold])
    
#     plt.yscale('log')
#     plt.xlabel('iterations')
#     plt.ylabel('distance')
#     plt.legend(loc = 'best')
#     plt.grid()
    
#     plt.savefig(namefig)
        

# def callback_generate_vtk(output_vtk, k, solver):
#     problem = solver.problem
#     ft.exportXDMF_gen( output_vtk.format(k), {'vertex': [problem.u], 'cell_vector': problem.z_mech + problem.z_db}, k) # sum of lists

# def callback_evolution_state(dbplot_file, k, solver):
#     problem = solver.problem
#     state_mech = problem.z_mech
#     state_db = problem.z_db
    
#     fig,(ax1,ax2) = plt.subplots(1,2)
#     ax1.set_xlabel(r'$\epsilon_{xx}+\epsilon_{yy}$')
#     ax1.set_ylabel(r'$\sigma_{xx}+\sigma_{yy}$')
#     ax1.scatter(state_mech[0].data()[:,0]+state_mech[0].data()[:,1],state_mech[1].data()[:,0]+state_mech[1].data()[:,1], c='gray')
#     ax1.scatter(state_db[0].data()[:,0]+state_db[0].data()[:,1],state_db[1].data()[:,0]+state_db[1].data()[:,1], marker = 'x', c='black')

#     ax2.set_xlabel(r'$\epsilon_{xy}$')
#     ax2.set_ylabel(r'$\sigma_{xy}$')
#     ax2.scatter(state_mech[0].data()[:,2], state_mech[1].data()[:,2], c='gray')
#     ax2.scatter(state_db[0].data()[:,2], state_db[1].data()[:,2], marker = 'x', c='black')
    
#     plt.savefig(dbplot_file.format(k))
    
#     plt.close()


# def callback_evolution_db(dbplot_file, k, solver):
    
#     problem = solver.problem
#     state_db = [ problem.ddmat.DB[:,0,:] , problem.ddmat.DB[:,1,:] ] 
    
#     fig,(ax1,ax2) = plt.subplots(1,2)
#     ax1.set_xlabel(r'$\epsilon_{xx}+\epsilon_{yy}$')
#     ax1.set_ylabel(r'$\sigma_{xx}+\sigma_{yy}$')
#     ax1.scatter(state_db[0][:,0]+state_db[0][:,1],state_db[1][:,0]+state_db[1][:,1], marker = 'x', c='black')

#     ax2.set_xlabel(r'$\epsilon_{xy}$')
#     ax2.set_ylabel(r'$\sigma_{xy}$')
#     ax2.scatter(state_db[0][:,2], state_db[1][:,2], marker = 'x', c='black')
    
#     plt.savefig(dbplot_file.format(k))
    
#     plt.close()
    

# def callback_get_update_counter_field(k, solver, list_upd_field):
#     list_upd_field.append(copy.deepcopy(solver.updater.count_field))

# def write_vtk_from_array(array, S, output_vtk, label):
#     v = ufl.Function(S)
#     v.rename(label, '')

#     with ufl.XDMFFile(output_vtk) as ofile: 
#         ofile.parameters["flush_output"] = True
#         ofile.parameters["functions_share_mesh"] = True
        
#         for k in range(len(array)):
#             v.vector().set_local(array[k])
#             ofile.write(v, k, encoding = ufl.XDMFFile.Encoding.HDF5) 
