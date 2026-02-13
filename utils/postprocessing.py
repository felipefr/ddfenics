# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """
# Created on Fri Jul 22 12:35:49 2022

# @author: felipefr
# """


# import sys, os
# import dolfin as df 
# import matplotlib.pyplot as plt
# import numpy as np
# import fetricks.fenics.postprocessing.wrapper_io as iofe
# from fetricks.fenics.la.wrapper_solvers import local_project_given_sol

# from ddfenics.dd.ddfunction import DDFunction
# from ddfenics.dd.ddspace import DDSpace

# def comparison_with_reference_sol(sol, sol_ref = None, output_sol_ref = None, labels = ["uh", "eps_mech", "sig_mech"], Vref = None, dxm = None): # Comparison
#     Uh = sol['u'].function_space()    
#     mesh = Uh.mesh()
#     dim = sol['state_db'][0].function_space().num_sub_spaces()
    
#     if(not Vref):
#         Vref = DDSpace(Uh, dim = dim, representation = 'DG') # for stress          
#         dxm = Vref.dxm
    
#     norm = lambda x : np.sqrt( df.assemble(df.inner(x,x)*dxm) )
    
#     if(output_sol_ref):
#         sol_ref = {"state" : [DDFunction(Vref), DDFunction(Vref)],
#                     "u" : df.Function(sol['u'].function_space()) }

#         with df.XDMFFile(output_sol_ref) as f:
#             f.read_checkpoint(sol_ref['u'], labels[0])
#             f.read_checkpoint(sol_ref['state'][0], labels[1])
#             f.read_checkpoint(sol_ref['state'][1], labels[2])
        
    
    
#     errors = {'u': norm(sol['u'] - sol_ref['u'])/norm(sol_ref['u']), 
#                'sig_db': norm(sol['state_db'][1] - sol_ref['state'][1])/norm(sol_ref['state'][1]),
#                'eps_db': norm(sol['state_db'][0] - sol_ref['state'][0])/norm(sol_ref['state'][0]),
#                'sig_mech': norm(sol['state_mech'][1] - sol_ref['state'][1])/norm(sol_ref['state'][1]),
#                'eps_mech': norm(sol['state_mech'][0] - sol_ref['state'][0])/norm(sol_ref['state'][0]) 
#     }
    
#     for key in errors.keys():
#         print("Norm {:s} : {:e}".format(key, errors[key]) ) 
        
#     print("{:e}, {:e}, {:e}, {:e}, {:e}".format(errors['u'], errors['sig_db'], errors['eps_db'], errors['sig_mech'], errors['eps_mech'] ) )

#     return errors

# def generate_vtk_db_mech(sol, ShDD, output_vtk, output_sol = None, labels = ["uh", "eps", "sig"]):
#     sol["u"].rename('uh', '')
#     sol["state_mech"][1].rename(labels[2] + '_mech', '')
#     sol["state_db"][1].rename(labels[2] + '_db', '')
#     sol["state_mech"][0].rename(labels[1] + '_mech', '')
#     sol["state_db"][0].rename(labels[2] + '_db', '')
    
#     uh = sol["u"]
#     state_mech = sol["state_mech"]
#     state_db = sol["state_db"]
    
#     Sh0 = state_mech[0].function_space()
#     mesh = uh.function_space().mesh()
    
#     Sh0_DG = df.VectorFunctionSpace(mesh, 'DG', degree = 0 , dim = Sh0.num_sub_spaces()) # for stress    
    
    
#     dxm = ShDD.dxm
    
#     sm = local_project_given_sol(state_mech[1], Sh0_DG, dxm = dxm)
#     sdb = local_project_given_sol(state_db[1], Sh0_DG, dxm = dxm)
#     em = local_project_given_sol(state_mech[0], Sh0_DG, dxm = dxm)
#     edb = local_project_given_sol(state_db[0], Sh0_DG, dxm = dxm)
    
#     sm.rename(labels[2] + '_mech', '')
#     sdb.rename(labels[2] + '_db', '')
    
#     em.rename(labels[1] + '_mech', '')
#     edb.rename(labels[1] + '_db', '')
    
#     fields = {'vertex': [uh], 'cell_vector': [sm, sdb, em, edb] }
#     fields_sol = {'vertex': [uh], 'cell': [sm, sdb, em, edb] }
#     iofe.exportXDMF_gen(output_vtk  , fields)
#     if(type(output_sol) is not type(None)):
#         iofe.exportXDMF_checkpoint_gen(output_sol  , fields_sol)

    
# def db_mech_scatter_plot(sol, DB, namefig, fig_num = 1, title = ''):
    
#     from fetricks.plotting import misc

#     misc.load_latex_options()
    
#     state_mech = sol["state_mech"]
#     state_db = sol["state_db"]
    
#     fig = plt.figure(fig_num, (6,4))
#     plt.suptitle(title)
    
#     # fig,(ax1,ax2) = plt.subplots(1,2)
#     ax1 = fig.add_subplot(1,2,1)
    
#     ax1.set_xlabel('$\\varepsilon_{11}+\\varepsilon_{22}$')
#     ax1.set_ylabel('$\sigma_{11}+\sigma_{22}$')
#     ax1.scatter(DB[:, 0, 0] + DB[:, 0, 1], DB[:, 1, 0] + DB[:, 1, 1], c='gray', label = 'DB')
#     ax1.scatter(state_db[0].data()[:,0]+state_db[0].data()[:,1],state_db[1].data()[:,0]+state_db[1].data()[:,1], c='blue', label = 'D')
#     ax1.scatter(state_mech[0].data()[:,0]+state_mech[0].data()[:,1],state_mech[1].data()[:,0]+state_mech[1].data()[:,1], marker = 'x', c='black' , label = 'E')
#     ax1.legend(loc = 'best')
#     ax1.grid()

#     ax2 = fig.add_subplot(1,2,2)
    
#     ax2.set_xlabel('$\\varepsilon_{12}$')
#     ax2.set_ylabel('$\sigma_{12}$')
#     ax2.scatter(DB[:, 0, 2], DB[:, 1, 2], c='gray', label = 'DB')
#     ax2.scatter(state_db[0].data()[:,2], state_db[1].data()[:,2], c='blue', label = 'D')
#     ax2.scatter(state_mech[0].data()[:,2], state_mech[1].data()[:,2], marker = 'x', c='black', label = 'E')
#     ax2.legend(loc = 'best')
#     ax2.grid()

#     plt.tight_layout()
#     plt.savefig(namefig)
    
# def db_mech_scatter_plot(sol, DB, namefig, fig_num = 1, title = '', op =1):
    
#     from fetricks.plotting import misc

#     misc.load_latex_options()
    
#     state_mech = sol["state_mech"]
#     state_db = sol["state_db"]
    
#     fig = plt.figure(fig_num, (8,3))
#     # plt.suptitle(title)
#     plt.suptitle("_")
    
    
#     if(op == 1): # classical plot strain - stress 
#         ax1 = fig.add_subplot(1,2,1)
        
#         ax1.set_xlabel('$\\varepsilon_{11}+\\varepsilon_{22}$')
#         ax1.set_ylabel('$\sigma_{11}+\sigma_{22}$')
#         ax1.scatter(DB[:, 0, 0] + DB[:, 0, 1], DB[:, 1, 0] + DB[:, 1, 1], c='gray', label = 'DB')
#         ax1.scatter(state_db[0].data()[:,0]+state_db[0].data()[:,1],state_db[1].data()[:,0]+state_db[1].data()[:,1], c='blue', label = 'D')
#         ax1.scatter(state_mech[0].data()[:,0]+state_mech[0].data()[:,1],state_mech[1].data()[:,0]+state_mech[1].data()[:,1], marker = 'x', c='black' , label = 'E')
#         ax1.legend(loc = 'best')
#         ax1.grid()
    
#         ax2 = fig.add_subplot(1,2,2)
        
#         ax2.set_xlabel('$\\varepsilon_{12}$')
#         ax2.set_ylabel('$\sigma_{12}$')
#         ax2.scatter(DB[:, 0, 2], DB[:, 1, 2], c='gray', label = 'DB')
#         ax2.scatter(state_db[0].data()[:,2], state_db[1].data()[:,2], c='blue', label = 'D')
#         ax2.scatter(state_mech[0].data()[:,2], state_mech[1].data()[:,2], marker = 'x', c='black', label = 'E')
#         ax2.legend(loc = 'best')
#         ax2.grid()

#     if(op == 2): # plot strain only 
    
#         hfm = np.sqrt(2.)**-1
        
#         ax1 = fig.add_subplot(1,3,1)
        
#         ax1.set_xlabel('$\\varepsilon_{11}$')
#         ax1.set_ylabel('$\\varepsilon_{22}$')
#         ax1.scatter(DB[:, 0, 0], DB[:, 0, 1], c='gray', 
#                     s = 5, marker = '.', label = '$D_L$')
#         ax1.scatter(state_db[0].data()[:,0], state_db[0].data()[:,1], 
#                     marker = '.', s=6, c='blue', label = '$\mathcal{Z}_D$')
#         # ax1.legend(loc = 'best')
#         ax1.grid()
        
#         handles, labels = ax1.get_legend_handles_labels()
#         fig.legend(handles, labels, loc='upper center', ncols = 2)
    
#         ax2 = fig.add_subplot(1,3,2)
        
#         ax2.set_xlabel('$\\varepsilon_{11}$')
#         ax2.set_ylabel('$\\varepsilon_{12}$')
#         ax2.scatter(DB[:, 0, 0], hfm*DB[:, 0, 2], c='gray', 
#                     s = 5, marker = '.', label = '$D_L$')
#         ax2.scatter(state_db[0].data()[:,0], hfm*state_db[0].data()[:,2], 
#                     marker = '.', s =6, c='blue', label = '$\mathcal{Z}_D$')
#         # ax2.legend(loc = 'best')
#         ax2.grid()
        
        
#         ax3 = fig.add_subplot(1,3,3)
        
#         ax3.set_xlabel('$\\varepsilon_{22}$')
#         ax3.set_ylabel('$\\varepsilon_{12}$')
#         ax3.scatter(DB[:, 0, 1], hfm*DB[:, 0, 2], c='gray', 
#                     s = 5, marker = '.', label = '$D_L$')
#         ax3.scatter(state_db[0].data()[:,1], hfm*state_db[0].data()[:,2], 
#                     marker = '.', s = 6, c='blue', label = '$\mathcal{Z}_D$')
#         # ax3.legend(loc = 'best')
#         ax3.grid()
        
#     plt.tight_layout()
#     plt.savefig(namefig, dpi=300)
    

# def convergence_plot(hist, namefig, threshold = None, fig_num = 1):
    
#     fig = plt.figure(fig_num)
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
        