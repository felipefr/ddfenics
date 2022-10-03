#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 29 16:46:06 2022

@author: felipe
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 11 17:54:10 2022

@author: felipe
"""

import numpy as np
import dolfin as df
import ufl

# This is based on the DDbilinear
class DDProblemAbstract:
    
    def __init__(self, a, L, sol, bcs, form_compiler_parameters = {}, bcsPF = [], bcs_eta = []):
        
        self.a = a
        self.L = L
        self.u = sol["u"]
        self.ddmat = self.a.ddmat
        self.dx = self.a.dx
        self.bcs = bcs
        self.bcs_eta = bcs_eta if len(bcs_eta)>0 else self.bcs
        self.bcsPF = bcsPF
        self.z_mech = sol["state_mech"]
        self.z_db = sol["state_db"]
        self.eta = df.Function(self.u.function_space())
        
        self.resetDB()
        self.createSubproblems()
        
    def reset(self, p):
        self.__init__(p.a, p.L, p.u, p.bcs, p.z_mech, p.z_db, bcs_eta = p.bcs_eta)

      
    def resetDB(self):
        self.DB = self.a.ddmat.DB.view()
        self.Nd = self.DB.shape[0]
    
    def createSubproblems(self):
        pass
        
    def update_state_mech(self):
        
        state_update = ( self.a.ddmat.grad(self.u), self.z_db[1] + self.a.action_disp(self.eta) )
        
        if(type(self.z_mech) == type([])):
            for i, z_i in enumerate(self.z_mech):
                z_i.update(state_update[i])
                
        else: # all state variables are in a same object
            self.z_mech.update(df.as_tensor(state_update))
                    
    def update_state_db(self, map):
        Nmap = len(map)
        
        if(type(self.z_db) == type([])):
            for i, z_i in enumerate(self.z_db):
                z_i.update(self.DB[map,i,:].reshape((Nmap,-1)))

        else: # all state variables are in a same object
            self.z_db.update(self.DB[map,:,:].reshape((Nmap,-1)))
            

    def get_state_mech_data(self):
        if(type(self.z_mech) == type([])):    
            return np.concatenate(tuple([z_i.data() for i, z_i in enumerate(self.z_mech)]) , axis = 1)
        
        else:    
            return self.z_mech.data()
        
        
        
        

    #     if(self.Sh0.ufl_element().family() == 'Quadrature'):
    #         metadata = {"quadrature_degree": Sh0.ufl_element().degree(), "quadrature_scheme": "default"}
    #         self.dx = df.Measure('dx', Uh.mesh(), metadata = metadata)
    #     else:
    #         self.dx = df.Measure('dx', Uh.mesh())
