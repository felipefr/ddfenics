import dolfin as df
import numpy as np
import fetricks.fenics.misc as misc

# PIOLA TRANSFORMATIONS 
def PiolaTransform_rotation(theta, Vref): #only detB = pm 1 in our context
    Mref = Vref.mesh()    
    
    s = np.sin(theta) ; c = np.cos(theta)
    B = np.array([[c,-s],[s,c]])

    Finv = misc.affineTransformationExpression(np.zeros(2), B.T, Mref)
    Bmultiplication = misc.affineTransformationExpression(np.zeros(2), B, Mref)
    
    return lambda sol: df.interpolate( misc.myfog_expression(Bmultiplication, misc.myfog(sol,Finv)), Vref) #


def PiolaTransform_rotation_matricial(theta, Vref): #only detB = pm 1 in our context
    Nh = Vref.dim()    
    Piola = PiolaTransform_rotation(theta,Vref)
    Pmat = np.zeros((Nh,Nh))
    
    phi_j = df.Function(Vref)
    ej = np.zeros(Nh) 
    
    for j in range(Nh):
        ej[j] = 1.0
        phi_j.vector().set_local(ej)
        Pmat[:,j] = Piola(phi_j).vector().get_local()[:]
        ej[j] = 0.0
    
    return Pmat
        

