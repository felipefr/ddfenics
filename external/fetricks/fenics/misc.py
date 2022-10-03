import dolfin as df
import numpy as np
from timeit import default_timer as timer


# NON-FLATTENED FUNCTIONS
def symgrad(v): 
    return df.sym(df.grad(v))
   

# Condensed version of the Integral: test
def Integral_shorter(u, dx, shape):
    if(len(shape) == 1):
        return np.array([ df.assemble(u[i]*dx) for i in range(shape[0])]) 

    elif(len(shape) == 2):
        return np.array( [ [ df.assemble(u[i, j]*dx) for j in range(shape[1])] 
                          for i in range(shape[0]) ])

# Vectorial and Tensorial integrals (Fenics integrals are scalars by default)
def Integral(u,dx,shape):
    
    n = len(shape)
    I = np.zeros(shape)
    
    if(type(dx) != type([])):
        dx = [dx]
 
    if(n == 1):
        for i in range(shape[0]):
            for dxj in dx:
                I[i] += df.assemble(u[i]*dxj)
            
    elif(n == 2):
        for i in range(shape[0]):
            for j in range(shape[1]):
                for dxk in dx:
                    I[i,j] += df.assemble(u[i,j]*dxk)
    
    else:
        print('not implement for higher order integral')
        
    
    return I

# Returns a expression affine tranformation x -> a + Bx (given a, B and a mesh)
def affineTransformationExpression(a,B, mesh):
    return df.Expression(('a0 + B00*x[0] + B01*x[1]','a1 + B10*x[0] + B11*x[1]'), a0 = a[0], a1 = a[1],
               B00=B[0,0], B01 = B[0,1], B10 = B[1,0], B11= B[1,1] ,degree = 1, domain = mesh)

# Returns a expression (x[0], x[1])
def VecX_expression(degree = 1):
    return df.Expression(('x[0]','x[1]'), degree = degree)


# Used to implement the Piola transofmation
class myfog(df.UserExpression): # fog f,g : R2 -> R2, generalise 
    def __init__(self, f, g, **kwargs):
        self.f = f 
        self.g = g
        f.set_allow_extrapolation(True)
        super().__init__(**kwargs)

    def eval(self, values, x):
        values[:2] = self.f(self.g(x))
        
    def value_shape(self):
        return (2,)

# Used to implement the Piola transofmation
class myfog_expression(df.UserExpression): # fog f,g : R2 -> R2, generalise 
    def __init__(self, f, g, **kwargs):
        self.f = f 
        self.g = g
        super().__init__(**kwargs)

    def eval(self, values, x):
        values[:2] = self.f(self.g(x))
        
    def value_shape(self):
        return (2,)


