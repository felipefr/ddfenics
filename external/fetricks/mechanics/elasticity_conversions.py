import numpy as np


def convertParam(param,foo):
    
    n = len(param)
    paramNew = np.zeros((n,2))
    for i in range(n):
        paramNew[i,0], paramNew[i,1] = foo( *param[i,:].tolist()) 
  
    return paramNew

convertParam2 = lambda p,f: np.array( [  f(*p_i) for p_i in p ] )

def youngPoisson2lame_planeStress(nu,E):
    lamb , mu = youngPoisson2lame(nu,E)
    
    lamb = (2.0*mu*lamb)/(lamb + 2.0*mu)
    
    return lamb, mu

def Celas_voigt(lamb,mu):
    return np.array( [[lamb + 2*mu, lamb, 0], [lamb, lamb + 2*mu, 0], [0, 0, mu]] )
    
youngPoisson2lame = lambda nu,E : [ nu * E/((1. - 2.*nu)*(1.+nu)) , E/(2.*(1. + nu)) ] # lamb, mu

gof = lambda g,f: lambda x,y : g(*f(x,y)) # composition, g : R2 -> R* , f : R2 -> R2

lame2youngPoisson  = lambda lamb, mu : [ 0.5*lamb/(mu + lamb) , mu*(3.*lamb + 2.*mu)/(lamb + mu) ]
youngPoisson2lame = lambda nu,E : [ nu * E/((1. - 2.*nu)*(1.+nu)) , E/(2.*(1. + nu)) ]


lame2lameStar = lambda lamb, mu: [(2.0*mu*lamb)/(lamb + 2.0*mu), mu]
lameStar2lame = lambda lambStar, mu: [(2.0*mu*lambStar)/(-lambStar + 2.0*mu), mu]

eng2lamb = lambda nu, E: nu * E/((1. - 2.*nu)*(1.+nu))
eng2mu = lambda nu, E: E/(2.*(1. + nu))
lame2poisson = lambda lamb, mu: 0.5*lamb/(mu + lamb)
lame2young = lambda lamb, mu: mu*(3.*lamb + 2.*mu)/(lamb + mu)
lame2lambPlane = lambda lamb, mu: (2.0*mu*lamb)/(lamb + 2.0*mu)
lamePlane2lamb = lambda lambStar, mu: (2.0*mu*lambStar)/(-lambStar + 2.0*mu)
eng2mu = lambda nu, E: E/(2.*(1. + nu))
eng2lambPlane = gof(lame2lambPlane,lambda x,y: (eng2lamb(x,y), eng2mu(x,y)))
