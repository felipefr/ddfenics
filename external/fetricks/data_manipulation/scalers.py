import os, sys
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import matplotlib.pyplot as plt
import numpy as np
import fetricks.data_manipulation.wrapper_h5py as myhd

def constructor_scaler(scalerType):    
    return {'MinMax': myMinMaxScaler() , 
            'Normalisation': myNormalisationScaler(), 
            'MinMax11': myMinMax11Scaler()}[scalerType]


class myScaler:
    
    ef __init__(self):
        self.n = 0
        
    def set_n(self,n):
        pass
    
    def load_param(self, param):
        pass 
    
    def export_scale(self):
        pass 
           
    def fit(self,x):
        pass
    
    def scaler(self, x, i):
        pass 
    
    def inv_scaler(self, x, i):
        pass 
    
    def transform(self,x):
        return np.array( [self.scaler(x[:,i],i) for i in range(self.n)] ).T
            
    def inverse_transform(self,x):
        return np.array( [self.inv_scaler(x[:,i],i) for i in range(self.n)] ).T


class myMinMaxScaler(myScaler):
    def __init__(self, eps_margin = 0.05):
        self.data_min_ = []
        self.data_max_ = []
        
        self.eps_margin = eps_margin
        super().__init__()
                
    def set_n(self,n):
        self.n = n

        self.data_min_ = self.data_min_[:self.n]
        self.data_max_ = self.data_max_[:self.n]

    def scaler(self, x, i):
        return (x - self.data_min_[i])/(self.data_max_[i]-self.data_min_[i])
    
    def inv_scaler(self, x, i):
        return (self.data_max_[i]-self.data_min_[i])*x + self.data_min_[i]

    def load_param(self, param):
        self.n = param.shape[0]
                
        self.data_min_ = param[:,0]
        self.data_max_ = param[:,1]
        
    def export_scale(self):
        return np.stack( (self.data_min_, self.data_max_ )).T
            
    def fit(self,x):
        self.n = x.shape[1]
        
        self.data_min_ = np.min(x , axis = 0)  
        self.data_max_ = np.max(x, axis = 0)
        
        max_abs = np.max(np.abs(np.stack((self.data_min_, self.data_max_))), axis = 0) # per feature
        
        self.data_min_ = self.data_min_ - self.eps_margin*max_abs
        self.data_max_ = self.data_max_ + self.eps_margin*max_abs
        
        print(self.data_min_.shape)
    
class myMinMax11Scaler(myMinMaxScaler):
    
        
    def scaler(self, x, i):
        return 2*(x - self.data_min_[i])/(self.data_max_[i]-self.data_min_[i]) - 1.0
    
    def inv_scaler(self, x, i):
        return (self.data_max_[i]-self.data_min_[i])*0.5*(x+1) + self.data_min_[i]


class myNormalisationScaler(myScaler):
    def __init__(self):
        self.data_mean = []
        self.data_std = []
        super().__init__()
        
    def set_n(self,n):
        self.n = n
        self.data_mean = self.data_mean[:self.n]
        self.data_std = self.data_std[:self.n]

    def scaler(self, x, i): 
        return (x - self.data_mean[i])/self.data_std[i]
    
    def inv_scaler(self, x, i):
        return (x*self.data_std[i] + self.data_mean[i])

    def load_param(self, param):
        self.n = param.shape[0]
                
        self.data_mean = param[:,0]
        self.data_std = param[:,1]
        
    def export_scale(self):
        return np.stack( (self.data_mean, self.data_std)).T
                             
    def fit(self,x):
        self.n = x.shape[1]
        
        for i in range(self.n):
            self.data_mean.append(np.mean(x[:,i]))
            self.data_std.append(np.std(x[:,i]))
        
        self.data_mean = np.array(self.data_mean)
        self.data_std = np.array(self.data_std)


    def transform(self,x):
        return np.array( [self.scaler(x[:,i],i) for i in range(self.n)] ).T
            
    def inverse_transform(self,x):
        return np.array( [self.inv_scaler(x[:,i],i) for i in range(self.n)] ).T


def exportScale(filenameIn, filenameOut, nX, nY, Ylabel = 'Y', scalerType = "MinMax"):
    scalerX, scalerY = getDatasetsXY(nX, nY, filenameIn, Ylabel = Ylabel, scalerType = scalerType)[2:4]
    scalerLimits = np.zeros((max(nX,nY),4))
    scalerLimits[:nX,0:2] = scalerX.export_scale()
    scalerLimits[:nY,2:4] = scalerY.export_scale()

    np.savetxt(filenameOut, scalerLimits)

def importScale(filenameIn, nX, nY, scalerType = 'MinMax'): ## It was wrong (I have to use it wrongly: Use minmax with the normalization values)

    scalerX = constructor_scaler(scalerType)      
    scalerY = constructor_scaler(scalerType)      
    scalerX.load_param(np.loadtxt(filenameIn)[:,0:2])
    scalerY.load_param(np.loadtxt(filenameIn)[:,2:4])
    scalerX.set_n(nX)
    scalerY.set_n(nY)
    
    return scalerX, scalerY

            
#     return scalerX.transform(X), scalerY.transform(Y), scalerX, scalerY
