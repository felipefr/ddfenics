import matplotlib.pyplot as plt
import numpy as np

def visualiseStresses(test, pred = None, figNum = 1, savefig = None):
    n = test.shape[1]
    indx =  np.arange(1,n,3)
    indy =  np.arange(2,n,3)

    plt.figure(figNum,(8,8))
    for i in range(test.shape[0]):
        plt.scatter(test[i,indx], test[i,indy], marker = 'o',  linewidth = 5)

        if(type(pred) != type(None)):
            plt.scatter(pred[i,indx], pred[i,indy], marker = '+', linewidth = 5)


    plt.legend(['test', 'pred'])
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.grid()
    
    if(type(savefig) != type(None)):
        plt.savefig(savefig)
 
    
def visualiseStresses9x9(test, pred = None, figNum = 1, savefig = None):
    n = test.shape[1]
    indx =  np.arange(1,n,3)
    indy =  np.arange(2,n,3)

    plt.figure(figNum,(13,12))
    for i in range(test.shape[0]):
        plt.subplot('33' + str(i+1))
        
        plt.scatter(test[i,indx], test[i,indy], marker = 'o',  linewidth = 5)
        if(type(pred) != type(None)):
            plt.scatter(pred[i,indx], pred[i,indy], marker = '+', linewidth = 5)

        plt.legend(['test', 'pred'])
        plt.xlim(0,1)
        plt.ylim(0,1)
        plt.grid()

    plt.subplots_adjust(wspace=0.3, hspace=0.25)
    
    if(type(savefig) != type(None)):
        plt.savefig(savefig)
 
    
    
 
def visualiseScatterErrors(test, pred, labels, gamma = 0.0, figNum = 1, savefig = None):
        
    plt.figure(figNum,(13,12))
    n = test.shape[1]
    
    for i in range(n): 
        plt.subplot('33' + str(i+1))
        plt.scatter(pred[:,i],test[:,i], marker = '+', linewidths = 0.1)
        xy = np.linspace(np.min(test[:,i]),np.max(test[:,i]),2)
        plt.plot(xy,xy,'-',color = 'black')
        plt.xlabel('test ' + labels[i])
        plt.ylabel('prediction ' + labels[i])
        plt.grid()
        
    for i in range(n): 
        plt.subplot('33' + str(i+4))
        plt.scatter(test[:,i],test[:,i] - pred[:,i], marker = '+', linewidths = 0.1)
        plt.xlabel('test ' + labels[i])
        plt.ylabel('error (test - pred) ' + labels[i])
        plt.grid()
    
    for i in range(n): 
        plt.subplot('33' + str(i+7))
        plt.scatter(test[:,i],(test[:,i] - pred[:,i])/(np.abs(test[:,i]) + gamma), marker = '+', linewidths = 0.1)
        plt.xlabel('test ' + labels[i])
        plt.ylabel('error rel (test - pred)/(test + gamma) ' + labels[i])
        plt.grid()
    
    plt.subplots_adjust(wspace=0.3, hspace=0.25)
    
    if(type(savefig) != type(None)):
        plt.savefig(savefig)


