import matplotlib.pyplot as plt
import matplotlib
import numpy as np

plt.rc("text", usetex = True)
plt.rc("font", family = 'serif')
plt.rc("font", size = 12)
plt.rc('text.latex', preamble=r'\usepackage{amsmath,amsfonts}')
# matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]

palletteCounter = 0
pallette = ['blue','red','green']

def plotMeanAndStd(x, y, l='', linetypes = ['-o','--','--'], axis = 0):
    plt.plot(x, np.mean(y, axis = axis), linetypes[0], label = l)
    plt.plot(x, np.mean(y, axis = axis) + np.std(y, axis = axis) , linetypes[1], label = l + ' + std')
    plt.plot(x, np.mean(y, axis = axis) - np.std(y, axis = axis) , linetypes[2], label = l + ' - std')
    
    
    
def plotMeanAndStd_noStdLegend(x, y, l='', linetypes = ['-o','--','--'], axis = 0):
    plt.plot(x, np.mean(y, axis = axis), linetypes[0], label = l)
    plt.plot(x, np.mean(y, axis = axis) + np.std(y, axis = axis) , linetypes[1])
    plt.plot(x, np.mean(y, axis = axis) - np.std(y, axis = axis) , linetypes[2])
    
def plotFillBetweenStd(x, y, l='', linetypes = ['-o','--','--'], axis = 0):
    global palletteCounter, pallette
    
    elementWiseMax = lambda a,b : np.array([max(ai,b) for ai in a])
    tol = 1.0e-6
    mean = np.mean(y, axis = axis) 
    std = np.std(y, axis = axis)
    
    
    color = pallette[palletteCounter]
    plt.plot(x, mean, linetypes[0], label = l, color = color)
    plt.fill_between( x,  elementWiseMax(mean - std, tol) , 
                           mean + std, facecolor = color , alpha = 0.3)

    palletteCounter += 1