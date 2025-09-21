import sys, os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
parent_dir = os.path.abspath(os.path.join(parent_dir, ".."))
sys.path.append(parent_dir) #Para poder importar o main, que está em uma pasta anterior

from main import Simulador
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.fft import fft, fftfreq
import scipy.signal as sig
from scipy.optimize import curve_fit
from scipy.interpolate import make_interp_spline

import pickle
import concurrent.futures

if __name__ == '__main__':
    
    pasta='PoincareGrid4'
    nomes =[]
    for x in os.listdir(pasta):
        if 'dado' in x:
            nomes.append(f'{pasta}\\{x}')
    
   
    
    n0 = 20
    n1 = 20
    
    r0s= []
    
    for nome in nomes:
        with open(nome, 'rb') as dbfile:
            dado = pickle.load(dbfile)
            dbfile.close()
        r0 = dado['r0']
        r0s.append(r0)
        
    r0s = np.array(r0s)
    inds = np.arange(len(r0s))

    arg = np.argsort(r0s[:,1])
    r0s = r0s[arg, :]
    inds = inds[arg]

    r0s = np.reshape(r0s, [n1,n0, 2])
    inds = np.reshape(inds, [n1,n0])

    arg2 = np.argsort(r0s[:,:,0], axis=1)
    r0s = np.take_along_axis(r0s, np.expand_dims(arg2, 2), axis=1)
    inds = np.take_along_axis(inds, arg2, axis=1)

    
    cmappoincare = mpl.colormaps['viridis']
    
    lista = [(2,0), (2,1),(3,2),#caos?
            
             
             ]


    
    #ax.view_init(elev=90, azim=-90)

    
    # ax.set_ylim(2.5, 5.2)
    # ax.set_xlim(-1.2, 1.2)
    # ax.set_zlim(-400, 400)
    
    manter = 0.05
    for i, coord in enumerate(lista):
        ind = inds[coord]
        nome = nomes[ind]
        fig = plt.figure(dpi=300, figsize=(10,8))
        ax = fig.add_subplot()

        with open(nome, 'rb') as dbfile:
            dado = pickle.load(dbfile)
            dbfile.close()
        r0 = dado['r0']
        rpic = dado['rpic']
        ts = dado['t']
        vpic = dado['vpic']

        descarten = int(len(rpic[0,:])*manter)
        
        ax.plot(range(len(rpic[0,:descarten])), rpic[0,:descarten], '.', markersize=1 )
        ax.plot(range(len(rpic[0,-1*descarten:])), rpic[0,-1*descarten:], '.', markersize=1 )
        
      
        #ax.plot(rpic[1,0], vpic[1,0], rpic[0,0], marker='x', color = colors[i])
    
        
        
        
        plt.show()
        


        


    
   

