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

    teste=inds.flatten()
    nomesOrg = np.array(nomes)[inds.flatten()]
    cmappoincare = mpl.colormaps['viridis']
    
    ns = []
    for d in range(20):
    #for d in[17]:        
        m = 1
        display = nomesOrg[d*20*m:(d+1)*20*m]
        #display = nomesOrg
        colors = cmappoincare(np.linspace(0, 1, len(display)))
        
           
        manter = .02
        #manter = 1
        fig = plt.figure(dpi=300, figsize=(10,8))
        ax = fig.add_subplot(projection='3d')
        #ax.view_init(elev=5, azim=20)
        ax.view_init(elev=45, azim=-90)
        #ax.view_init(elev=90, azim=-90)
        for i, nome in enumerate(display):    
            with open(nome, 'rb') as dbfile:
                dado = pickle.load(dbfile)
                dbfile.close()
            r0 = dado['r0']
            rpic = dado['rpic']
            vpic = dado['vpic']
            #print(r0)
            ns.append(len(rpic[0,:]))
            descarten = int(len(rpic[0,:])*manter)
            
            ax.plot( rpic[0,-1*descarten:], rpic[1,-1*descarten:], vpic[1,-1*descarten:], '.', markersize=1, color = colors[i] )
            ax.plot(r0[0], r0[1],0, marker='+', color = colors[i])
            #ax.plot(rpic[1,0], vpic[1,0], rpic[0,0], marker='x', color = colors[i])
        
        ax.set_xlabel(r'$z_A$')
        ax.set_ylabel(r'$z_B$')
        ax.set_zlabel(r'$v_B$')
        
        ax.set_ylim(2.5, 5.2)
        ax.set_xlim(-1.2, 1.2)
        ax.set_zlim(-400, 400)
        
        plt.show()
    print(len(ns))
    print(np.max(ns))
    print(np.mean(ns))
    print(np.std(ns))
    print(np.min(ns))    
        


    
   

