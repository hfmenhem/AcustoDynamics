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
    
    pasta='PoincareGrid-con_dev2-RK8'
    pasta2='PoincareGrid-con_dev2'
    nomes =[]
    nomes2 =[]
    for x in os.listdir(pasta):
        if 'dado' in x:
            nomes.append(f'{pasta}\\{x}')
    for x in os.listdir(pasta2):
        if 'dado' in x:
            nomes2.append(f'{pasta2}\\{x}')
   
    
    n0 = 5
    n1 = 5
    
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
    nomesOrg2 = np.array(nomes2)[inds.flatten()]
    cmappoincare = mpl.colormaps['viridis']
    
    ns = []
    for d in range(n1):
    #for d in[17]:        
        m = 1
        display = nomesOrg[d*n0*m:(d+1)*m*n0]
        display2 = nomesOrg2[d*n0*m:(d+1)*m*n0]
        #display = nomesOrg
        colors = cmappoincare(np.linspace(0, 1, len(display)))
        
           
        manter = .05
        manter = 1
        fig = plt.figure(dpi=300, figsize=(10,8))
        ax = fig.add_subplot( 1,2,1, projection='3d')
        ax2= fig.add_subplot( 1,2,2, projection='3d')
        #ax.view_init(elev=5, azim=20)

        ax.view_init(elev=45, azim=-90)
        ax2.view_init(elev=45, azim=-90)

        for i, nome in enumerate(display):  
            nome2 = display2[i]
            with open(nome, 'rb') as dbfile:
                dado = pickle.load(dbfile)
                dbfile.close()
                
            with open(nome2, 'rb') as dbfile:
                dado2 = pickle.load(dbfile)
                dbfile.close()
                
            r0 = dado['r0']
            rpic = dado['rpic']
            vpic = dado['vpic']
            
            r02 = dado2['r0']
            rpic2 = dado2['rpic']
            vpic2 = dado2['vpic']
            
            #print(r0)
            ns.append(len(rpic[0,:]))
            descarten = int(len(rpic[0,:])*manter)
            
            ax.plot( rpic[0,-1*descarten:], rpic[1,-1*descarten:], vpic[1,-1*descarten:], '.', markersize=1, color = colors[i] )
            #ax.plot( rpic[0,:descarten], rpic[1,:descarten], vpic[1,:descarten], '.', markersize=1, color = colors[i] )
            ax.plot(r0[0], r0[1],0, marker='+', color = colors[i])
            #ax.plot(rpic[1,0], vpic[1,0], rpic[0,0], marker='x', color = colors[i])
            
            ax2.plot( rpic2[0,-1*descarten:], rpic2[1,-1*descarten:], vpic2[1,-1*descarten:], '.', markersize=1, color = colors[i] )
            #ax2.plot( rpic2[0,:descarten], rpic2[1,:descarten], vpic2[1,:descarten], '.', markersize=1, color = colors[i] )
            ax2.plot(r02[0], r02[1],0, marker='+', color = colors[i])
            
        ax.set_xlabel(r'$z_A$')
        ax.set_ylabel(r'$z_B$')
        ax.set_zlabel(r'$v_B$')
        
        # ax.set_ylim(2.5, 5.2)
        # ax.set_xlim(-1.2, 1.2)
        # ax.set_zlim(-400, 400)
        
        # ax2.set_ylim(2.5, 5.2)
        # ax2.set_xlim(-1.2, 1.2)
        # ax2.set_zlim(-400, 400)
        
        ax.set_xlim(-1.1,-.9)
        ax.set_ylim(3.2,3.4)
        ax.set_zlim(-30,10)
        
        ax2.set_xlim(-1.1,-.9)
        ax2.set_ylim(3.2,3.4)
        ax2.set_zlim(-30,10)
        
        plt.show()
    print(len(ns))
    print(np.max(ns))
    print(np.mean(ns))
    print(np.std(ns))
    print(np.min(ns))    
        


    
   

