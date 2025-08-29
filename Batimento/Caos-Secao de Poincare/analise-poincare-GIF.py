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
import matplotlib.animation as animation

import pickle
import concurrent.futures

if __name__ == '__main__':
    
    pasta='PoincareGrid3'
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
    fig = plt.figure(dpi=300, figsize=(10,8))
    fig = plt.figure(dpi=300, figsize=(10,8))
    ax = fig.add_subplot(projection='3d')
    #ax.view_init(elev=5, azim=20)
    ax.view_init(elev=5, azim=40)
    
    
    
    ax.set_xlabel(r'$z_B$')
    ax.set_ylabel(r'$v_B$')
    ax.set_zlabel(r'$z_A$')
    ax.set_xlim(2.5, 5.2)
    ax.set_ylim(-400, 400)
    ax.set_zlim(-1.2, 1.2)
    #plt.show()
    art=[[None]*20, [None]*20, [None]*20]
    def update(frame):
        display = nomesOrg[frame*20:(frame+1)*20]
        #display = nomesOrg
        colors = cmappoincare(np.linspace(0, 1, len(display)))
        
        #ax.cla() 
        manter = .02
        # art[0].remove()
        # art[1].remove()
        # art[2].remove()
        for i, nome in enumerate(display):    
            with open(nome, 'rb') as dbfile:
                dado = pickle.load(dbfile)
                dbfile.close()
            r0 = dado['r0']
            rpic = dado['rpic']
            vpic = dado['vpic']
            #print(r0)
            descarten = int(len(rpic[0,:])*manter)
            
            if frame == 0:
                art[0][i] = (ax.plot(rpic[1,-1*descarten:], vpic[1,-1*descarten:], rpic[0,-1*descarten:], '.', markersize=1, color = colors[i]) )
                art[1][i] = (ax.plot(r0[1],0, r0[0], marker='+', color = colors[i]))
                art[2][i] = (ax.plot(rpic[1,0], vpic[1,0], rpic[0,0], marker='x', color = colors[i]))
            else:
                teste = art[0][i]
                teste2 = teste[0]
                teste2.set_data(rpic[1,-1*descarten:], vpic[1,-1*descarten:], rpic[0,-1*descarten:])
                art[0][i][0].set_data(rpic[1,-1*descarten:], vpic[1,-1*descarten:], rpic[0,-1*descarten:])
                art[1][i][0].set_data(r0[1],0, r0[0])
                art[2][i][0].set_data(rpic[1,0], vpic[1,0], rpic[0,0])
        print(frame)
        return art
        
    ani = animation.FuncAnimation(fig=fig, func=update, frames=20, interval=5)
    ani.save('teste.gif')
