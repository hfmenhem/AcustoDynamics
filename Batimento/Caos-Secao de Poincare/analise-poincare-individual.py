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
    
    
    #===Mapeamento geral
    # pasta='PoincareGrid4'
    # n0 = 20
    # n1 = 20
    # lista = [(2,0), (2,1),(3,2),#caos?
    #          (2,2), (7,5), (7,6),#parecido com alto batimento
    #          (7,4),(17,0), #dentro
    #           (17,1), (17,2),#alto batimento
    #           (16,3),(7,7)#+ próximo do equilibrio?
             
    #          ]
    
    # listacor = [18,17,16,
    #             13,14,15,
    #             2,3,
    #             8,9,
    #             4,5
    #             ]
    
    # zorderlist=[12,11,10, 
    #             10.5,7.8,7.5,
    #             9,8,
    #             7,6,
    #             5,4
    #     ] 
    # manter = [1-0, 1]
    
    # #===Mapeamento geral - próxima da possível separatriz
    # pasta='PoincareGrid4'
    # n0 = 20
    # n1 = 20
    # lista = [(2,0), (2,1),(3,2),#caos?
    #          (3,0),
    #          (4,0)
    #          ]            
             
    
    # listacor = [18,17,16,
    #             8,
    #             15
                
    #             ]
    
    # zorderlist=[12,11,10, 10,10
    #     ]
    # manter = [1-.05, 1]
    
    #===Região próxima da possível separatriz
    # pasta='PoincareGrid-con_dev'
    # n0 = 5
    # n1 = 5
    # lista = [ (4,0),(0,4),(4,2), #circulos
    #            (1,4), #direita
    #          (4,1),(3,0),(2,4),(3,3),(4,3),(1,2),(3,1),(2,0),#cima-baixo
    #          (3,2),#esquerda
             
    #          #(0,0),(0,3),
             
    #          ]
  

    # listacor = [9,10, 8,
    #             18,
    #             15,14,13,12,4,5,6,7,
    #             0,
         
    #             #0,1,
    #             ]
    
    # zorderlist= np.full(len(lista), 1)
    # manter = [1-.05, 1]
    
    # #===Região próxima da possível separatriz, V2
    pasta='PoincareGrid-con_dev2-RK8'
    n0 = 5
    n1 = 5
    lista = [ (4,0),(0,4),(4,2), #circulos
               (1,4), #direita
             (4,1),(3,0),(2,4),(3,3),(4,3),(1,2),(3,1),(2,0),#cima-baixo
             (3,2),#esquerda
             
             #(0,0),(0,3),
             
             ]
  

    listacor = [9,10, 8,
                18,
                15,14,13,12,4,5,6,7,
                0,
         
                #0,1,
                ]
    
    zorderlist= np.full(len(lista), 1)
    #manter = [1-.0125, 1]
    shift=5*0.1
    comprimento = 0.1
    manter = [shift, shift+comprimento]
    manter=[0,1]
    # #===Região próxima da possível separatriz - trajetorias que convergiram, V2
    # pasta='PoincareGrid-con_dev2-RK8'
    # pasta='PoincareGrid-con_dev2'
    
    # n0 = 5
    # n1 = 5
    # lista = [ (4,0),(0,4),(4,2), #circulos
        
             
    #          ]
  

    # listacor = [9,10, 8,
              
    #             ]
    
    # zorderlist= np.full(len(lista), 1)
    # #manter = [1-.0125, 1]
    # shift=0*0.1
    # comprimento = 0.1
    # manter = [shift, shift+comprimento]
    
    print(manter)
    #manter = [0.75, 1]
    
    nomes =[]
    for x in os.listdir(pasta):
        if 'dado' in x:
            nomes.append(f'{pasta}\\{x}')
    
   
    
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

    
    cmap = mpl.colormaps['tab20b']
    colors= cmap(np.linspace(0,1, 20))
    
    cmap2 = mpl.colormaps['viridis']
    colors2= cmap(np.linspace(0,1, 25))
    
    fig = plt.figure(dpi=300, figsize=(14,14))
    ax = fig.add_subplot(projection='3d')
    #ax.view_init(elev=90, azim=-90)
    ax.view_init(elev=15, azim=-120)

    ax.set_xlabel(r'$z_A$ [mm]')
    ax.set_ylabel(r'$z_B$ [mm]')
    ax.set_zlabel(r'$v_B$ [mm/s]')
    ax.set_title('Espaço de fase seccionado em $v_a=0$')
    ax.set_ylim(2.5, 5.2)
    ax.set_xlim(-1.2, 1.2)
    ax.set_zlim(-400, 400)
    
    # fig2 = plt.figure(dpi=300, figsize=(30,30))
    # ax2 = fig2.add_subplot()
    # ax2.set_xlabel(r'$z_A$ [mm]')
    # ax2.set_ylabel(r'$z_B$ [mm]')
    # ax2.set_title('Espaço de fase seccionado em $v_a=0$ projetado em $z_a \\times z_b$')
    # ax2.set_aspect(1)
    
    # fig3 = plt.figure(dpi=300, figsize=(20,20))
    # ax3 = fig3.add_subplot()
    # ax3.set_xlabel(r'$z_A$ [mm]')
    # ax3.set_ylabel(r'$v_B$ [mm/s]')
    # ax3.set_title('Espaço de fase seccionado em $v_a=0$ projetado em $z_a \\times v_b$')
    #ax2.set_aspect(1)
    
    
    for i, coord in enumerate(lista):
        ind = inds[tuple(coord)]
        nome = nomes[ind]

        with open(nome, 'rb') as dbfile:
            dado = pickle.load(dbfile)
            dbfile.close()
        r0 = dado['r0']
        rpic = dado['rpic']
        vpic = dado['vpic']

        manterLen = np.round(len(rpic[0,:])*np.array(manter)).astype(int)
        
        ax.plot( rpic[0,manterLen[0]:manterLen[1]], rpic[1,manterLen[0]:manterLen[1]], vpic[1,manterLen[0]:manterLen[1]], '.', markersize=1, color = colors[listacor[i]], zorder=zorderlist[i] )
        ax.plot(r0[0], r0[1],0, marker='+', color = colors[listacor[i]],zorder=zorderlist[i])
        ax.plot(rpic[1,0], vpic[1,0], rpic[0,0], marker='x', color = colors[i])
      
        # ax2.plot( rpic[0,manterLen[0]:manterLen[1]], rpic[1,manterLen[0]:manterLen[1]], '.', markersize=1, color = colors[listacor[i]], zorder=zorderlist[i] )
        # ax2.plot(r0[0], r0[1], marker='+', color = colors[listacor[i]],zorder=zorderlist[i], markersize=50)

        # ax3.plot( rpic[0,manterLen[0]:manterLen[1]], vpic[1,manterLen[0]:manterLen[1]], '.', markersize=1, color = colors[listacor[i]], zorder=zorderlist[i] )
        # ax3.plot(r0[0], 0, marker='+', color = colors[listacor[i]],zorder=zorderlist[i])


    ax.set_xlim(-1.1,-.9)
    ax.set_ylim(3.2,3.4)
    ax.set_zlim(-30,30)
    
    # ax2.set_xlim(-1.1,-.9)
    # ax2.set_ylim(3.2,3.4)
    
    # ax3.set_xlim(-1.1,-.9)
    # ax3.set_ylim(-30,30)
    
    plt.show()
        


        


    
   
