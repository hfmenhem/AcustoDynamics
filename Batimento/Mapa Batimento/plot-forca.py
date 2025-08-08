import sys, os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir) #Para poder importar o main, que está em uma pasta anterior


from main import Simulador
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as ticker
from scipy.interpolate import RectBivariateSpline
import pickle

numeroSim='Sim8'
diretorio ='Sim8'

tipos = {'Sim8': "Onda plana", 'Sim9': "TinyLev"} 

    
with open(f'{diretorio}\\força', 'rb') as dbfile:
    dado = pickle.load(dbfile)
    dbfile.close()

Fz0 = dado[0]
Fz1 = dado[1]

zrange0 =[np.min(Fz0.get_knots()[0]),np.max(Fz0.get_knots()[0])]
zrange1 =[np.min(Fz0.get_knots()[1]),np.max(Fz0.get_knots()[1])]

Npts = 1000

z0 = np.linspace(*zrange0, Npts)
z1 = np.linspace(*zrange1, Npts)
grid = np.meshgrid(z0, z1)

Fz0eval = Fz0.ev(*grid)
Fz1eval = Fz1.ev(*grid)


fig, ax = plt.subplots(1,2,dpi=300, figsize=(13, 5))
fig.suptitle(f'Força acústica total no eixo z - {tipos[diretorio]}')

ax[0].set_title('partícula A')
ax[1].set_title('partícula B')
    
ax[0].set_aspect(1)
ax[1].set_aspect(1)

t1=np.min(np.where(np.isnan([Fz0eval, Fz1eval]),np.inf, [Fz0eval, Fz1eval]))
t2=np.max(np.where(np.isnan([Fz0eval, Fz1eval]),-np.inf, [Fz0eval, Fz1eval]))
norm = mpl.colors.Normalize(vmin=t1, vmax=t2, clip=False)

psm0 = ax[0].contourf(z0,z1, Fz0eval,shading='nearest', norm=norm, levels=20)
psm1 = ax[1].contourf(z0,z1, Fz1eval,shading='nearest', norm=norm, levels=20)

for i in range(2):
    ax[i].xaxis.set_major_locator(ticker.LinearLocator(3))
    ax[i].xaxis.set_minor_locator(ticker.LinearLocator(31))
    ax[i].yaxis.set_major_locator(ticker.LinearLocator(3))
    ax[i].yaxis.set_minor_locator(ticker.LinearLocator(31))
    
    ax[i].tick_params(axis="x", labelrotation=0)
    ax[i].set_xlabel('$z$ - partícula A [mm]')
  
ax[0].set_ylabel('$z$ - partícula B [mm]')
ax[1].tick_params(axis="y", which = 'both', labelleft=False, right=True)
ax[0].tick_params(axis="y", which = 'both', right=True)
fig.colorbar(psm1, ax=ax, location='right', label=r'$\hat z\cdot  Fsc z$ [$\mu N$]', format="{x:.1e}")

plt.show()
   