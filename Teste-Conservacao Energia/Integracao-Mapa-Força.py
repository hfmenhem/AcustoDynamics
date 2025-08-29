import sys, os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
parent_dir = os.path.abspath(os.path.join(parent_dir, ".."))
sys.path.append(parent_dir) #Para poder importar o main, que está em uma pasta anterior

from main import Simulador
import numpy as np
import pickle
import concurrent.futures
import time
from scipy.integrate import odeint
start_time = time.time()

def Simular(nome,r0):
    ts = np.arange(0, tsim, dt)
    
    sol = odeint(SimAc, [*r0, 0,0], ts, args=(g,))
    
    sol = np.transpose(sol)
    rs = sol[:2,:]
    vsf = sol[2:,:]
    
    salvar = {'r0': r0, 'rs': rs, 'vs': vsf, 't': ts}

    with open(nome, 'wb') as dbfile:
        pickle.dump(salvar, dbfile)
        dbfile.close()
    
    return nome

def SimAcArrasto(y, t, g, C):
    r = y[:2]
    v = np.array(y[2:])
    
    Fac = np.array([Fz0.ev(*r) ,Fz1.ev(*r)])
    
    Far = -6*np.pi*C*v
    A =( (Fac+Far)/m) + g
    
    y = [*v.flatten(), *A.flatten()]
    return y

def SimAc(y, t, g):
    r = y[:2]
    v = np.array(y[2:])
    
    Fac = np.array([Fz0.ev(*r) ,Fz1.ev(*r)])
    
    A =(Fac/m) + g
    
    y = [*v.flatten(), *A.flatten()]
    return y

f=40e3 #Hz
dicMeio = Simulador.ar(1)


Lamb=dicMeio["c"]/f
a = np.array([1,1]) # [mm]
rhoPol = (900*(10**-6)) #[g/mm^3] #Material: PP
cPol = 2740*(10**3) #[mm/s] 
m = (a**3*(4*np.pi/3))*rhoPol # [g], densidade do ar vezes seu volume

#g=-9.81e3
g=0

tsim = 1
dt = 1e-4
    
numeroSim='sg'
diretorio ='sg'


    
with open(f'{diretorio}\\força', 'rb') as dbfile:
    dado = pickle.load(dbfile)
    dbfile.close()

Fz0 = dado[0]
Fz1 = dado[1]
    
if __name__ == '__main__':
    #print(f'lambda = {Lamb:.2f} mm ')

    z0eq=[0, Lamb/2]
    #dzeq=[-.05, 0.0]
    # ampdzeq0=[-0.2, 0.0] # amplitude em relação ao ponto de equilíbrio da partícula 0
    # ampdzeq1=[0.0, 1.0] # amplitude em relação ao ponto de equilíbrio da partícula 1
    # ampdzeq0=[-1.0, 1.0] # amplitude em relação ao ponto de equilíbrio da partícula 0
    # ampdzeq1=[-1.0, 1.0] # amplitude em relação ao ponto de equilíbrio da partícula 1
    # ampdzeq0=[-1.5, -0.7] # amplitude em relação ao ponto de equilíbrio da partícula 0
    # ampdzeq1=[-1.2, 0.2] # amplitude em relação ao ponto de equilíbrio da partícula 1
    
    # ampdzeq0=[ -1.2, -0.8] # amplitude em relação ao ponto de equilíbrio da partícula 0
    # ampdzeq1=-1.0 # amplitude em relação ao ponto de equilíbrio da partícula 1
    ampdzeq0=-0.5 # amplitude em relação ao ponto de equilíbrio da partícula 0
    ampdzeq1=-0.6 # amplitude em relação ao ponto de equilíbrio da partícula 1
    
    Npts = 1

    if (type(ampdzeq1) is float or type(ampdzeq1) is int) and (type(ampdzeq0) is float or type(ampdzeq0) is int):
        dzs0 = np.array(ampdzeq0)
        dzs1 = np.array(ampdzeq1)
        ampdzeq1=[ampdzeq1,ampdzeq1]
        ampdzeq0=[ampdzeq0,ampdzeq0]
    elif type(ampdzeq0) is float or type(ampdzeq0) is int:
        dzs0 = np.array(ampdzeq0)
        dzs1 = np.linspace(ampdzeq1[0], ampdzeq1[1], Npts, endpoint=True)
        ampdzeq0=[ampdzeq0,ampdzeq0]
    elif type(ampdzeq1) is float or type(ampdzeq1) is int:
        dzs0 = np.linspace(ampdzeq0[0], ampdzeq0[1], Npts, endpoint=True)
        dzs1 = np.array(ampdzeq1)
        ampdzeq1=[ampdzeq1,ampdzeq1]
    else:
        dzs1 = np.linspace(ampdzeq1[0], ampdzeq1[1], Npts, endpoint=True)
        dzs0 = np.linspace(ampdzeq0[0], ampdzeq0[1], Npts, endpoint=True)
    dzs=np.reshape(np.transpose(np.meshgrid(dzs0, dzs1), (1,2,0)), (-1, 2))
    dzs=np.reshape(np.transpose(np.meshgrid(dzs0, dzs1), (1,2,0)), (-1, 2))
    
    
    #=====================Achar ponto de quilíbrio em 2 partículas=====================
    
    C = (dicMeio['dinvis']*1e3)
    
    tcar = m[0]/C#tempo característico devido à força de arrasto
    tscar = np.arange(0, tcar, dt)
    
    r0 = np.array(z0eq)
    v0 = np.zeros(len(z0eq))
    
    
    sol = odeint(SimAcArrasto, [*r0, *v0], tscar, args=(g,C))
    sol = np.transpose(sol)
    rs = sol[:2,:]
    vs = sol[2:,:]
    
    while not np.all(np.isclose(vs[:,-1],0)): #Simula até que a velocidade seja nula
        sol = odeint(SimAcArrasto, [*rs[:,-1], *vs[:,-1]], tscar, args=(g,C))
        sol = np.transpose(sol)
        rs = sol[:2,:]
        vs = sol[2:,:]
        print(vs[-1,:])
    
        
    req = rs[:,-1] #pontos de equilíbrio
    print(req)
    z0s = np.expand_dims(req, 0)+dzs #Posições iniciais de simulação
    
    
    #=====================Salvamento=====================
    

    arraysalvaValores=[
                  #['deslocamento em z', f'{dz:.1f} mm'],
                  ['pontos de equilíbrio, partículas simultâneas', ''],
                  ['z0', f'{req[0]:.4e} mm'],
                  ['z1', f'{req[1]:.4e} mm'],
                  ['amplitude de amostragem', ''],
                  ['Delta zmin0', f'{ampdzeq0[0]:.4e} mm'],
                  ['Delta zmax0', f'{ampdzeq0[1]:.4e} mm'],
                  ['Delta zmin1', f'{ampdzeq1[0]:.4e} mm'],
                  ['Delta zmax1', f'{ampdzeq1[1]:.4e} mm'],
                  ['número de pontos por lado', f'{Npts}'],
                  ['tempo de simulação', f'{tsim:.4e} s'],
                  ['discretização de simulação', f'{dt:.4e} s'],
                  ]
    
    
    
    np.savetxt(f'{diretorio}\\{numeroSim}-geral-mapa-força.txt', arraysalvaValores, fmt='%s')
    
    
    #=====================Religamento=====================
    
    nomes = [f'{diretorio}\\{numeroSim}-dado-{i}' for i in range(len(z0s))]
    
    with concurrent.futures.ProcessPoolExecutor() as executor:
        result = executor.map(Simular, nomes, z0s)
    
    for r in result:
        print(r)
        
    print(f'O código demorou {(time.time() - start_time)/60:.1f} min')
    









