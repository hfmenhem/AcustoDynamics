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

g=-9.81e3

tsim = 100
dt = 1e-4
razaoSalvar = 1e1

rtol = 1.49012e-11
atol = 1.49012e-11

if rtol is None:
    rtol = 1.49012e-8 #Valor padrão usado pela biblioteca
if atol is None:
    atol = 1.49012e-8 #Valor padrão usado pela biblioteca
    

numerosim ='Sim-Longo-1'

forca = 'estacionaria'

    
with open(f'{forca}-força', 'rb') as dbfile:
    dado = pickle.load(dbfile)
    dbfile.close()

Fz0 = dado[0]
Fz1 = dado[1]
    

#print(f'lambda = {Lamb:.2f} mm ')

z0eq=[0, Lamb/2]

ponto = np.array([-1.1, -1.0]) # amplitude em relação ao ponto de equilíbrio das partículas (0, 1)



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

z0s = req+ponto #Posições iniciais de simulação


#=====================Salvamento=====================

if os.path.exists(f'{numerosim}'):
    versao = 2
    while(os.path.exists(f'{numerosim}-v{versao}')):
        versao+=1
    diretorio = f'{numerosim}-v{versao}'
else:
    diretorio = f'{numerosim}'
os.mkdir(diretorio)

arraysalvaValores=[
              #['deslocamento em z', f'{dz:.1f} mm'],
              ['pontos de equilíbrio, partículas simultâneas', ''],
              ['z0', f'{req[0]:.4e} mm'],
              ['z1', f'{req[1]:.4e} mm'],
              ['posição simulada', ''],
              ['z0', f'{ponto[0]:.4e} mm'],
              ['z0', f'{ponto[1]:.4e} mm'],
              ['tempo de simulação', f'{tsim:.4e} s'],
              ['discretização de simulação', f'{dt:.4e} s'],
              ['discretização de salvamento', f'{dt*razaoSalvar:.4e} s'],
              ['rtol', f'{rtol:.4e} s'],
              ['atol', f'{atol:.4e} s']
              ]


np.savetxt(f'{diretorio}\\geral-trajetorias.txt', arraysalvaValores, fmt='%s')
    
with open(f'{diretorio}\\resolucao', 'wb') as dbfile:
    pickle.dump({'rtol': rtol, 'atol': atol, 'dt': dt, 'tsim': tsim}, dbfile)
    dbfile.close()

#=====================Religamento=====================

ts = np.arange(0, tsim, dt)

tss = np.array_split(ts[1:], np.ceil(len(ts)/1e4))     

r0 = ponto
v0 = [0,0]
t0 = 0
rs=[[],[]]
vsf=[[],[]]
ts =[] 

print(len(tss))
teste= np.concatenate(( [t0],tss[0]))
for i, tsi in enumerate(tss):
    tsief = np.concatenate(( [t0],tsi))
    sol = odeint(SimAc, [*r0, *v0], tsief, args=(g,), rtol=rtol, atol=atol)

    sol = np.transpose(sol)
    filtro =( np.array(range(len(tsief)))%razaoSalvar==0)
    filtro[-1]=False#o último valor não salvo
    r0 = sol[:2,-1]
    v0 = sol[2:,-1]
    t0 = tsi[-1]
    rs = np.concatenate((rs, sol[:2,filtro]), axis=1)
    vss = np.concatenate((vsf, sol[2:,filtro]), axis=1)
    ts = np.concatenate((ts, tsief[filtro]))
    
    print(i+1)
salvar = {'r0': ponto, 'rs': rs, 'vs': vsf, 't': ts}

nome = f'{diretorio}\\dado-0'
with open(nome, 'wb') as dbfile:
    pickle.dump(salvar, dbfile)
    dbfile.close()
    
print(f'O código demorou {(time.time() - start_time)/60:.1f} min')
    









