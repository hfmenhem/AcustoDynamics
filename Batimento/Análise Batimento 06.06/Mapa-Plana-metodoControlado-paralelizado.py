import sys, os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
parent_dir = os.path.abspath(os.path.join(parent_dir, ".."))
sys.path.append(parent_dir) #Para poder importar o main, que está em uma pasta anterior

from main import Simulador
import numpy as np
import pickle
import concurrent.futures
import time
start_time = time.time()

def operacao(nome,r0):
    rs, vsf, t = sim.Simular(r0, np.zeros(np.shape(r0)), dt, tsim, g=g)
    salvar = {'r0': r0, 'rs': rs, 'vs': vsf, 't': t}

    with open(nome, 'wb') as dbfile:
        pickle.dump(salvar, dbfile)
        dbfile.close()
    
    return nome

f=40e3 #Hz
dicMeio = Simulador.ar(1)

Npar=2
a = np.array(Npar*[[1]]) # [mm]
rhoPol = (900*(10**-6)) #[g/mm^3] #Material: PP
cPol = 2740*(10**3) #[mm/s] 
m = (a**3*(4*np.pi/3))*rhoPol # [g], densidade do ar vezes seu volume
f1 = 1- ((dicMeio['rho']*(dicMeio['c']**2))/ (rhoPol*(cPol**2)))
f2 = 2*((rhoPol-dicMeio['rho'])/((2*rhoPol)+dicMeio['rho']))
v0t = 10e3 #mm/s

Lamb=dicMeio["c"]/f

h=0
g=[0,0,-9.81e3]

tsim = 1
dt = 1e-4

sim = Simulador(np.array(Npar*[[f1]]), np.array(Npar*[[f2]]), f, dicMeio['c'], a, m, dicMeio['rho'], v0t, h, 0*dicMeio['dinvis'])



if __name__ == '__main__':
    #print(f'lambda = {Lamb:.2f} mm ')
    
    simeq = Simulador(np.array(Npar*[[f1]]), np.array(Npar*[[f2]]), f, dicMeio['c'], a, m, dicMeio['rho'], v0t, h, 1e3*dicMeio['dinvis'])

    z0eq=[0, Lamb/2]
    #dzeq=[-.05, 0.0]
    ampdzeq=[-1.0, 1.0]
    Npts = 100
    
    numeroSim='Sim3'
    dzs = np.linspace(ampdzeq[0], ampdzeq[1], Npts, endpoint=True)
    dzs=np.reshape(np.transpose(np.meshgrid(dzs, dzs), (1,2,0)), (-1, 2))
    
    
    #=====================Achar ponto de quilíbrio em 2 partículas=====================
    
    tcar = m[0,0]/(dicMeio['dinvis']*1e3)#tempo característico devido à força de arrasto
    
    r0 = np.array([[[0.0,0,z0eq[0]]],[[0.0,0,z0eq[1]]]])
    v0 = np.array([[[0.0,0,0]],[[0.0,0,0]]])
    
    rs, vs, t = simeq.Simular(r0, v0, dt, tcar, g=g)
    
    while not np.all(np.isclose(vs[:,-1, :],0)): #Simula até que a velocidade seja nula
        rs, vs, t = simeq.Simular(rs[:,-1,:], vs[:,-1,:], dt, tcar, g=g)
        print(vs[:,-1,:])
        
    req = rs[:,-1,:] #pontos de equilíbrio
    
    
    r0s = np.expand_dims(req, 0)+np.expand_dims(dzs,2)*np.expand_dims([0,0,1], (0,1))
    
    
    #=====================Salvamento=====================
    
    if os.path.exists(f'{numeroSim}'):
        versao = 2
        while(os.path.exists(f'{numeroSim}-v{versao}')):
            versao+=1
        diretorio = f'{numeroSim}-v{versao}'
    else:
        diretorio = f'{numeroSim}'
            
    
    os.mkdir(diretorio)
    arraysalvaValores=[
                  #['deslocamento em z', f'{dz:.1f} mm'],
                  ['Partículas de PP no ar', 'usando onda plana sem arrasto'],
                  ['Amplitude de velocidade da onda plana', f'{v0t:.4e} mm/s'],
                  ['f', f'{f:.4e} Hz'],
                  ['c', f'{dicMeio["c"]:.4e} Hz'],
                  ['rho', f'{dicMeio["rho"]:.4e} Hz'],
                  ['a, todas iguais', f'{a[0,0]:.4e} mm'],
                  ['m, todas iguais', f'{m[0,0]:.4e} mm'],
                  ['rhoPP', f'{rhoPol:.4e} g/mm^3'],
                  ['cPP', f'{cPol:.4e} mm/s'],
                  ['f1', f'{f1:.4e} mm'],
                  ['f2', f'{f2:.4e} mm'],
                  ['Nodos de pressão das ondas planas', ''],
                  ['z0', f'{z0eq[0]:.4e} mm'],
                  ['z1', f'{z0eq[1]:.4e} mm'],
                  ['pontos de equilíbrio, partículas simultâneas', ''],
                  ['z0', f'{req[0,2]:.4e} mm'],
                  ['z1', f'{req[1,2]:.4e} mm'],
                  ['amplitude de amostragem', ''],
                  ['Delta zmin', f'{ampdzeq[0]:.4e} mm'],
                  ['Delta zmax', f'{ampdzeq[1]:.4e} mm'],
                  ['número de pontos por lado', f'{Npts}'],
                  ['tempo de simulação', f'{tsim:.4e} s'],
                  ['discretização de simulação', f'{dt:.4e} s'],
                  ]
    
    
    
    np.savetxt(f'{diretorio}\\{numeroSim}-geral.txt', arraysalvaValores, fmt='%s')
    
    
    #=====================Religamento=====================
    
    nomes = [f'{diretorio}\\{numeroSim}-dado-{i}' for i in range(len(r0s))]
    
    with concurrent.futures.ProcessPoolExecutor() as executor:
        result = executor.map(operacao, nomes, r0s)
    
    for r in result:
        print(r)
        
    print(f'O código demorou {(time.time() - start_time)/60:.1f} min')
    







