from main import Simulador
import numpy as np
import matplotlib.pyplot as plt

#Este código testa se o programa calcula corretamente a força acústica primária. 
#Para isso, é usado a formula padrão do potencial de Gor'kov (Uteste), 
#que é derivado numericamente e plotado junto com a força calculada pelo programa, 
#inserindo apenas Pin, Gin e Hin na função "FGorKov".


f1 = np.array([[-0.08]])
f2 = np.array([[0.07]])

f = 2*(10**6) #[1/s]
c = 1500*(10**3) #[mm/s]
k = 2*np.pi*f/c #[1/mm]
a = np.array([[0.1/k]]) # [mm]
m = (a**3*(4*np.pi/3))*(1*10**3) # [g], densidade do ar vezes seu volume

rho = 1000*(10**(-6)) #g/mm^3
v0 = (np.sqrt(2)/10) * (10**3) # [mm/s] velocidade equivalente para uma energia de 10J
h=(np.pi/2)/k

sim = Simulador(f1, f2, f, c, a, m, rho, v0, h,0)



zs = np.linspace(0, 1.5, 300)
r = np.expand_dims([0,0,1]*np.expand_dims(zs, 1),1)

Pin = sim.PhiIn(r) #[mm^2/s]
GPin = sim.GradPhiIn(r)
HPin = sim.HPhiIn(r)
F = sim.FGorKov(Pin, GPin, HPin) #[uN]

def Uteste(p, gp):
    return np.pi * rho * (a[0,0]**3)*(((k**2)*f1[0,0]*((np.abs(p)**2))/3)-(f2[0,0]*((np.abs(np.linalg.norm(gp, axis=1))**2))/2))
    
Ut  = Uteste(Pin[:, 0], GPin[:, 0, :]) 
FUteste = -(Ut[1:]-Ut[:-1])/(r[1:, 0, 2]-r[:-1, 0, 2])

plt.figure(dpi=300)
plt.plot(r[:, 0, 2],F[:, 0, 2], linestyle='', marker='.',markersize=2,  label='F programa')
plt.plot(r[:, 0, 2],k*Uteste(Pin[:, 0], GPin[:, 0, :]), linestyle='', marker='.',markersize=2, label='U teste * k') #Foi multiplicado k para ficar na mesma unidade de F
plt.plot((r[:-1, 0, 2]+r[1:, 0, 2])/2,FUteste, linestyle='', marker='.',markersize=2, label='F de U teste')

plt.legend()
plt.show()

print(f'λ = {2*np.pi/k}')










