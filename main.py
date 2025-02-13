import numpy as np
import matplotlib.pyplot as plt

def PhiIn(r):
    return (v0/k)*np.sin(k*(r[:,:, 2]-h))

def GradPhiIn(r):
    f = lambda a: [0,0, v0*np.cos(k*(a[2]-h))]
    return np.apply_along_axis(f, 2, r)

def HPhiIn(r):
    f =lambda a: [[0,0,0], [0,0,0], [0,0, k*v0*np.sin(k*(a[2]-h)) ]]
    return np.apply_along_axis(f, 2, r)

def PhiSc(R, pin, gpin):
    Rn = np.linalg.norm(R, axis= 2)
    coef1 = -np.transpose(f1*(a**3)*(k**2))/(3)
    coef2 = (-1j)*np.transpose(f2*(a**3))/(2)
    #np.einsum('ijk,ijk->ij', gpin, R) #= produto escalar element-wise do útlimo eixo
    resul = (np.e**(1j*k*Rn))*( (coef1*pin/Rn) + (coef2*np.einsum('ijk,ijk->ij', gpin, R)*(1j+k*Rn)/(Rn**3)))
    np.nan_to_num(resul, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    return resul

def GradPhiSc(R, pin, gpin):
    Rn = np.linalg.norm(R, axis= 2)
    coef1 = (-1j)*np.transpose(f1*(a**3)*(k**2))/(3)
    coef2 = (-1j)*np.transpose(f2*(a**3))/(2)
    # tt= np.einsum('ijk,ijk->ij', gpin, R) = produto escalar element-wise do útlimo eixo
    #
    part1=np.expand_dims(coef1*pin*(1j+(k*Rn))/(Rn**3), axis=2)*R
    part2=np.expand_dims(coef2*(1j+(k*Rn))/(Rn**3),axis=2)*gpin+np.expand_dims(coef2*((-3j -(3*k*Rn)+1j*((k*Rn)**2))/(Rn)**5)*np.einsum('ijk,ijk->ij', gpin, R), axis=2)*R

    resul = np.expand_dims(np.e**(1j*k*Rn), axis=2)*(part1 + part2)
    np.nan_to_num(resul, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    return resul

def HPhiSc(R, pin, gpin):
    Rn = np.linalg.norm(R, axis= 2)
    RxR = np.expand_dims(R, 2) * np.expand_dims(R, 3)
    I = np.expand_dims(np.identity(3, dtype=float), (0,1))
    Rgp = np.expand_dims(R, 2) * np.expand_dims(gpin, 3)
    
    coef1 = np.transpose(-f1*(a**3)*(k**2)/(3))
    coef2 = (-1j)*np.transpose(f2*(a**3))/(2)
    # tt= np.einsum('ijk,ijk->ij', gpin, R) = produto escalar element-wise do útlimo eixo
    
    part11 = np.expand_dims((3-(3j*k*Rn)-((k*Rn)**2))/(Rn**2), (2,3))*RxR 
    part12 = np.expand_dims(1j*(1j+(k*Rn)), (2,3))*I
    part1 = np.expand_dims(coef1*pin*(np.e**(1j*k*Rn))/(Rn**3), (2,3))*( part11 + part12)
    
    part211 = np.expand_dims((15J+(15*k*Rn) - (((k*Rn)**2)*(6j+(k*Rn))))/(Rn**2), (2,3))*RxR 
    part212 =  np.expand_dims((-3j-(3*k*Rn)-(1j*((k*Rn)**2))), (2,3))*I
    part21 = np.expand_dims(np.einsum('ijk,ijk->ij', gpin, R), (2,3)) * (part211 + part212)
    
    part22 =  np.expand_dims((-3+(3j*k*Rn)+((k*Rn)**2)), (2,3))*(Rgp+np.transpose(Rgp, axes=(0,1,3,2)))
    part2 = np.expand_dims(coef2*(np.e**(1j*k*Rn))/(Rn**5), (2,3))*( part21 + part22)
    
    resul = part1+part2
    np.nan_to_num(resul, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    return resul

def FGorKov(Pt, GPt, HPt):
    part1 =np.expand_dims(f1*2/3, 2)*np.real(np.expand_dims(np.conjugate(Pt), 2)*GPt)
    part2 =np.expand_dims(-f2/(k**2), 2)*np.real( np.einsum('abj,abjk->abk', np.conjugate(GPt), HPt))

    f = np.expand_dims(-1*(a**3)*np.pi*rho*(k**2), 2) *(part1+part2)
    return f

Npar = 1
frames = 10000
f1 = np.array(Npar*[[-0.08]])
f2 = np.array(Npar*[[0.07]])

f = 2*(10**6) #[1/s]
c = 1500*(10**3) #[mm/s]
k = 2*np.pi*f/c #[1/mm]
a = np.array(Npar*[[0.1/k]]) # [mm]
m = (a**3*(4*np.pi/3))*(1*10**3) # [g], densidade do ar vezes seu volume


dt = 0.1

rho = 1000*(10**(-6)) #g/mm^3
v0 = (np.sqrt(2)/10) * (10**3) # [mm/s] velocidade equivalente para uma energia de 10J
h=(np.pi/2)/k

#r = np.array([[[0,0,0]], [[1/3,0,2*np.pi/k/4.1]],[[2/3,0,2*np.pi/k/2]],[[1,0,0]]]) 
#v = np.array([[[0,0,0]], [[0,0,0]],[[0,0,0]],[[0,0,0]]]) 

r = np.array([[[0,0,0]]]) 
v = np.array([[[0.001,0,0.01]]])

rs = np.empty((Npar, frames, 3))
vs = np.empty((Npar, frames, 3))

rs[:,0,:] = r[:,0,:]
vs[:,0,:] = v[:,0,:]

# r = np.array([[[0,0,0]]]) 
# v = np.array([[[0,0,0]]]) 

#Inicio do loop
for t in range(frames-1):
    MR = r - np.transpose(r, (1,0,2))
    Pin = PhiIn(r) #[mm^2/s]
    GPin = GradPhiIn(r)
    HPin = HPhiIn(r)
    
    Psc = PhiSc(MR, np.transpose(Pin), np.transpose(GPin, axes=(1,0,2)) )
    GPsc = GradPhiSc(MR, np.transpose(Pin), np.transpose(GPin, axes=(1,0,2)))
    HPsc = HPhiSc(MR, np.transpose(Pin), np.transpose(GPin, axes=(1,0,2)))
    
    Pt  = np.sum(Psc , axis = 1,keepdims=True) + Pin
    GPt = np.sum(GPsc, axis = 1,keepdims=True) + GPin
    HPt = np.sum(HPsc, axis = 1,keepdims=True) + HPin
    
    F = FGorKov(Pt, GPt, HPt) #[uN]
    
    A = F/np.expand_dims(m, 2)
    
    dr = v*dt + A*(dt**2)/2
    dv = A*dt
    
    v = v+dv
    r = r+dr
    
    rs[:,t+1,:] = r[:,0,:]
    vs[:,t+1,:] = v[:,0,:]


plt.figure(dpi=300)
#plt.axes().set_aspect('equal')
for i in range(Npar):
    plt.plot(rs[i, :, 0], rs[i, :, 2], linestyle='', marker='.',markersize=2)
    
plt.show()
print(2*np.pi/k)











