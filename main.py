import numpy as np

class Simulador:
    
    def __init__(self, f1, f2, f, c, k, a, m, rho, v0, h):   
        self.f1 = f1
        self.f2 = f2
        self.f = f
        self.c = c
        self.k = k
        self.a = a
        self.m = m
        self.rho = rho
        self.v0 = v0
        self.h = h
     
    
    def PhiIn(self,r):
        return (self.v0/self.k)*np.sin(self.k*(r[:,:, 2]-self.h))
    
    def GradPhiIn(self,r):
        f = lambda a: [0,0, self.v0*np.cos(self.k*(a[2]-self.h))]
        return np.apply_along_axis(f, 2, r)
    
    def HPhiIn(self,r):
        f =lambda a: [[0,0,0], [0,0,0], [0,0, -self.k*self.v0*np.sin(self.k*(a[2]-self.h)) ]]
        return np.apply_along_axis(f, 2, r)
    
    def PhiSc(self, R, pin, gpin):
        Rn = np.linalg.norm(R, axis= 2)
        coef1 = -np.transpose(self.f1*(self.a**3)*(self.k**2))/(3)
        coef2 = (-1j)*np.transpose(self.f2*(self.a**3))/(2)
        #np.einsum('ijk,ijk->ij', gpin, R) #= produto escalar element-wise do útlimo eixo
        resul = (np.e**(1j*self.k*Rn))*( (coef1*pin/Rn) + (coef2*np.einsum('ijk,ijk->ij', gpin, R)*(1j+self.k*Rn)/(Rn**3)))
        np.nan_to_num(resul, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        return resul
    
    def GradPhiSc(self, R, pin, gpin):
        Rn = np.linalg.norm(R, axis= 2)
        coef1 = (-1j)*np.transpose(self.f1*(self.a**3)*(self.k**2))/(3)
        coef2 = (-1j)*np.transpose(self.f2*(self.a**3))/(2)
        # tt= np.einsum('ijk,ijk->ij', gpin, R) = produto escalar element-wise do útlimo eixo
        #
        part1=np.expand_dims(coef1*pin*(1j+(self.k*Rn))/(Rn**3), axis=2)*R
        part2=np.expand_dims(coef2*(1j+(self.k*Rn))/(Rn**3),axis=2)*gpin+np.expand_dims(coef2*((-3j -(3*self.k*Rn)+1j*((self.k*Rn)**2))/(Rn)**5)*np.einsum('ijk,ijk->ij', gpin, R), axis=2)*R
    
        resul = np.expand_dims(np.e**(1j*self.k*Rn), axis=2)*(part1 + part2)
        np.nan_to_num(resul, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        return resul
    
    def HPhiSc(self, R, pin, gpin):
        Rn = np.linalg.norm(R, axis= 2)
        RxR = np.expand_dims(R, 2) * np.expand_dims(R, 3)
        I = np.expand_dims(np.identity(3, dtype=float), (0,1))
        Rgp = np.expand_dims(R, 2) * np.expand_dims(gpin, 3)
        
        coef1 = np.transpose(-self.f1*(self.a**3)*(self.k**2)/(3))
        coef2 = (-1j)*np.transpose(self.f2*(self.a**3))/(2)
        # tt= np.einsum('ijk,ijk->ij', gpin, R) = produto escalar element-wise do útlimo eixo
        
        part11 = np.expand_dims((3-(3j*self.k*Rn)-((self.k*Rn)**2))/(Rn**2), (2,3))*RxR 
        part12 = np.expand_dims(1j*(1j+(self.k*Rn)), (2,3))*I
        part1 = np.expand_dims(coef1*pin*(np.e**(1j*self.k*Rn))/(Rn**3), (2,3))*( part11 + part12)
        
        part211 = np.expand_dims((15J+(15*self.k*Rn) - (((self.k*Rn)**2)*(6j+(self.k*Rn))))/(Rn**2), (2,3))*RxR 
        part212 =  np.expand_dims((-3j-(3*self.k*Rn)-(1j*((self.k*Rn)**2))), (2,3))*I
        part21 = np.expand_dims(np.einsum('ijk,ijk->ij', gpin, R), (2,3)) * (part211 + part212)
        
        part22 =  np.expand_dims((-3+(3j*self.k*Rn)+((self.k*Rn)**2)), (2,3))*(Rgp+np.transpose(Rgp, axes=(0,1,3,2)))
        part2 = np.expand_dims(coef2*(np.e**(1j*self.k*Rn))/(Rn**5), (2,3))*( part21 + part22)
        
        resul = part1+part2
        np.nan_to_num(resul, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        return resul
    
    def FGorKov(self, Pt, GPt, HPt):
        part1 =np.expand_dims(self.f1*2/3, 2)*np.real(np.expand_dims(np.conjugate(Pt), 2)*GPt)
        part2 =np.expand_dims(-self.f2/(self.k**2), 2)*np.real( np.einsum('abj,abjk->abk', np.conjugate(GPt), HPt))
    
        f = np.expand_dims(-1*(self.a**3)*np.pi*self.rho*(self.k**2), 2) *(part1+part2)
        return f
    
    def Simular(self, r0, v0, dt, tempo):
        frames = int(tempo/dt)
        nPar = np.shape(v0)[0]
        r = r0
        v = v0

        rs = np.empty((nPar, frames, 3))
        vs = np.empty((nPar, frames, 3))

        rs[:,0,:] = r0[:,0,:]
        vs[:,0,:] = v0[:,0,:]

        #Inicio do loop
        for t in range(0, frames-1):
            MR = r - np.transpose(r, (1,0,2))
            Pin = self.PhiIn(r) #[mm^2/s]
            GPin = self.GradPhiIn(r)
            HPin = self.HPhiIn(r)
            
            Psc = self.PhiSc(MR, np.transpose(Pin), np.transpose(GPin, axes=(1,0,2)) )
            GPsc = self.GradPhiSc(MR, np.transpose(Pin), np.transpose(GPin, axes=(1,0,2)))
            HPsc = self.HPhiSc(MR, np.transpose(Pin), np.transpose(GPin, axes=(1,0,2)))
            
            Pt  = np.sum(Psc , axis = 1,keepdims=True) + Pin
            GPt = np.sum(GPsc, axis = 1,keepdims=True) + GPin
            HPt = np.sum(HPsc, axis = 1,keepdims=True) + HPin
            
            F = self.FGorKov(Pt, GPt, HPt) #[uN]
            
            A = F/np.expand_dims(self.m, 2)
            
            dr = v*dt + A*(dt**2)/2
            dv = A*dt
            
            v = v+dv
            r = r+dr
            
            rs[:,t+1,:] = r[:,0,:]
            vs[:,t+1,:] = v[:,0,:]
        
        return rs, vs, np.array(range(frames))*dt




























