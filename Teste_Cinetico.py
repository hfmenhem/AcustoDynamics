Npar = 2

f1 = np.array(Npar*[[0.623]])
f2 = np.array(Npar*[[0.034]])
f = 10*(10**6) #[1/s]
c = 1480*(10**3) #[mm/s]
k = 2*np.pi*f/c #[1/mm]
a = np.array(Npar*[[0.1*np.pi/k]]) # [mm]
m = (a**3*(4*np.pi/3))*(1*10**3) # [g], densidade do ar vezes seu volume
m = np.array([[1000], [1]]) # [g]
rho = 998*(10**(-6)) #g/mm^3
#v0 = (np.sqrt(2)/10) * (10**3) # [mm/s] velocidade equivalente para uma energia de 10J
v0 = (50*(10**3))*k/(2*np.pi*f*rho) #Pressão = 50*(10**3) Pa = 50*(10**3) g/mm*s^2

h=(np.pi/2)/k

sim = Simulador(f1, f2, f, c, k, a, m, rho, v0, h)

r0 = np.array([[[0,0,0]], [[4/k,0,-3/k]]])
v0 = np.array([[[0,0,0]], [[0,0,0]]])

# r0 = np.array([[[0,0,(np.pi/4)/k]]])
# v0 = np.array([[[0.00,0,0]]])

dt = 0.01
tempo = 400
rs, vs, t = sim.Simular(r0, v0, dt, tempo)

plt.figure(dpi=300)
#plt.axes().set_aspect('equal')
for i in range(Npar):
    plt.plot(t, rs[i, :, 2], linestyle='', marker='.',markersize=2)

plt.ylabel("z [mm]")
plt.xlabel("t [s]")

plt.show()

plt.figure(dpi=300)
#plt.axes().set_aspect('equal')
for i in range(Npar):
    plt.plot(rs[i, :, 0], rs[i, :, 2], linestyle='', marker='.',markersize=2)

plt.xlabel("x [mm]")
plt.ylabel("z [mm]")
plt.show()
