import numpy as np
from mc_const import *
import matplotlib.pyplot as plt

nphs = np.heaviside

rng = np.random.default_rng(12345)

m_vf = np.zeros((Ns,ITER+1))
m_vf[:,-1] = PHIT


def r_vdot(r_v):
	dv = r_v[0] - r_v[1]
	vdot1 = MU1*(dv) - PHI[0] + 0.25*((B1*(dv))**2)*nphs(dv,0)
	dv = -dv
	vdot2 = MU2*(dv) - PHI[1] + 0.25*((B2*(dv))**2)*nphs(dv,0)

	return np.array([vdot1,vdot2])

for j in range(ITER,0,-1):
	m_vf[:,j-1] = m_vf[:,j] - r_vdot(m_vf[:,j])*STEP


r_X = np.zeros(ITER+1)
r_dZ = np.zeros(ITER+1)
r_X[0] = int(rng.choice(2, 1, p=PROB0))
#r_dZ[0] = r_X[0]*H[0] + (1-r_X[0])*H[1]

#r_dw = np.random.normal(0,W*STEP,ITER+1)
r_dw = rng.normal(0,W*STEP,ITER+1)

sim_iter = int(0.0)

Xnow = r_X[0]

while (sim_iter <= ITER):
	hold_param = lambda1*Xnow + lambda2*(1-Xnow) 
	hold_time = rng.exponential(1/hold_param)
	hold_iter = int(hold_time*ITER/T)
	del_iter = int(0.0) # in terms of number of iterations
	
	while ((del_iter <= hold_iter) and (sim_iter + del_iter <= ITER) ):
		r_X[sim_iter] = Xnow
		r_dZ[sim_iter] = (Xnow*H[0] + (1.0-Xnow)*H[1])*STEP + r_dw[sim_iter]
		del_iter += 1
		sim_iter += 1

	Xnow = 1 - Xnow

m_fpr = np.zeros((Ns,ITER+1))
m_fpr[:,0] = PROB0


for j in range(0,ITER):
	Hhat = np.dot(H,m_fpr[:,j])
	m_fpr[:,j+1] = np.dot(A,m_fpr[:,j])*STEP + np.dot((Hd - Hhat*np.eye(Ns)),m_fpr[:,j])*(r_dZ[j+1] - Hhat*STEP)


m_X = np.zeros((Np,ITER+1))
m_Xe = np.zeros((ITER+1,Ns,Np))
# r_dZ = np.zeros(ITER+1)
m_X[:,0] = int(rng.choice(2, Np, p=PROB0))
#r_dZ[0] = r_X[0]*H[0] + (1-r_X[0])*H[1]

#r_dw = np.random.normal(0,W*STEP,ITER+1)
r_dw = rng.normal(0,W*STEP,ITER+1)

m_pN = np.zeros((Ns,ITER+1))
for i in range(0,Ns):
	m_pN[i,0] = np.count_nonzero(m_X[:,0]==i)

m_Tmodel = np.zeros((Np,Ns)) # Ns for Ns = 2 only, otherwise it needs Np(Np-1)
m_Tmodelj = np.zeros((Np,Ns))
m_dNmodel = np.zeros((Np,Ns))
m_Tcon = np.zeros((Np,Ns))
m_Tconj = np.zeros((Np,Ns))
m_dNcon = np.zeros((Np,Ns))
for i in range(0,Np):
	for j in range(0,Ns):
		m_Tmodelj[i,j] = rng.exponential(1.0)
		m_Tmodelj[i,j] = rng.exponential(1.0)

for iter in range(0,ITER):

	m_Tmodel[:,0] = m_Tmodel[:,0] + lambda12*STEP
	m_Tmodel[:,1] = m_Tmodel[:,1] + lambda21*STEP

	ind12 = np.argwhere(m_Tmodel[:,0] > m_Tmodelj[:,0])
	ind21 = np.argwhere(m_Tmodel[:,1] > m_Tmodelj[:,1])

	for index in ind12:
		m_Tmodel[index,0] = 0.0
		m_Tmodelj[index,0] = rng.exponential(1.0)
		m_dNmodel[index,0] = int(1)

	for index in ind21:
		m_Tmodel[index,1] = 0.0
		m_Tmodelj[index,1] = rng.exponential(1.0)
		m_dNmodel[index,1] = int(1)

	m_dX1 = np.dot(G[0,1,:,:],np.dot(m_Xe[iter,:,:],np.diag(m_dNmodel[:,0])))
	m_dX1 = mdX1 + np.dot(G[1,0,:,:],np.dot(m_Xe[iter,:,:],np.diag(m_dNmodel[:,1])))

	m_Xe[:,iter+1] = m_Xe[:,iter] + m_dX1 

	m_Tmodel[:,0] = m_Tmodel[:,0] + lambda12*STEP
	m_Tmodel[:,1] = m_Tmodel[:,1] + lambda21*STEP

Xnow = r_X[0]


plt.figure(1)
plt.subplot(111)
plt.plot(np.arange(ITER+1), r_X, 'bx')
#plt.subplot(212)
#plt.plot(t, 2*s1)
plt.show()

plt.figure(2)
plt.subplot(121)
plt.plot(np.arange(ITER+1), m_vf[0,:])
plt.subplot(122)
plt.plot(np.arange(ITER+1), m_vf[1,:])
#plt.subplot(212)
#plt.plot(t, 2*s1)
plt.show()

plt.figure(3)
plt.subplot(121)
plt.plot(np.arange(ITER+1), m_fpr[0,:])
plt.subplot(122)
plt.plot(np.arange(ITER+1), m_fpr[1,:])
#plt.subplot(212)
#plt.plot(t, 2*s1)
plt.show()