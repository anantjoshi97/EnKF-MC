import numpy as np
import scipy.linalg as SLA
from mc_const import *
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['font.size'] = 16


nphs = np.heaviside

rng = np.random.default_rng(12345)

m_vf = np.zeros((ITER+1,Nst))
m_w = np.zeros((ITER+1,Nst))
m_lambda = np.zeros((ITER+1,Nst))
m_vf[-1,:] = PHIT # check
m_w[-1,:] = np.exp(-PHIT) # check
m_lambda[-1,:] = np.exp(-PHIT)/np.sum(np.exp(-PHIT)) # check

a_optconw = np.ones((ITER+1,Nst,Nst,Nst))
for i in range(0,Nst):
	a_optconw[-1,i,i,:] = (1/m_w[-1,i])*m_w[-1,:]

for iter in range(ITER,0,-1):
	m_st = SLA.expm((iter-ITER)*STEP*(0.5*Hd - A))
	m_w[iter-1,:] = np.dot(m_st,m_w[-1,:])
	m_lambda[iter-1,:] = m_w[iter-1,:]/(np.sum(m_w[iter-1,:]))
	#m_vf[iter-1,:] = -np.log(m_w[iter-1,:])
	for i in range(0,Nst):
		a_optconw[iter-1,i,i,:] = (1/m_w[iter-1,i])*m_w[iter-1,:]

PROB0 = np.exp(-PHIT)/np.sum(np.exp(-PHIT))
m_mu = np.zeros((ITER+1,Nst))
m_mu[0,:] = PROB0
a_optconmu = np.ones((ITER+1,Nst,Nst,Nst))
for i in range(0,Nst):
	a_optconmu[-1,i,i,:] = (1/m_mu[0,i])*m_mu[0,:]

for iter in range(0,ITER):
	#print(np.sum((np.dot(m_mu[iter,:],H))*m_mu[iter,:] - np.dot(Hd,m_mu[iter,:])))
	m_mu[iter+1,:] = m_mu[iter,:] + np.dot(A.T,m_mu[iter,:])*STEP + 0.5*STEP*((np.dot(m_mu[iter,:],H))*m_mu[iter,:] - np.dot(Hd,m_mu[iter,:]))
	#m_mu[iter+1,:] = m_mu[iter+1,:]/np.sum(m_mu[iter+1,:])
	for i in range(0,Nst):
		a_optconmu[ITER-iter-1,i,i,:] = (1/m_mu[iter+1,i])*m_mu[iter+1,:]

print(m_mu[0,:])



m_X = np.zeros((ITER+1,Np),dtype=int) # stores the integer corresponding to the state
m_Xe = np.zeros((ITER+1,Np,Nst)) # stores the basis vector corresponding to the state
m_X[0,:] = (rng.choice(Nst, Np, p=PROB0)).astype(int)
for i in range(0,Np):
	m_Xe[0,i,:] = ID[m_X[0,i],:]

#print(m_X[0,:])
#print(m_Xe[0,:,:])

#r_dw = rng.normal(0,W*STEP,ITER+1)

m_pN = np.zeros((ITER+1,Nst))
# for i in range(0,Nst):
# 	#print(np.count_nonzero(m_X[0,:] ==i))
# 	m_pN[0,i] = (1/Np)*(np.count_nonzero(m_X[0,:] ==i)) # check

#m_pN[0,:] = (1/Np)*(np.bincount(m_X[0,:],minlength=int(Nst))) 
m_pN[0,:] = (1/Np)*(np.sum(m_Xe[0,:,:],axis=0)) 
print(m_pN[0,:])

a_optconN = np.ones((ITER+1,Nst,Nst,Nst))
for i in range(0,Nst):
	a_optconN[-1,i,i,:] = (1/m_pN[0,i])*m_pN[0,:]

m_Tmodel = np.zeros((Np,Nst,Nst)) # i,j th element represents transition from i to j
m_dTmodel = np.zeros((Np,Nst,Nst))
#m_dNmodel = np.zeros((Np,Nst,Nst))
m_Tcon = np.zeros((Np,Nst))
m_dTcon = np.zeros((Np,Nst))
#m_dNcon = np.zeros((Np,Nst))

m_dX_model = np.zeros((Np,Nst))
m_dX_con = np.zeros((Np,Nst))

m_ind = np.zeros(Nst)

for k in range(0,Np):
	for i in range(0,Nst):
		m_Tcon[k,i] = rng.exponential(1.0)
		for j in range(0,Nst):
			if (i!=j):
				m_Tmodel[k,i,j] = rng.exponential(1.0)

for iter in range(0,ITER): # simplistic model
	m_delta_model = np.maximum(m_dTmodel - m_Tmodel,0.0)
	m_ind_model = np.argwhere(m_delta_model > 0.0)
	m_dX_model = 0*m_dX_model
	for i in range(0,m_ind_model.shape[0]):
		particle = m_ind_model[i,0]
		Xnow = m_Xe[iter,particle,:]
		m_dX_model[particle,:] += np.dot(G[m_ind_model[i,1],m_ind_model[i,2],:,:],Xnow)
		m_dTmodel[particle,m_ind_model[i,1],m_ind_model[i,2]] = 0.0
		m_Tmodel[particle,m_ind_model[i,1],m_ind_model[i,2]] = rng.exponential(1.0)

	m_delta_con = np.maximum(m_dTcon - m_Tcon,0.0)
	m_ind_con = np.argwhere(m_delta_con > 0.0)
	m_dX_con = 0*m_dX_con
	for i in range(0,m_ind_model.shape[0]):
		particle = m_ind_con[i,0]
		Xnow = m_Xe[iter,particle,:]
		m_dX_con[particle,:] += np.dot(B[m_ind_con[i,1],:,:],Xnow)
		m_dTcon[particle,m_ind_con[i,1]] = 0.0
		m_Tcon[particle,m_ind_con[i,1]] = rng.exponential(1.0) 

	m_dX = m_dX_con + m_dX_model

	m_Xe[iter+1,:,:] = m_Xe[iter,:,:] + m_dX

	m_pN[iter+1,:] = (1/Np)*(np.sum(m_Xe[iter+1,:,:],axis=0))
	m_dTmodel += AREP*STEP
	m_dTcon = m_dTcon + (STEP*np.dot(U,m_pN[iter,:]))
	
	for i in range(0,Nst):
		a_optconN[ITER-iter-1,i,i,:] = (1/m_pN[iter+1,i])*m_pN[iter+1,:] 




# plt.figure(2)
# plt.subplot(311)
# plt.plot(np.arange(ITER+1), m_mu[:,0],'k')
# plt.plot(np.arange(ITER+1), m_pN[:,0],'b')
# plt.subplot(312)
# plt.plot(np.arange(ITER+1), m_mu[:,1],'k')
# plt.plot(np.arange(ITER+1), m_pN[:,1],'b')
# plt.subplot(313)
# plt.plot(np.arange(ITER+1), np.sum(m_mu,axis=1),'k')
# plt.plot(np.arange(ITER+1), np.sum(m_pN,axis=1),'b')
# plt.show()

# plt.figure(3)
# plt.subplot(121)
# plt.plot(np.arange(ITER+1), m_fpr[0,:])
# plt.subplot(122)
# plt.plot(np.arange(ITER+1), m_fpr[1,:])
# #plt.subplot(212)
# #plt.plot(t, 2*s1)
# plt.show()

fig1 = plt.figure()
ax = plt.subplot(211)
plt.plot(np.arange(ITER+1), m_mu[:,0],'k',label="$\mu_t$")
plt.plot(np.arange(ITER+1), np.flip(m_lambda[:,0]),'--r',label="$\lambda_{T-t}$")
plt.plot(np.arange(ITER+1), m_pN[:,0],'b',label="$\hat{\mu}_t$")
ax.legend(ncol=3)
ax = plt.subplot(212)
plt.plot(np.arange(ITER+1), m_mu[:,1],'k')
plt.plot(np.arange(ITER+1), np.flip(m_lambda[:,1]),'--r')
plt.plot(np.arange(ITER+1), m_pN[:,1],'b')
ax.set_xlabel("Time (ms)")
#fig.suptitle("Convergence of 1000 particle \n mean field system to true probability")
plt.show()
#plt.savefig('results-mc-p.pdf')

fig2 = plt.figure()
ax = plt.subplot(211)
plt.plot(np.arange(ITER+1), a_optconw[:,0,0,1],'k',label="$u_t$")
plt.plot(np.arange(ITER+1), a_optconN[:,0,0,1],'b',label="$\hat{u}_t$")
ax.legend(ncol=2)
ax = plt.subplot(212)
plt.plot(np.arange(ITER+1), a_optconw[:,1,1,0],'k')
plt.plot(np.arange(ITER+1), a_optconN[:,1,1,0],'b')
ax.set_xlabel("Time (ms)")
#fig.suptitle("Convergence of 1000 particle \n mean field system to true probability")
plt.show()
#plt.savefig('results-mc-u.pdf')
