import numpy as np
import scipy.linalg as SLA
from mctest_const import *
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

# a_optconw = np.ones((ITER+1,Nst,Nst,Nst))
# for i in range(0,Nst):
# 	a_optconw[-1,i,i,:] = (1/m_w[-1,i])*m_w[-1,:]

print("eig=",np.linalg.eig(0*0.5*Hd - A))
for iter in range(ITER,0,-1):
	m_stm = SLA.expm((iter-ITER)*STEP*(0*0.5*Hd - A))
	m_w[iter-1,:] = np.dot(m_stm,m_w[-1,:])
	m_lambda[iter-1,:] = m_w[iter-1,:]/(np.sum(m_w[iter-1,:]))
	#m_vf[iter-1,:] = -np.log(m_w[iter-1,:])
	# for i in range(0,Nst):
	# 	a_optconw[iter-1,i,i,:] = (1/m_w[iter-1,i])*m_w[iter-1,:]

PROB0 = np.exp(-PHIT)/np.sum(np.exp(-PHIT))
m_mu = np.zeros((ITER+1,Nst))
m_mu[0,:] = PROB0
# a_optconmu = np.ones((ITER+1,Nst,Nst,Nst))
# for i in range(0,Nst):
# 	a_optconmu[-1,i,i,:] = (1/m_mu[0,i])*m_mu[0,:]

for iter in range(0,ITER):
	#print(np.sum((np.dot(m_mu[iter,:],H))*m_mu[iter,:] - np.dot(Hd,m_mu[iter,:])))
	m_mu[iter+1,:] = m_mu[iter,:] + np.dot(A.T,m_mu[iter,:])*STEP + 0*0.5*STEP*((np.dot(m_mu[iter,:],H))*m_mu[iter,:] - np.dot(Hd,m_mu[iter,:]))
	m_mu[iter+1,:] = m_mu[iter+1,:]/np.sum(m_mu[iter+1,:])
	# for i in range(0,Nst):
	# 	a_optconmu[ITER-iter-1,i,i,:] = (1/m_mu[iter+1,i])*m_mu[iter+1,:]

print("prob0=",m_mu[0,:])

m_X = np.zeros(Np,dtype=int) # stores the integer corresponding to the state
# m_Xplot = np.zeros((ITER+1,Np),dtype=int) # stores the integer corresponding to the state
#m_Xe = np.zeros((ITER+1,Nst,Np)) # stores the basis vector corresponding to the state
m_X = (rng.choice(Nst, Np, p=PROB0)).astype(int)
# m_Xplot[0,:] = m_X
# for i in range(0,Nst):
# 	m_Xe[0,:,i] = ID[m_X[i,0],:]

#print(m_X[0,:])
#print(m_Xe[0,:,:])

#r_dw = rng.normal(0,W*STEP,ITER+1)

m_pN = np.zeros(Nst)
m_pNplot = np.zeros((ITER+1,Nst))
# for i in range(0,Nst):
# 	#print(np.count_nonzero(m_X[0,:] ==i))
# 	m_pN[0,i] = (1/Np)*(np.count_nonzero(m_X[0,:] ==i)) # check

m_pNplot[0,:] = (1/Np)*(np.bincount(m_X,minlength=int(Nst))) 
m_pN = m_pNplot[0,:]
print("m pN=", m_pN)

# a_optconN = np.ones((ITER+1,Nst,Nst,Nst))
# for i in range(0,Nst):
# 	a_optconN[-1,i,i,:] = (1/m_pN[i])*m_pN

# m_Tmodel = np.zeros((Np,Nst,Nst)) # i,j th element represents transition from i to j
# m_dTmodel = np.zeros((Np,Nst,Nst))
# #m_dNmodel = np.zeros((Np,Nst,Nst))
# m_Tcon = np.zeros((Np,Nst))
# m_dTcon = np.zeros((Np,Nst))
# #m_dNcon = np.zeros((Np,Nst))

m_T = np.zeros((Np,Nst+1,Nst)) # """ for axis1 = 0 to Nst-1, for every particle, i,j th element represents 
# transition from i to j, and for axis1 = Nst, for every particle it represents the control """
m_rates = np.zeros((Np,Nst+1,Nst))
m_rates[:,0:Nst,:] = AREP
m_rates[:,Nst,:] = -1.0 #np.dot(U,m_pN)


# for k in range(0,Np):
# 	for i in range(0,Nst):
# 		m_Tcon[k,i] = rng.exponential(1.0)
# 		for j in range(0,Nst):
# 			if (i!=j):
# 				m_Tmodel[k,i,j] = rng.exponential(1.0)

m_T = rng.exponential(1.0, (Np,Nst+1,Nst))

sim_time = 0.0

while (sim_time < T):

	# m_pN = (1/Np)*(np.bincount(m_X,minlength=int(Nst)))
	# #print(m_pN)
	# m_rates[:,Nst,:] = np.dot(U,m_pN)

	m_dT2 = np.divide(m_T,m_rates)
	m_dT = np.where(m_dT2 >= 0.0, m_dT2, np.inf)
	ind_jump = np.unravel_index(np.argmin(m_dT, axis=None), m_dT.shape)
	#print(ind_jump)
	jump_time = m_dT[ind_jump]
	#print("jump time = ", jump_time)
	particle = ind_jump[0]

	t1 = int((sim_time/STEP))
	#print((np.minimum(sim_time + jump_time,T)/STEP))
	t2 = int((np.minimum(sim_time + jump_time,T)/STEP)) + 1
	
	# m_Xplot[t1:t2,:] = m_X
	m_pNplot[t1:t2,:] = m_pN

	if sim_time <= 0.05:
		print(m_pN, t1, t2, jump_time)

	if(ind_jump[1] == Nst):
		#m_X[particle] = ind_jump[2]
		print("control")
	elif(m_X[particle] == ind_jump[1]):
		m_X[particle] = ind_jump[2]
	else:
		pass
	m_T = m_T - m_rates*jump_time
	if (np.abs(m_T[ind_jump]) >= 1e-17):
		print(m_T[ind_jump])
	m_T[ind_jump] = rng.exponential(1.0)
	sim_time = sim_time + jump_time

	m_pN = (1/Np)*(np.bincount(m_X,minlength=int(Nst)))
	#print(m_pN)
	m_rates[:,Nst,:] = -1.0 #np.dot(U,m_pN)


# for iter in range(0,ITER): # simplistic model
# 	m_delta_model = np.maximum(m_dTmodel - m_Tmodel,0.0)
# 	m_delta_con = np.maximum(m_dTcon - m_Tcon,0.0)
# 	m_delta = np.concatenate((m_delta_model,np.expand_dims(m_delta_con,axis=2)),axis=2) # size (Np,Nst,Nst+1)
# 	#print(m_delta.shape)
# 	m_delta2D = np.reshape(m_delta,(Np,Nst*(Nst+1)))
# 	#print(m_delta2D.shape)
# 	r_ind2D = np.argmax(m_delta2D,axis=1)
# 	#print(r_ind2D.shape)
# 	r_indr, r_indc = np.unravel_index(r_ind2D,(Nst,Nst+1)) #m_delta.shape[1:3]) # r_indr[i], r_indc[i] is the index of maximum in array m_delta for ith particle 
# 	m_ind = np.argwhere(m_delta > 0.0)
# 	m_X[iter+1,:] = m_X[iter,:]
# 	for i in range(0,m_ind.shape[0]):
# 		#print(iter,"jump")
# 		particle = m_ind[i,0]
# 		Xnow = m_X[iter,particle]
# 		Xnew = r_indr[particle]
# 		if (r_indc[particle] == Nst): # control counter ticked			
# 			m_X[iter+1,particle] = Xnew
# 			m_dTcon[particle,Xnew] = 0.0
# 			m_Tcon[particle,Xnew] = rng.exponential(1.0) 
# 			#print(iter,"con",Xnow,Xnew)
# 		elif( Xnow == Xnew ):
# 			Xnew = r_indc[particle]
# 			m_X[iter+1,particle] = Xnew
# 			m_dTmodel[particle,Xnow,Xnew] = 0.0
# 			m_Tmodel[particle,Xnow,Xnew] = rng.exponential(1.0) 
# 			#print(iter,"model",Xnow,Xnew)
# 		else:			
# 			m_dTmodel[particle,r_indr[particle],r_indc[particle]] = 0.0
# 			m_Tmodel[particle,r_indr[particle],r_indc[particle]] = rng.exponential(1.0)
# 		#print(iter,"coutner reset")
# 	# for i in range(0,m_ind.shape[0]):
# 	# 	#print(iter,"jump")
# 	# 	particle = m_ind[i,0]
# 	# 	Xnow = m_X[iter,particle]
# 	# 	Xnew = r_indr[particle]
# 	# 	if (r_indc[particle] == Nst): # control counter ticked			
# 	# 		m_X[iter+1,particle] = Xnew
# 	# 		m_dTcon[particle,Xnew] = 0.0
# 	# 		m_Tcon[particle,Xnew] = rng.exponential(1.0) 
# 	# 		#print(iter,"con",Xnow,Xnew)
# 	# 	elif( Xnow == Xnew ):
# 	# 		Xnew = r_indc[particle]
# 	# 		m_X[iter+1,particle] = Xnew
# 	# 		m_dTmodel[particle,Xnow,Xnew] = 0.0
# 	# 		m_Tmodel[particle,Xnow,Xnew] = rng.exponential(1.0) 
# 	# 		#print(iter,"model",Xnow,Xnew)
# 	# 	else:
# 	# 		if (m_ind[i,2] == Nst):
# 	# 			m_dTcon[particle,m_ind[i,1]] = 0.0
# 	# 			m_Tcon[particle,m_ind[i,1]] = rng.exponential(1.0) 
# 	# 		else:
# 	# 			m_dTmodel[particle,m_ind[i,1],m_ind[i,2]] = 0.0
# 	# 			m_Tmodel[particle,m_ind[i,1],m_ind[i,2]] = rng.exponential(1.0)
# 	# 		#print(iter,"coutner reset")

# 	m_pN[iter+1,:] = (1/Np)*(np.bincount(m_X[iter+1,:],minlength=Nst))
# 	m_dTmodel += AREP*STEP
# 	m_dTcon = m_dTcon + (STEP*np.dot(U,m_pN[iter,:]))
	
# 	for i in range(0,Nst):
# 		a_optconN[ITER-iter-1,i,i,:] = (1/m_pN[iter+1,i])*m_pN[iter+1,:] 




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
plt.plot(np.arange(ITER+1), m_pNplot[:,0],'b',label="$\hat{\mu}_t$")
ax.legend(ncol=3)
ax = plt.subplot(212)
plt.plot(np.arange(ITER+1), m_mu[:,1],'k')
plt.plot(np.arange(ITER+1), np.flip(m_lambda[:,1]),'--r')
plt.plot(np.arange(ITER+1), m_pNplot[:,1],'b')
ax.set_xlabel("Time (ms)")
#fig.suptitle("Convergence of 1000 particle \n mean field system to true probability")
plt.show()
#plt.savefig('results-mc-p.pdf')

# fig2 = plt.figure()
# ax = plt.subplot(211)
# plt.plot(np.arange(ITER+1), a_optconw[:,0,0,1],'k',label="$u_t$")
# plt.plot(np.arange(ITER+1), a_optconN[:,0,0,1],'b',label="$\hat{u}_t$")
# ax.legend(ncol=2)
# ax = plt.subplot(212)
# plt.plot(np.arange(ITER+1), a_optconw[:,1,1,0],'k')
# plt.plot(np.arange(ITER+1), a_optconN[:,1,1,0],'b')
# ax.set_xlabel("Time (ms)")
#fig.suptitle("Convergence of 1000 particle \n mean field system to true probability")
plt.show()
#plt.savefig('results-mc-u.pdf')
