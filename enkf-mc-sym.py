import numpy as np
from mc_const import *
import matplotlib.pyplot as plt

nphs = np.heaviside

rng = np.random.default_rng(12345)

m_vf = np.zeros((ITER+1,Nst))
m_w = np.zeros((ITER+1,Nst))
m_lambda = np.zeros((ITER+1,Nst))
m_vf[-1,:] = PHIT # check
m_w[-1,:] = np.exp(PHIT) # check
m_lambda[-1,:] = np.exp(PHIT)/np.sum(np.exp(PHIT)) # check

m_mu = np.zeros((ITER+1,Nst))
m_mu[0,:] = PROB0

for iter in range(0,ITER):
	#print(np.sum((np.dot(m_mu[iter,:],H))*m_mu[iter,:] - np.dot(Hd,m_mu[iter,:])))
	m_mu[iter+1,:] = m_mu[iter,:] + np.dot(A.T,m_mu[iter,:])*STEP + 0.5*STEP*((np.dot(m_mu[iter,:],H))*m_mu[iter,:] - np.dot(Hd,m_mu[iter,:]))
	#m_mu[iter+1,:] = m_mu[iter+1,:]/np.sum(m_mu[iter+1,:])

print(m_mu[0,:])

m_X = np.zeros((ITER+1,Np),dtype=int) # stores the integer corresponding to the state
m_Xe = np.zeros((ITER+1,Nst,Np)) # stores the basis vector corresponding to the state
m_X[0,:] = (rng.choice(Nst, Np, p=PROB0)).astype(int)
for i in range(0,Nst):
	m_Xe[0,:,i] = ID[m_X[i,0],:]

#print(m_X[0,:])
#print(m_Xe[0,:,:])

#r_dw = rng.normal(0,W*STEP,ITER+1)

m_pN = np.zeros((ITER+1,Nst))
# for i in range(0,Nst):
# 	#print(np.count_nonzero(m_X[0,:] ==i))
# 	m_pN[0,i] = (1/Np)*(np.count_nonzero(m_X[0,:] ==i)) # check

m_pN[0,:] = (1/Np)*(np.bincount(m_X[0,:],minlength=int(Nst))) 
print(m_pN[0,:])

m_Tmodel = np.zeros((Np,Nst,Nst)) # i,j th element represents transition from i to j
m_dTmodel = np.zeros((Np,Nst,Nst))
#m_dNmodel = np.zeros((Np,Nst,Nst))
m_Tcon = np.zeros((Np,Nst))
m_dTcon = np.zeros((Np,Nst))
#m_dNcon = np.zeros((Np,Nst))


m_ind = np.zeros(Nst)

for k in range(0,Np):
	for i in range(0,Nst):
		m_Tcon[k,i] = rng.exponential(1.0)
		for j in range(0,Nst):
			if (i!=j):
				m_Tmodel[k,i,j] = rng.exponential(1.0)

#m_Tmodel = rng.exponential(1.0, (Np,Nst,Nst))

# for iter in range(0,ITER):
# 	for i in range(0,Np):
# 		#m_dNmodel[i,:,:] = 0.0*m_dNmodel[i,:,:]
# 		#m_dNcon[i,:] = 0.0*m_dNcon[i,:]
# 		#ind = np.argwhere(m_dTmodel[i,:,:] > m_Tmodel[i,:,:]) # ind is a matrix in which each row holds the row and column of m_dTmodel where it is larger than m_Tmodelj
# 		m_delta_model = np.maximum(m_dTmodel[i,:,:] - m_Tmodel[i,:,:],0.0)
# 		m_delta_con = np.maximum(m_dTcon[i,:] - m_Tcon[i,:],0.0)
# 		#m_delta = np.vstack((m_delta_model[m_X[iter,i],:],m_delta_con))
# 		m_delta = np.vstack((m_delta_model,m_delta_con))
# 		ind = np.unravel_index(np.argmax(m_delta, axis=None), m_delta.shape) # if ind[0] < Nst, need ind[1] = state for jump
# 		# ind_model = np.argmax(m_delta_model[m_X[i,iter],:])
# 		# ind_con = np.argmax(m_delta_con)		
# 		if (m_delta[ind] > 0.0):	
# 			if (np.count_nonzero(m_delta) == 1):
# 				if (ind[0] == m_X[iter,i]):
# 					m_X[iter+1,i] = ind[1]
# 					m_Tmodel[i,m_X[iter,i],ind[1]] = rng.exponential(1.0)
# 					m_dTmodel[i,m_X[iter,i],ind[1]] = 0.0
# 				elif (ind[0] == Nst):
# 					m_X[iter+1,i] = ind[1]
# 					m_Tcon[i,ind[1]] = rng.exponential(1.0)
# 					m_dTcon[i,ind[1]] = 0.0
# 					print("con")
# 				else:
# 					m_Tmodel[i,ind[0],ind[1]] = rng.exponential(1.0)
# 					m_dTmodel[i,ind[0],ind[1]] = 0.0
# 					m_X[iter+1,i] = m_X[iter,i]
# 			else:
# 				m_Xnow = m_X[iter,i]
# 				m_Xnext = m_X[iter,i]
# 				indf = np.argsort(m_delta,axis=None)
# 				counter = -1
# 				ind = np.unravel_index(indf[counter],m_delta.shape)
# 				while (m_delta[ind] > 0.0):
# 					if (ind[0] == m_Xnow):
# 						m_Xnext = ind[1]
# 						m_Tmodel[i,m_Xnow,ind[1]] = rng.exponential(1.0)
# 						m_dTmodel[i,m_Xnow,ind[1]] = 0.0
# 					elif (ind[0] == Nst):
# 						m_Xnext = ind[1]
# 						m_Tcon[i,ind[1]] = rng.exponential(1.0)
# 						m_dTcon[i,ind[1]] = 0.0
# 						print("con")
# 					else:
# 						m_Tmodel[i,ind[0],ind[1]] = rng.exponential(1.0)
# 						m_dTmodel[i,ind[0],ind[1]] = 0.0
# 						m_X[iter+1,i] = m_Xnow
# 					m_Xnow = m_Xnext
# 				m_X[iter+1,i] = m_Xnext
				
# 		else:
# 			m_X[iter+1,i] = m_X[iter,i]

# 		m_dTmodel[i,:,:] += A*STEP
# 		m_dTcon[i,:] += 0*np.dot(U,m_pN[iter,:])*STEP

# 	for i in range(0,Nst): 
# 		m_pN[iter+1,i] = (1/Np)*(np.count_nonzero(m_X[iter+1,:] ==i)) 


for iter in range(0,ITER): # simplistic model
	m_delta_model = np.maximum(m_dTmodel - m_Tmodel,0.0)
	m_delta_con = np.maximum(m_dTcon - m_Tcon,0.0)
	m_delta = np.concatenate((m_delta_model,np.expand_dims(m_delta_con,axis=2)),axis=2) # size (Np,Nst,Nst+1)
	#print(m_delta.shape)
	m_delta2D = np.reshape(m_delta,(Np,Nst*(Nst+1)))
	#print(m_delta2D.shape)
	r_ind2D = np.argmax(m_delta2D,axis=1)
	#print(r_ind2D.shape)
	r_indr, r_indc = np.unravel_index(r_ind2D,(Nst,Nst+1)) #m_delta.shape[1:3]) # r_indr[i], r_indc[i] is the index of maximum for ith particle 
	m_ind = np.argwhere(m_delta > 0.0)
	m_X[iter+1,:] = m_X[iter,:]
	for i in range(0,m_ind.shape[0]):
		#print(iter,"jump")
		particle = m_ind[i,0]
		Xnow = m_X[iter,particle]
		Xnew = r_indr[particle]
		if (r_indc[particle] == Nst): # control counter ticked			
			m_X[iter+1,particle] = Xnew
			m_dTcon[particle,Xnew] = 0.0
			m_Tcon[particle,Xnew] = rng.exponential(1.0) 
			#print(iter,"con",Xnow,Xnew)
		elif( Xnow == Xnew ):
			Xnew = r_indc[particle]
			m_X[iter+1,particle] = Xnew
			m_dTmodel[particle,Xnow,Xnew] = 0.0
			m_Tmodel[particle,Xnow,Xnew] = rng.exponential(1.0) 
			#print(iter,"model",Xnow,Xnew)
		else:
			if (m_ind[i,2] == Nst):
				m_dTcon[particle,m_ind[i,1]] = 0.0
				m_Tcon[particle,m_ind[i,1]] = rng.exponential(1.0) 
			else:
				m_dTmodel[particle,m_ind[i,1],m_ind[i,2]] = 0.0
				m_Tmodel[particle,m_ind[i,1],m_ind[i,2]] = rng.exponential(1.0)
			#print(iter,"coutner reset")

	m_dTmodel += AREP*STEP
	m_dTcon = m_dTcon + (STEP*np.dot(U,m_pN[iter,:]))
	m_pN[iter+1,:] = (1/Np)*(np.bincount(m_X[iter+1,:],minlength=Nst)) 




plt.figure(2)
plt.subplot(311)
plt.plot(np.arange(ITER+1), m_mu[:,0],'k')
plt.plot(np.arange(ITER+1), m_pN[:,0],'b')
plt.subplot(312)
plt.plot(np.arange(ITER+1), m_mu[:,1],'k')
plt.plot(np.arange(ITER+1), m_pN[:,1],'b')
plt.subplot(313)
plt.plot(np.arange(ITER+1), np.sum(m_mu,axis=1),'k')
plt.plot(np.arange(ITER+1), np.sum(m_pN,axis=1),'b')
plt.show()

# plt.figure(3)
# plt.subplot(121)
# plt.plot(np.arange(ITER+1), m_fpr[0,:])
# plt.subplot(122)
# plt.plot(np.arange(ITER+1), m_fpr[1,:])
# #plt.subplot(212)
# #plt.plot(t, 2*s1)
# plt.show()