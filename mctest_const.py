import numpy as np

T = 50.0
STEP = 1e-3
ITER = int(T/STEP)


Nst = int(2) # no of states
# MU1 = 10.0
# MU2 = 20.0 
MU1 = 0.03
MU2 = 0.08
A = np.array([[-MU1,MU1],[MU2,-MU2]])
# B1 = 1.0
# B2 = 1.2
PHIT = np.array([1.0,1.5])
PHI = np.array([0.06,0.08])
PROB0 = np.array([0.3,0.7])

# lambda1 = 1.0
# lambda2 = 1.5

H = 0.02*np.array([5.0,10.0])
Hd = np.diag(H)
W = 1.0
ALPHA = np.sum(H)
U = ALPHA*np.eye(Nst) - Hd

Np = 1000 # number of particles

G  = np.zeros((Nst,Nst,Nst,Nst))
for i in range(0,Nst):
	for j in range(0,Nst):
		if (j!=i):
			G[i,j,i,j] = 1		
	G[i,j,i,i] = -1


B  = np.zeros((Nst,Nst,Nst))
for i in range(0,Nst):
	for j in range(0,Nst):
		if (j!=i):
			B[i,j,j] = -1		
			B[i,i,j] = 1

ID = np.eye(Nst)

AREP = np.repeat(A.reshape((1,Nst,Nst)),Np,axis=0)
#print(AREP)