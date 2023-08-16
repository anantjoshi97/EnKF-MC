import numpy as np

T = 40.0
STEP = 0.5*1e-2
ITER = int(T/STEP)


Nst = int(2) # no of states
MU1 = 0.8
MU2 = 0.8
A = np.array([[-MU1,MU1],[MU2,-MU2]])

PHIT = np.array([1.0,1.5])

H = 0.02*np.array([5.0,10.0])
Hd = np.diag(H)

Nst = int(3)
r12 = 0.9
r23 = 0.6
r13 = 0.8
A = np.array([[-(r12 + r13), r12, r13],[r12,-(r12 + r23), r23],[r13, r23, -(r13 + r23)]])

PHIT = np.array([1.0,1.5,3.4])

H = 0.02*np.array([5.0,10.0,3.0])
Hd = np.diag(H)


ALPHA = np.sum(H)
U = 0.5*(ALPHA*np.eye(Nst) - Hd)

Np = 5000 # number of particles

ID = np.eye(Nst)

AREP = np.repeat(A.reshape((1,Nst,Nst)),Np,axis=0)
#print(AREP)

# G  = np.zeros((Nst,Nst,Nst,Nst))
# for i in range(0,Nst):
# 	for j in range(0,Nst):
# 		if (j!=i):
# 			G[i,j,i,j] = 1		
# 	G[i,j,i,i] = -1


# B  = np.zeros((Nst,Nst,Nst))
# for i in range(0,Nst):
# 	for j in range(0,Nst):
# 		if (j!=i):
# 			B[i,j,j] = -1		
# 			B[i,i,j] = 1

