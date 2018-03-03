# EDGE Estimator for Shannon Mutual Information
# I = EDGE(X,Y): estimate of mutual information between X snd Y
# X is N * d_x and Y is N * d_Y data sets
# U is optional upper bound for the pointwise mutual information: I = EDGE(X,Y, U)

# Version: Automatic epsilon for all dimensions, Ensemble Estimation with Optimized wights, 
#          Repeat Estimation for different random epsilons

import numpy as np
import math
import pdb
import cvxpy as cvx # Need to install CVXPY package 

# Return the normalizing factor of epsilon regarding STD and some randomness
def gen_eps(X,Y):

	# Parameter: range of random epsilon coefficient:
	eps_l, eps_u = 0.7, 1.8
	
	dim_X , dim_Y  = X.shape[1], Y.shape[1]

	std_X = np.array([np.std(X[:,[i]]) for i in range(dim_X)])
	std_Y = np.array([np.std(Y[:,[i]]) for i in range(dim_Y)])

	# random coeffs 
	cf_X = np.random.rand(1,dim_X)*(eps_u - eps_l) + eps_l
	cf_Y = np.random.rand(1,dim_Y)*(eps_u - eps_l) + eps_l

	# Random Shifts
	b_X = 10.0*np.random.rand(1,dim_X)*std_X
	b_Y = 10.0*np.random.rand(1,dim_Y)*std_Y

	# Random espilons
	eps_X = std_X * cf_X
	eps_Y = std_Y * cf_Y

	return (eps_X,eps_Y,b_X,b_Y)
	

# Compute the number of collisions in each bucket
def compute_collision(X,Y,t_m,eps_X,eps_Y,b_X,b_Y,Ni_max):

	# Num of Samples
	N = X.shape[0]

	# Find dimensions
	dim_X , dim_Y  = X.shape[1], Y.shape[1]
	dim = dim_X + dim_Y

	# normalize epsilons
	eps_X = eps_X * 1.0*t_m / pow(N,1.0/(2*dim))
	eps_Y = eps_Y * 1.0*t_m / pow(N,1.0/(2*dim))

	# Compute X_tilde and Y_tilde, the mappings of X and Y using H_1
	X_t, Y_t = 1.0*(X+b_X)/eps_X, 1.0*(Y+b_Y)/eps_Y
	X_t, Y_t= X_t.astype(int), Y_t.astype(int)
	
	# Hash vectors as dictionaries
	CX, CY, CXY = {}, {}, {} 

	# Computing Collisions
	for i in range(N):

		# Convert to list
		X_l, Y_l = tuple(X_t[i].tolist()), tuple(Y_t[i].tolist())
		
		# X collisions
		if X_l in CX:
			CX[X_l] += 1
		else: 
			CX[X_l] = 1
			
		# Y collisions
		if Y_l in CY:
			CY[Y_l] += 1
		else: 
			CY[Y_l] = 1
		# XY collisions
		if (X_l,Y_l) in CXY:
			CXY[(X_l,Y_l)] += 1
		else: 
			CXY[(X_l,Y_l)] = 1

	f = max(CXY.values())/(Ni_max*pow(N,2.0/4))

	return (f, CX, CY, CXY)


def find_interval(X,Y, eps_X,eps_Y,b_X,b_Y,t_l, t_u, Ni_max = 1):

	# Num of Samples
	N = X.shape[0]

	# parameter: C_balance: Minimum ratio of number of distinct hashing (L_XY) with respect to max collision  
	C_balance_l, C_balance_u = 0.7 , 1.5

	# Find the appropriate interval
	f_l, f_u = 0, 3.0
	err = 0
	while  (f_l < C_balance_l)  or ( C_balance_u < f_u): 
		# If cannot find the right interval make error
		err +=1
		if err > 200:
			raise ValueError('Error: Correct interval cannot be found. Try modifying t_l and t_u', t_l,t_u)  

		t_m = (t_u+t_l)/2

		(f_m, CX, CY, CXY) = compute_collision(X,Y, t_m,eps_X,eps_Y,b_X,b_Y,Ni_max)
		
		not_in_interval = (f_l < C_balance_l) or (C_balance_u < f_u) 
		if f_m < 1 and not_in_interval:
			t_l=t_m
			f_l=f_m
		elif f_m > 1 and not_in_interval:
			t_u=t_m
			f_u=f_m

	return (t_l,t_u)

# Compute mutual information function given epsilons and radom shifts
def Compute_MI(X,Y,U,t,eps_X,eps_Y,b_X,b_Y,Ni_min,Ni_max):
	
	# Num of Samples
	N = X.shape[0]

	# Parameter: Lower bound for Ni and Mj for being counted:
	mini = Ni_min * pow(N,2.0/4)

	(f_m, CX, CY, CXY) = compute_collision(X,Y,t,eps_X,eps_Y,b_X,b_Y,Ni_max)

	# Computing Mutual Information Function
	I = 0
	N_c=0

	for e in CXY.keys():
		Ni, Mj, Nij = CX[e[0]], CY[e[1]], CXY[e]

		#if mini<Ni and mini<Mj: 
		if mini<Nij:
			I += Nij* math.log(max(min(1.0*Nij*N/(Ni*Mj), U),1.0/U),2)
			N_c+=Nij

	I = 1.0* I / N_c
	return I

# A single run of Mutual Information estimation using EDGE
def EDGE_single_run(X,Y,U=20):

	# parameter: Ni_min * sqrt(N) < N_i, M_i < Ni_max * sqrt(N)
	Ni_min = 0.2
	Ni_max = 1.0 

	# Find dimensions
	dim_X , dim_Y  = X.shape[1], Y.shape[1]
	dim = dim_X + dim_Y
	
	# Number of terms in ensemble method
	L = dim+1

	# Num of Samples
	N = X.shape[0]
	t_l,t_u = 0.1, 40
	
	# Use less number of samples for learning the interval
	N_t=1000
	if N>N_t:
		X_test,Y_test=X[:N_t],Y[:N_t]
	else:
		X_test,Y_test=X,Y

	# get eps and b:
	(eps_X,eps_Y,b_X,b_Y) = gen_eps(X_test,Y_test)

	# Find the appropriate Interval
	(t_l,t_u) = find_interval(X,Y, eps_X,eps_Y,b_X,b_Y,t_l, t_u, Ni_max)

	# Find a range of indices
	l = t_u - t_l
	c = 1.0*l / L
	
	T_U = np.linspace(t_l, t_u, L)

	if L**(1.0/(2*dim)) < 1.0*t_u/t_l:
		T = t_l* np.array(range(1,L+1))**(1.0/(2*dim))
	else:
		T = np.linspace(t_l, t_u, L)

	# Vector of weights
	W = compute_weights(L, dim, T, N)
	
	
	# Vector of MI
	MV = np.array([Compute_MI(X,Y,U,T[i],eps_X,eps_Y,b_X,b_Y,Ni_min,Ni_max) for i in range(L)])

	# Ensemble MI
	I = np.dot(MV,W.T)
	
	return I

##### Linear Program for Ensemble Estimation ####
def compute_weights(L, d, T, N):
	
	# Correct T
	T = 1.0*T/T[0]

	# Create optimization variables.
	cvx_eps = cvx.Variable()
	cvx_w = cvx.Variable(L)

	# Create constraints:
	constraints = [cvx.sum_entries(cvx_w)==1, cvx.pnorm(cvx_w, 2)- cvx_eps/2 < 0 ]
	for i in range(1,d+1):
		Tp = ((1.0*T/N)**(1.0*i/(2*d)))
		cvx_mult = cvx_w.T * Tp
		constraints.append(cvx.sum_entries(cvx_mult) - cvx_eps*2 < 0)
	
	# Form objective.
	obj = cvx.Minimize(cvx_eps)

	# Form and solve problem.
	prob = cvx.Problem(obj, constraints)
	prob.solve()  # Returns the optimal value.

	sol = np.array(cvx_w.value)

	return sol.T

#########
# EDGE Estimator of Mutual Information
def EDGE(X,Y,U=20):
	r = 5	# Repeat for LSH with different random parameters and Take mean: By increasing this parameter you get more accurate estimate
	I = np.zeros(r)
	for i in range(r):
		I[i] = EDGE_single_run(X,Y,U)
	return np.mean(I)


####################################
####################################


if __name__ == "__main__":
	
	# Independent Datasets
	X = np.random.rand(1000,2)
	Y = np.random.rand(1000,2)

	I = EDGE(X,Y) # Estimated Mutual Information between X and Y using EDGE method
	print ('Independent Datasets: ', I)

	# Dependent Datasets
	X = np.random.rand(1000,2)
	Y = X + np.random.rand(1000,2)

	I = EDGE(X,Y) # Estimated Mutual Information between X and Y using EDGE method
	print ('Dependent Datasets: ', I)

	# Stronger dependency between datasets
	X = np.random.rand(1000,2)
	Y = X + np.random.rand(1000,2)/4

	I = EDGE(X,Y) # Estimated Mutual Information between X and Y using EDGE method
	print ('Stronger dependency between datasets: ',I)

	# Large size independent datasets
	X = np.random.rand(5000,40)
	Y = X**2 + np.random.rand(5000,40)/2

	I = EDGE(X,Y) # Estimated Mutual Information between X and Y using EDGE method
	print ('Large size independent datasets: ', I)



