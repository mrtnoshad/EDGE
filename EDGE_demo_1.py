# Test EDGE
from EDGE_4_3_1 import EDGE
from sklearn.datasets import make_classification
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

###########################
# The estimator is in the following form:
#
# EDGE(X,Y,U=u,gamma=[gamma_X,gamma_Y], epsilon=[eps_X, eps_Y], hashing)
#
# Arguments: 
# X and Y are the data for which we compute MI: I(X,Y)
# U (optional) is an upper bound on the MI. It doesn't need to be accurate, but more accurate upper bound we set, faster convergence rates we get
# gamma=[gamma_X,gamma_Y] (optional) is the vector of soothness for X and Y. 
#		For example, if the data is discrete we set gamma close to 0, 
#		and if the data is continuous we set gamma close to 1 (or maybe higher if it is very smooth) 
# epsilon=[eps_X, eps_Y] (optional) is the vector of bandwidths for X and Y. If no epsilon is set, 
#		automatic bandwidths will be set.
# hashing (optional): possible arguments are 'p-stable' (default) which is a common type of LSH
#		or 'floor' which uses the simple floor function as hashing
#  
###########################
# Number of clusters
M=[2,4,8,16,32]
I=np.zeros(len(M))
i=0
for m in M:
	X, Y = make_classification(n_samples=2000,n_features=10, n_redundant=0, n_informative=10,
                           n_classes=m, class_sep=10,flip_y=0.01, n_clusters_per_class=1)

	I[i] = EDGE(X,Y,U=20,gamma=[1,0.001]) # Estimated Mutual Information between X and Y using EDGE method
	print('# cluster: m, I',m, I[i])
	i+=1

plt.subplot(222)
plt.plot(M,I,'--o')
#plt.title('Number of classes')
plt.xlabel('Number of classes')
plt.ylabel('MI')

#plt.show()
#a=input('stop')


############################
# Distance between clusters
D=[0.1, 0.5, 2, 10, 20]
I=np.zeros(len(D))
i=0
for d in D:
	X, Y = make_classification(n_samples=2000,n_features=10, n_redundant=0, n_informative=10,
                           n_classes=8, class_sep=d,flip_y=0.01, n_clusters_per_class=1)

	I[i] = EDGE(X,Y,U=10,gamma=[1,0.001]) # Estimated Mutual Information between X and Y using EDGE method
	print('D, I', d, I[i])
	i+=1

plt.subplot(221)
plt.plot(D,I,'--o')
plt.title('Separation of clusters')
plt.xlabel('distance between clusters')
plt.ylabel('MI')



###########################
# Percentage of random labels
F=[0.01, 0.05, 0.1, 0.15, 0.2]
I=np.zeros(len(F))
i=0
for f in F:
	X, Y = make_classification(n_samples=4000,n_features=100, n_redundant=0, n_informative=100,
                           n_classes=8, class_sep=20,flip_y=f, n_clusters_per_class=1)

	I[i] = EDGE(X,Y,U=20,gamma=[1,0.001]) # Estimated Mutual Information between X and Y using EDGE method
	print('f, I',f, I[i])
	i+=1

plt.subplot(223)
plt.plot(F,I,'--o')
#plt.title('Number of classes')
plt.xlabel('Percentage of random labels')
plt.ylabel('MI')

###########################
# Number of samples
N = [50,80, 500, 2000, 4000]
T = 20 # Number of times to repeat experiments 
I=np.zeros((T,len(N)))

for t in range(T):
	i=0
	for n in N:
		X, Y = make_classification(n_samples=n,n_features=40, n_redundant=0, n_informative=40,
	                           n_classes=4, class_sep=20,flip_y=0.01, n_clusters_per_class=1)

		I[t,i] = EDGE(X,Y,U=20,gamma=[1,0.001],hashing='p-stable', stochastic = True) # Estimated Mutual Information between X and Y using EDGE method
		print(I[t,i])
		i+=1

I_mean = np.mean(I,0)
plt.subplot(224)
plt.plot(N,I_mean,'--o')
#plt.title('Number of classes')
plt.xlabel('Number of Samples')
plt.ylabel('MI')

plt.suptitle('Continuous-discrete Dependency', fontsize=16)
plt.show()

