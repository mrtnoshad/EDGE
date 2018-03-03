# 
# EDGE: Scalable, optimum and linear time mutual information estimator based on the paper: https://arxiv.org/abs/1801.09125 

# EDGE Estimator for Shannon Mutual Information
# I = EDGE(X,Y): estimate of mutual information between X snd Y
# X is N * d_x and Y is N * d_Y data sets
# U is optional upper bound for the pointwise mutual information: I = EDGE(X,Y, U)

# This Version: Automatic epsilon for all dimensions, Ensemble Estimation with Optimized wights, 
#          Repeat Estimation for different random epsilons


# Examples:
	
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

# Abtract of the Theoretical Results:

The proposed low complexity estimator is based on a bipartite graph, referred to as dependence graph. The data points are mapped to the vertices of this graph using randomized Locality Sensitive Hashing (LSH). The vertex and edge weights are defined in terms of marginal and joint hash collisions. For a given set of hash parameters ϵ(1),…,ϵ(k), a base estimator is defined as a weighted average of the transformed edge weights. The proposed estimator, called the ensemble dependency graph estimator (EDGE), is obtained as a weighted average of the base estimators, where the weights are computed offline as the solution of a linear programming problem. EDGE achieves optimal computational complexity O(N), and can achieve the optimal parametric MSE rate of O(1/N) if the density is d times differentiable. To the best of our knowledge EDGE is the first non-parametric MI estimator that can achieve parametric MSE rates with linear time complexity.

