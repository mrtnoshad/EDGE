# Scalable Mutual Information Estimation using Dependence Graphs

EDGE is a scalable and optimum estimator of mutual information for deep learning applications


https://arxiv.org/abs/1801.09125 


## Abstract of the method

The Mutual Information (MI) is an often used measure of dependency between two random variables utilized in information theory, statistics and machine learning. Recently several MI estimators have been proposed that can achieve parametric MSE convergence rate. However, most of the previously proposed estimators have the high computational complexity of at least O(N2). We propose a unified method for empirical non-parametric estimation of general MI function between random vectors in ℝd based on N i.i.d. samples. The reduced complexity MI estimator, called the ensemble dependency graph estimator (EDGE), combines randomized locality sensitive hashing (LSH), dependency graphs, and ensemble bias-reduction methods. We prove that EDGE achieves optimal computational complexity O(N), and can achieve the optimal parametric MSE rate of O(1/N) if the density is d times differentiable. To the best of our knowledge EDGE is the first non-parametric MI estimator that can achieve parametric MSE rates with linear time complexity. We illustrate the utility of EDGE for the analysis of the information plane (IP) in deep learning. Using EDGE we shed light on a controversy on whether or not the compression property of information bottleneck (IB) in fact holds for ReLu and other rectification functions in deep neural networks (DNN).


### How to use this estimator?

The estimator is in the following form:

 
```
I = EDGE(X,Y,U=10, gamma=[1, 1], epsilon=[0,0], epsilon_vector = 'fixed', eps_range_factor=0.1, normalize_epsilon = True ,
				ensemble_estimation = 'optimal_weights', L_ensemble=10 ,hashing='p-stable', stochastic = True)


 X: is N * d_x and Y is N * d_Y data sets
 U: (optional) is an upper bound on the MI. It doesn't need to be accurate, but more accurate upper bound we set, faster 	convergence rates we get
 gamma=[gamma_X,gamma_Y]: (optional) is the vector of soothness for X and Y. 
		For example, if the data is discrete we set gamma close to 0, 
		and if the data is continuous we set gamma close to 1 (or maybe higher if it is very smooth) 
 epsilon=[eps_X, eps_Y]: (optional) is the vector of bandwidths for X and Y. If no epsilon is set, 
		automatic bandwidths according to KNN distances will be set.
 epsilon_vector (optional): possible arguments are 'fixed' or 'range'. If 'fixed' is given, all of 
		the bandwidths for the ensemble estimation will be the same, while, if 'range' is chosen, 
		the badwidths will be arithmetically increasing in a range.	 
 eps_range_factor (optional): If epsilon_vector == 'range', then the range of epsilon is 
		[epsilon, epsilon*(1+epsilon_vector)].
 normalize_epsilon: If it is True, then the badwidth will be normalized according to the MI estimate 
 ensemble_estimation: several options are available:
		'average': the ensemble estimator is the average of the base estimators
		'optimal_weights': the ensemble estimator is the wighted sum of the base estimators
							where the weights are computed using an optimization problem
							* You need to import cvxpy as cvx (install cvxpy if you do not have it)
		'median': the ensemble estimator is the median of the base estimators
 L_ensemble: number of different base estimators used in ensemble estimation. For more accurate estimates
				you can increase L_ensemble, but runtime increases linearly as well.
 hashing (optional): possible arguments are 'p-stable' (default) which is a common type of LSH
		or 'floor' which uses the simple floor function as hashing. For small dimensions, 'floor', a
		for higher dimensions, 'p-stable' are preferred.
 stochastic: it is stochastic, the hashing is generated using a random seed.
 
 Output: I is the estimation of mutual information between X snd Y 
```

### Examples


```
	
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
```


## Demo

In the demo file, several examples of continuous-discrete dependencies are provided. 


## License

This project is licensed under the University of Michigan License.

