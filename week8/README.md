

# Unsupervised Learning

## Clustering (K means algorithm)
### Unsupervised Learning Introduction
In unsupervised learning what we do is we give this sort of unlabeled training set to an algorithm and we just ask the algorithm find some structure in the data for us. One of the possible implementation for it is clustering.

Clustering is very helpful to a lot of things:
* Market Segmentation
* Social Network Analysis
* Organize Computing Clusters
* ...

### K Means Algorithm

In the clustering problem we are given an unlabeled data set and we would like to have an algorithm automatically group the data into coherent subsets or into coherent clusters for us. The K Means algorithm is by far the most popular and widely used clustering algorithm.

K Means is an iterative algorithm and it does two things. First is a **cluster assignment step**, and second is a **move centroid step**.

The first of the two steps in the loop of K means, is this cluster assignment step. What that means is that, it's going through each of the examples and it is going to assign each of the data points to one of the cluster centroids.

The other part of K means, in the loop of K means, is the move centroid step,  and we are going to move them to the average of the points belonging to the same centroid.

And we continually do this until we have very clear clusters.

![IMG](img/img1.png)

Inputs:

* K: number of clusters
* Training set {x<sup>(1)</sup>, x<sup>(2)</sup>, ... , x<sup>(m)</sup>}

x<sup>(i)</sup> E **R**<sup>*n*</sup>


#### The algorithm
Randomly initialize K cluster centroids μ<sub>1</sub>, μ<sub>2</sub>, ... , μ<sub>k</sub> E **R**<sup>*n*</sup>
Repeat:
  * For i = 1:m
    * c<sup>(i)</sup> = index (from 1 to K) of cluster centroid closest to x<sup>(i)</sup>
   * for k = 1:K
     * μ<sub>k</sub> = average (mean) of points assigned to cluster *k*

### Optimization Objective

We have to know the optimzation objective to first, knowing what is the optimization objective of k-means will help us to debug the learning algorithm and just make sure that k-means is running correctly. And second, and perhaps more importantly, how we can use this to help k-means find better costs for this and avoid the local optima.

Our variables:
* c<sup>(i)</sup> = index of clusters (1,2, ..., K) to which example x<sup>(i)</sup> is currently assigned
*  μ<sub>k</sub> = cluster centroid k (μ<sub>k</sub> E **R**<sup>*n*</sup>)
* μ<sub>c<sup>(i)</sup></sub> = cluster centroid of cluster to which example x<sup>(i)</sup> is currently assigned.

Optimization objective:

J(c<sup>(1)</sup>, ... , c<sup>(m)</sup>, μ<sub>1</sub>, ... , μ<sub>k</sub>) = (1/m) * Σ || x<sup>(i)</sup> - μ<sub>c<sup>(i)</sup></sub> ||<sup>2</sup>

### Random Initialization

When running K-means, you should have the number of cluster centroids, K, set to be less than the number of training examples M, then we would randomly pick k training examples. So, and, what I do is then set μ<sub>1</sub> of μ<sub>k</sub> equal to these k examples.

We can observe that we could end up converging to different solutions depending on exactly how the clusters were initialized, and so, depending on the random initialization K-means can end up at different solutions. And, in particular, K-means can actually end up at local optima.

	![IMG](img/img2.png)

So, instead of just initializing K-means once and hopping that that works, what we can do is, initialize K-means lots of times and run K-means lots of times, and use that to try to make sure we get as good a solution, as good a local or global optima as possible.

Concretely, here's how you could go about doing that. Let's say, I decide to run K-meanss a hundred times so I'll execute this loop a hundred times and it's fairly typical a number of times when came to will be something from 50 up to may be 1000.

### Choosing the Number of Clusters

There actually isn't a great way of answering this or doing this automatically and by far the most common way of choosing the number of clusters, is still choosing it manually by looking at visualizations or by looking at the output of the clustering algorithm or something else.

#### The elbow method
It is not a such good method, but we are going to plot the cost function and the K numbers of clusters in a chart and follow the descent of the cost function. Sometimes there is an elbow (a very large distortion between two values of K) and then we can know that for that K on we will have much smaller reduction in cost function J.

![IMG](img/img3.png)

It turns out the Elbow Method isn't used that often, and one reason is that, if you actually use this on a clustering problem, it turns out that fairly often, you know, you end up with a curve that looks much more ambiguous.

# Dimensionality Reduction
## Motivation
## Principal Component Analysis
## Applying PCA
