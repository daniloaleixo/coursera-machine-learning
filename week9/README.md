

# Anomaly Detection
## Density Estimation
### Problem Motivation
![IMG](img/img1.png)
Let's say that on, the next day, you have a new aircraft engine that rolls off the assembly line and your new aircraft engine has some set of features x<sub>test</sub>. What the anomaly detection problem is, we want to know if this aircraft engine is anomalous in any way, in other words, we want to know if, maybe, this engine should undergo further testing because, or if it looks like an okay engine, and so it's okay to just ship it to a customer without further testing.

So we have a new example in which x<sub>1</sub> and x<sub>2</sub> are the features of this new example. If our test features are all the way out there, then we would call that an anomaly, and maybe send that aircraft engine for further testing before we ship it to a customer, since it looks very different than the rest of the aircraft engines we've seen before.

More formally in the anomaly detection problem, we're give some data sets, {x<sub>1</sub>, ..., X<sub>m</sub>} of examples, and we usually assume that these end examples are normal or non-anomalous examples, and we want an algorithm to tell us if some new example x<sub>test</sub> is anomalous. The approach that we're going to take is that given this training set, given the unlabeled training set, we're going to build a model for P(x).
And so, having built a model of the probability of x we're then going to say that for the new aircraft engine, if:

**P(x<sub>test</sub>) < epsilon**

then we flag this as an anomaly.

#### Applications
* Fraud detection
* Manufacturing
* Monitor computers in data center

### Gaussian Distribution
Also called the normal distribution. 
![IMG](img/img2.png)
And the Gaussian distribution is parametarized by two parameters, by a mean parameter which we denote µ and a variance parameter which we denote via σ<sup>2</sup>. If we plot the Gaussian distribution or Gaussian probability density. It'll look like the bell shaped curve which you may have seen before.

And so this bell shaped curve is parametrized by those two parameters, µ and σ<sup>2</sup>. And the location of the center of this bell shaped curve is the mean mu. And the width of this bell shaped curve, roughly that, is this parameter, sigma, is also called one standard deviation, and so this specifies the probability of x taking on different values.
The probability density of the normal distribution is:
![IMG](img/img3.png)

### Algorithm
Given the training set {x<sup>(1)</sup>, ... , x<sup>(m)</sup>}. Each example x ∈ **R**<sup>n</sup>.

P(x) = P(x<sub>1</sub>; µ<sub>1</sub>, σ<sub>1</sub><sup>2</sup>) * P(x<sub>2</sub>; µ<sub>2</sub>, σ<sub>2</sub><sup>2</sup>) * ... * P(x<sub>n</sub>; µ<sub>n</sub>, σ<sub>n</sub><sup>2</sup>)

We can prove that all features x<sub>k</sub>, k **c** [1, n] are independent.

![IMG](img/img4.png)


## Building an Anomaly Detection System
### Developing and Evaluating an Anomaly Detection System
Given the training cross validation and test sets, here's how you evaluate or here is how you develop and evaluate an algorithm.

Fit model P(x) on training set {x<sup>(1)</sup>, ... , x<sup>(m)</sup>}

On a cross-validation/test example x, predict: 

y = 1 if P(x) < ε (anomaly) or 0 if P(x) >= ε (normal) 

Possible evaluation metrics: 
* True positive, false positive, false negative, true negative
* Precision/recall
* F1-score
### Anomaly Detection vs. Supervised Learning 
#### Anomaly Detection
We use more anomaly detection, with:
* Very smal number of positive examples
* Large number of negative examples
* Many different types of anomalies. Hard for any algorithm to learn from positive examples what the anormalies look like
* Future anomalies may look nothing like any anomalous example so far
* Examples:
	* Fraud detection
	* Manufacturing
	* Monitoring machines in a data center

#### Supervised Learning
We use supervised learning with:
* Large number of positive examples
* Enough positive examples for algorithm to get a sense of what positive examples are like, future positive examples likely to be similar to ones in the training set
* Examples:
	* Email spam classifier
	* Weather prediction
	* Cancer classification
### Choosing What Features to Use 
We may want to test a few times, combine some features of transform features in order the algorithm better.

## Multivariate Gaussian Distribution (Optional)
### Multivariate Gaussian Distribution

In multivariate Gaussian instead of model P(x) = P(x<sub>1</sub>) * P(x<sub>2</sub>) * ... * P(x<sub>n</sub>) separately, we will model P(x) all in on go.

Paratemeters:
* µ ∈ **R**<sup>n</sup>, where µ = <sup>1</sup> / <sub>m</sub> * Σ<sub>i=1 to m</sub> x<sup>(i)</sup>
* Σ (covariance matrix), where Σ = <sup>1</sup> / <sub>m</sub> * Σ<sub>i=1 to m</sub> (x<sup>(i)</sup> - µ)(x<sup>(i)</sup> - µ)<sup>T</sup>

P(x; µ, Σ) = [ (2π)<sup><sup>n</sup>/<sub>2</sub></sup> |Σ|<sup><sup>1</sup>/<sub>2</sub></sup> ]<sup>-1</sup> * exp[ <sup>-1</sup>/<sub>2</sub> (x - µ)<sup>T</sup> Σ<sup>-1</sup> (x - µ) ]

### Anomaly Detection using the Multivariate Gaussian Distribution

And we flag an anomaly if P(x; µ, Σ) < ε

#### Original Model vs. Multivariate
##### Original Model
P(x<sub>1</sub>; µ<sub>1</sub>, σ<sub>1</sub><sup>2</sup>) * P(x<sub>2</sub>; µ<sub>2</sub>, σ<sub>2</sub><sup>2</sup>) * ... * P(x<sub>n</sub>; µ<sub>n</sub>, σ<sub>n</sub><sup>2</sup>)

* Manually creates features to capture anomalies where x<sub>1</sub>, x<sub>2</sub> take unusual combinations of values (e.g cpu load / memory)
* Computationaly cheaper (scales better)
* Okay even if *m*(training set size) is small 
##### Muiltivariate Gaussian
P(x; µ, Σ) = [ (2π)<sup><sup>n</sup>/<sub>2</sub></sup> |Σ|<sup><sup>1</sup>/<sub>2</sub></sup> ]<sup>-1</sup> * exp[ <sup>-1</sup>/<sub>2</sub> (x - µ)<sup>T</sup> Σ<sup>-1</sup> (x - µ) ]

* Automatically captures correlation between features
* Computationally more expensive
* Must have m > n or else Σ is non-invertible


# Recommender Systems
## Predicting Movie Ratings
### Problem Formulation
The idea of a recommender system is to give the best choice possible, given the movies you've liked. 

Notations: 
* n<sub>u</sub> = number of users
* n<sub>m</sub> = number of movies
* r(i, j) = 1 if the user *j* has rated movie *i*
* y(i, j) = rating given by user *j* to movie *i* (defined onlyl if r(i, j) =1 )

### Content Based Recommendations

We want to predict the movies that was not rated by the users, so each movie will have a set of features correspondent to it, giving information about genre, actors, etc.

For each user *j*, learn a parameter θ<sup>(j)</sup> ∈ **R**<sup>n</sup>. Predict user *j* as rating movie *i* with  (θ<sup>(j)</sup>)<sup>T</sup>x<sup>(i)</sup> stars.


Where θ<sup>(j)</sup> is the parameter vector for user *j* and m<sup>(y)</sup> the number of movies rated by user *j*.

#### Optimization objective

To learn θ<sup>(j)</sup> (parameter for user *j*):

min <sup>1</sup>/<sub>2</sub> * Σ<sub>i:r(i, j)=1</sub> [ (θ<sup>(j)</sup>)<sup>T</sup>x<sup>(i)</sup> - y<sup>(i, j)</sup> ]<sup>2</sup> + <sup>λ</sup>/<sub>2</sub> * Σ<sub>k=1..n</sub> [ θ<sub>k</sub><sup>(j)</sup> ]<sup>2</sup>

To learn θ<sup>(1)</sup>, θ<sup>(2)</sup>, ... , θ<sup>(n<sub>u</sub>)</sup>:

min <sup>1</sup>/<sub>2</sub> * Σ<sub>j=1..n<sub>u</sub></sub> Σ<sub>i:r(i, j)=1</sub> [ (θ<sup>(j)</sup>)<sup>T</sup>x<sup>(i)</sup> - y<sup>(i, j)</sup> ]<sup>2</sup> + <sup>λ</sup>/<sub>2</sub> * Σ<sub>j=1..n<sub>u</sub></sub>  Σ<sub>k=1..n</sub> [ θ<sub>k</sub><sup>(j)</sup> ]<sup>2</sup>

![IMG](img/img5.png)
 
## Collaborative Filtering
### Collaborative Filtering

![IMG](img/img6.png)

We have a list of features to each movie and the ratings for the user for each movie, collaborative filtering makes one user rating improves the whole algorithm.

Previously we viewed that given x<sup>(1)</sup>, ... , x<sup>(n<sub>m</sub>)</sup> we can learn θ<sup>(1)</sup>, ... , θ<sup>(n<sub>u</sub>)</sup>

So we can use  θ<sup>(1)</sup>, ... , θ<sup>(n<sub>u</sub>)</sup> to improve the features x<sup>(1)</sup>, ... , x<sup>(n<sub>m</sub>)</sup>.

And we can starting looping and improving thetas and x's at the same time.

### Algorithm
Minimizing x<sup>(1)</sup>, ... , x<sup>(n<sub>m</sub>)</sup> and θ<sup>(1)</sup>, ... , θ<sup>(n<sub>u</sub>)</sup> simultaneously:

J(x<sup>(1)</sup>, ... , x<sup>(n<sub>m</sub>)</sup>, θ<sup>(1)</sup>, ... , θ<sup>(n<sub>u</sub>)</sup>) = <sup>1</sup>/<sub>2</sub> * Σ<sub>j=1..n<sub>u</sub></sub> Σ<sub>i:r(i, j)=1</sub> [ (θ<sup>(j)</sup>)<sup>T</sup>x<sup>(i)</sup> - y<sup>(i, j)</sup> ]<sup>2</sup> + <sup>λ</sup>/<sub>2</sub> * Σ<sub>j=1..n<sub>u</sub></sub>  Σ<sub>k=1..n</sub> [ θ<sub>k</sub><sup>(j)</sup> ]<sup>2</sup>  + <sup>λ</sup>/<sub>2</sub> * Σ<sub>i=1..n<sub>m</sub></sub>  Σ<sub>k=1..n</sub> [ x<sub>k</sub><sup>(i)</sup> ]<sup>2</sup>

1. Initialize x<sup>(1)</sup>, ... , x<sup>(n<sub>m</sub>)</sup>, θ<sup>(1)</sup>, ... , θ<sup>(n<sub>u</sub>)</sup> to small random values.
2. Minimize J(x<sup>(1)</sup>, ... , x<sup>(n<sub>m</sub>)</sup>, θ<sup>(1)</sup>, ... , θ<sup>(n<sub>u</sub>)</sup>) using gradient descent (or an advanced optimization algorithm). E.g. for every j = 1, ... , n<sub>u</sub> and i = 1, ... , n<sub>m</sub>:

x<sub>k</sub><sup>(i)</sup> := x<sub>k</sub><sup>(i)</sup> - α [ Σ<sub>j:r(i, j)=1</sub> ( (θ<sup>(j)</sup>)<sup>T</sup>x<sup>(i)</sup> - y<sup>(i, j)</sup> ) θ<sub>k</sub><sup>(j)</sup> + λ x<sub>k</sub><sup>(i)</sup> ]

θ<sub>k</sub><sup>(j)</sup> := θ<sub>k</sub><sup>(j)</sup> - α [ Σ<sub>i:r(i, j)=1</sub> ( (θ<sup>(j)</sup>)<sup>T</sup>x<sup>(i)</sup> - y<sup>(i, j)</sup> ) θ<sub>k</sub><sup>(i)</sup> + λ x<sub>k</sub><sup>(j)</sup> ]

3. For a user with parameters θ and a movie with (learned) features x, predict a star rating of θ<sup>T</sup>x

## Low Rank Matrix Factorization
### Vectorization: Low Rank Matrix Factorization
![IMG](img/img7.png)

#### Finding related movies
We can find related movies by using the vector of features x<sup>(i)</sup>, if || x<sup>(i)</sup> - x<sup>(j)</sup> || is small we can conclude that movie *i* and *j* are related.

### Implementational Detail: Mean Normalization

To motivate the idea of mean normalization, let's consider an example of where there's a user that has not rated any movies. So the θ of the user will be all 0's and if we multiply by x(i) for any movie we will have predicted rating equal 0.

So to correct this implementation error we use mean normalization:

 (θ<sup>(j)</sup>)<sup>T</sup>x<sup>(i)</sup> + μ<sub>i</sub>
