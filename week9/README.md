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
## Multivariate Gaussian Distribution (Optional)

# Recommender Systems
## Predicting Movie Ratings
## Collaborative Filtering
## Low Rank Matrix Factorization
