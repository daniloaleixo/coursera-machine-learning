# Support Vector Machines

## Large Margin Classification
### Optimization Objective
#### Alternative view of logistic regression
We're gonna start with logistic regression, and show how we can modify it a bit, and get what is essentially the support vector machine.

Let z = Θ<sup>T</sup>x.

In sigmoid function, we hope:

- If y = 1, we want h<sub>Θ</sub>(x) close to 1 and then Θ<sup>T</sup>x >> 0.
- If y = 0, we want h<sub>Θ</sub>(x) close to 0 and then Θ<sup>T</sup>x << 0.

If we look at the cost function of the logistic regression:
```
-1 *  (y * log h<sub>Θ</sub>(x) + (1 - y) log (1 - h<sub>Θ</sub>(x)))) 
# que é igual a:
y log sigmoid(Θ<sup>T</sup>x) - (1 - y) log (1 - Θ<sup>T</sup>x)
```

So each example (x, y) contributes a term like the expression above for the overall cost function.

For the SVM we are going to take this cost function and modify it a little bit:

![IMG](img/img1.png)

Now we have a flat portion and the straight line portion and this will give the SVM computational advantages to us.

We''do the same for y = 0:

![IMG](img/img2.png)

We'll name it cost<sub>1</sub>(z) when y = 1 and cost<sub>0</sub>(z) when y = 0.

Here's the const function for logistic regression:

![IMG](img/img3.png)

and replace it with our new cost functions, this will give us the **SVM hypothesis**:
![IMG](img/img4.png)

and h<sub>Θ</sub>(x) = 1 if Θ<sup>T</sup>x >> 0, or 0 otherwise.


### Large Margin Intuition
### Mathematics Behind Large Margin Classification

## Kernels

## SVMs in Practice