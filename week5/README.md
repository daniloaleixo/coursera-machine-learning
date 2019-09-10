

  

# Neural Networks: Learning

  

  

## Cost Function and Backpropagation

  

  

### Const Function

  

  

Let's first define a few variables that we will need to use:

  

  

* L = total number of layers in the network

  

* s<sub>l</sub> = number of units (not counting bias unit) in layer l

  

* K = number of output units/classes

  

  

Recall that in neural networks, we may have many output nodes. We denote h<sub>Θ</sub>(x)<sub>k</sub> as being a hypothesis that results in the k<sup>th</sup> output. Our cost function for neural networks is going to be a generalization of the one we used for logistic regression. Recall that the cost function for regularized logistic regression was:

  

  

![IMG](img/img1.png)

  

  

For neural networks, it is going to be slightly more complicated:

  

  

![IMG](img/img2.png)

  

  

We have added a few nested summations to account for our multiple output nodes. In the first part of the equation, before the square brackets, we have an additional nested summation that loops through the number of output nodes.

  

  

In the regularization part, after the square brackets, we must account for multiple theta matrices. The number of columns in our current theta matrix is equal to the number of nodes in our current layer (including the bias unit). The number of rows in our current theta matrix is equal to the number of nodes in the next layer (excluding the bias unit). As before with logistic regression, we square every term.

  

  

Note:

  

  

* the double sum simply adds up the logistic regression costs calculated for each cell in the <b>output layer</b>

  

* the triple sum simply adds up the squares of all the individual Θs in the entire network.

  

* the i in the triple sum does not refer to training example i

  

  

### Backpropagation Algorithm

  

#### What is backpropagation?

  

Training a neural network typically happens in two phases.

1. Forward Pass: We compute the outputs of every node in the forward pass and calculate the final loss of the network.

2. Backward Pass: We start at the end of the network, backpropagate or feed the errors back, recursively apply chain rule to compute gradients all the way to the inputs of the network and then update the weights. This method of backpropagating the errors and computing the gradients is called backpropagation.

  

Backpropagation is a “local” process and can be viewed as a recursive application of the chain rule.

  

"Backpropagation" is neural-network terminology for minimizing our cost function, just like what we were doing with gradient descent in logistic and linear regression. Our goal is to compute:

  

  

min<sub>Θ</sub> J(Θ)

  

  

That is, we want to minimize our cost function J using an optimal set of parameters in theta. In this section we'll look at the equations we use to compute the partial derivative of J(Θ):

  

  

![IMG](img/img3.png)

  

  

To do so, we use the following algorithm:

  

  

![IMG](img/img4.png)

  

  

#### Back propagation Algorithm

  
  

  

Given training set {(x<sup>(1)</sup>,y<sup>(1)</sup>) ... (x<sup>(m)</sup>,y<sup>(m)</sup>)}

  

Set Δ<sub>i, j</sub><sup>(l)</sup> := 0 for all (l,i,j), (hence you end up having a matrix full of zeros)

  

For training example t =1 to m:

  

1. Set a<sup>(1)</sup> := x<sup>t</sup>

2. Perform forward propagation to compute a<sup><i>(l)</i></sup> for l=2,3,…,L

  

![IMG](img/img5.png)

  

3. Using y<sup>(t)</sup> compute δ<sup>(L)</sup>=a<sup>(L)</sup> − y<sup>(t)</sup>

  

Where L is our total number of layers and a<sup>(L)</sup> is the vector of outputs of the activation units for the last layer. So our "error values" for the last layer are simply the differences of our actual results in the last layer and the correct outputs in y. To get the delta values of the layers before the last layer, we can use an equation that steps us back from right to left:

  

4. Compute δ<sup>(L-1)</sup>, δ<sup>(L-2)</sup>, ... ,δ<sup>(2)</sup> using δ<sup>(t)</sup> =( (Θ<sup>(t)</sup>)<sup>T</sup> δ<sup>(<i> l</i>+1)</sup>) .∗ a<sup>(<i>l</i>)</sup> .∗ (1−a<sup>(<i>l</i>)</sup>)

  

The delta values of layer l are calculated by multiplying the delta values in the next layer with the theta matrix of layer l. We then element-wise multiply that with a function called g', or g-prime, which is the derivative of the activation function g evaluated with the input values given by z<sup>(<i>l</i>)</sup>.

The g-prime derivative terms can also be written out as:

  

![IMG](img/img6.png)

  

5. Δ<sub>i,j</sub><sup>(l)</sup> ​:= Δ<sub>i,j</sub><sup>(l)</sup> + a<sub>j</sub><sup>(l)</sup> ​δ<sub>i</sub><sup>(l+1)</sup>​ or with vectorization, Δ<sup>(l)</sup> := Δ<sup>(l)</sup>+δ<sup>(l+1)</sup> (a<sup>(l)</sup>)<sup><i>T</i></sup>

Hence we update our new \DeltaΔ matrix.

  

![IMG](img/img7.png)

  

The capital-delta matrix D is used as an "accumulator" to add up our values as we go along and eventually compute our partial derivative. Thus we get <sup>∂</sup> / <sub>∂Θ<sub>ij</sub><sup>(l)</sup></sub> J(Θ) = D<sub>ij</sub><sup>(l)​</sup>

  

### Backpropagation Intuition

  

Recall that the cost function for a neural network is:

  

![IMG](img/img8.png)

  

If we consider simple non-multiclass classification (k = 1) and disregard regularization, the cost is computed with:

  

![IMG](img/img9.png)

  

Intuitively, δ<sub>j</sub><sup>(l)</sup> is the "error" for a<sub>j</sub><sup>(l)</sup> (unit j in layer l). More formally, the delta values are actually the derivative of the cost function:

  

![IMG](img/img10.png)

  

Recall that our derivative is the slope of a line tangent to the cost function, so the steeper the slope the more incorrect we are. Let us consider the following neural network below and see how we could calculate some δ<sub>j</sub><sup>(l)</sup>

  

![IMG](img/img11.png)

  

In the image above, to calculate δ<sub>2</sub><sup>(2)</sup>, we multiply the weights Θ<sub>12</sub><sup>(2)</sup> and Θ<sub>22</sub><sup>(2)</sup> by their respective δ values found to the right of each edge. So we get δ<sub>2</sub><sup>(2)</sup> = Θ<sub>12</sub><sup>(2)</sup> * δ<sub>1</sub><sup>(3)</sup> + Θ<sub>22</sub><sup>(2)</sup> * δ<sub>2</sub><sup>(3)</sup>. To calculate every single possible δ<sub>j</sub><sup>(l)</sup>, we could start from the right of our diagram. We can think of our edges as our Θ<sub>ij</sub>. Going from right to left, to calculate the value of δ<sub>j</sub><sup>(l)</sup>, you can just take the over all sum of each weight times the δ it is coming from. Hence, another example would be δ<sub>2</sub><sup>(3)</sup> = Θ<sub>12</sub><sup>(3)</sup> * δ<sub>1</sub><sup>(4)</sup>.


##  Backpropagation in Practice

###  Implementation Note: Unrolling Parameters
With neural networks, we are working with sets of matrices:

* Θ<sup>(1)</sup>,Θ<sup>(2)</sup>,Θ<sup>(3)</sup>,…
* D<sup>(1)</sup>,D<sup>(2)</sup>,D<sup>(3)</sup>,…

In order to use optimizing functions such as "fminunc()", we will want to "unroll" all the elements and put them into one long vector:

```
thetaVector = [ Theta1(:); Theta2(:); Theta3(:); ]
deltaVector = [ D1(:); D2(:); D3(:) ]
```

If the dimensions of Theta1 is 10x11, Theta2 is 10x11 and Theta3 is 1x11, then we can get back our original matrices from the "unrolled" versions as follows:

```
Theta1 = reshape(thetaVector(1:110),10,11)
Theta2 = reshape(thetaVector(111:220),10,11)
Theta3 = reshape(thetaVector(221:231),1,11)
```

To summarize:

![IMG](img/img12.png)


##  Gradient Checking

Gradient checking will assure that our backpropagation works as intended. We can approximate the derivative of our cost function with:

![IMG](img/img13.png)

With multiple theta matrices, we can approximate the derivative **with respect to**  Θ<sub>j</sub> as follows:

![IMG](img/img14.png)

A small value for ϵ (epsilon) such as ϵ = 10<sup>−4</sup>, guarantees that the math works out properly. If the value for ϵ is too small, we can end up with numerical problems.

Hence, we are only adding or subtracting epsilon to the Θ<sub>j</sub> matrix. In octave we can do it as follows:

```
epsilon = 1e-4;
for i = 1:n,
  thetaPlus = theta;
  thetaPlus(i) += epsilon;
  thetaMinus = theta;
  thetaMinus(i) -= epsilon;
  gradApprox(i) = (J(thetaPlus) - J(thetaMinus))/(2*epsilon)
end;
```

We previously saw how to calculate the deltaVector. So once we compute our gradApprox vector, we can check that gradApprox ≈ deltaVector.

Once you have verified **once** that your backpropagation algorithm is correct, you don't need to compute gradApprox again. The code to compute gradApprox can be very slow, so comment it.

## Random Initialization

Initializing all theta weights to zero does not work with neural networks. When we backpropagate, all nodes will update to the same value repeatedly. Instead we can randomly initialize our weights for our Θ matrices using the following method:

![IMG](img/img15.png)

Hence, we initialize each Θ<sub>ij</sub><sup>(l)</sup> to a random value between[−ϵ,ϵ]. Using the above formula guarantees that we get the desired bound. The same procedure applies to all the Θ's. Below is some working code you could use to experiment.

```
% If the dimensions of Theta1 is 10x11, Theta2 is 10x11 and Theta3 is 1x11.

Theta1 = rand(10,11) * (2 * INIT_EPSILON) - INIT_EPSILON;
Theta2 = rand(10,11) * (2 * INIT_EPSILON) - INIT_EPSILON;
Theta3 = rand(1,11) * (2 * INIT_EPSILON) - INIT_EPSILON;
```

rand(x,y) is just a function in octave that will initialize a matrix of random real numbers between 0 and 1.

##  Putting it Together

First, pick a network architecture; choose the layout of your neural network, including how many hidden units in each layer and how many layers in total you want to have.

-   Number of input units = dimension of features  x<sup>(i)</sup>
-   Number of output units = number of classes
-   Number of hidden units per layer = usually more the better (must balance with cost of computation as it increases with more hidden units)
-   Defaults: 1 hidden layer. If you have more than 1 hidden layer, then it is recommended that you have the same number of units in every hidden layer.

### Training a Neural Network

1. Randomly initialize the weights
2. Implement forward propagation to get h<sub>Θ</sub>(x<sup>(i)</sup>) for any x<sup>(i)</sup>
3. Implement the cost function
4. Implement backpropagation to compute partial derivatives
5. Use gradient checking to confirm that your backpropagation works. Then disable gradient checking.
6. Use gradient descent or a built-in optimization function to minimize the cost function with the weights in theta.

When we perform forward and back propagation, we loop on every training example:
```
for i = 1:m,
   Perform forward propagation and backpropagation using example (x(i),y(i))
   (Get activations a(l) and delta terms d(l) for l = 2,...,L
```

Ideally, you want h<sub>Θ</sub>(x<sup>(i)</sup>)  ≈  y<sup>(i)</sup>. This will minimize our cost function. However, keep in mind that **J(Θ) is not convex** and thus we can end up in a local minimum instead.
