
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

4. Compute δ<sup>(L-1)</sup>, δ<sup>(L-2)</sup>, ... ,δ<sup>(2)</sup> using δ<sup>(t)</sup> =( (Θ<sup>(t)</sup>)<sup>T</sup> δ<sup>(<i> l</i>+1)</sup>)  .∗  a<sup>(<i>l</i>)</sup>  .∗  (1−a<sup>(<i>l</i>)</sup>)
The delta values of layer l are calculated by multiplying the delta values in the next layer with the theta matrix of layer l. We then element-wise multiply that with a function called g', or g-prime, which is the derivative of the activation function g evaluated with the input values given by  z<sup>(<i>l</i>)</sup>.
The g-prime derivative terms can also be written out as:

![IMG](img/img6.png)

5. Δ<sub>i,j</sub><sup>(l)</sup> ​:= Δ<sub>i,j</sub><sup>(l)</sup> + a<sub>j</sub><sup>(l)</sup> ​δ<sub>i</sub><sup>(l+1)</sup>​ or with vectorization, Δ<sup>(l)</sup> := Δ<sup>(l)</sup>+δ<sup>(l+1)</sup> (a<sup>(l)</sup>)<sup><i>T</i></sup>
Hence we update our new \DeltaΔ matrix.

![IMG](img/img7.png)

The capital-delta matrix D is used as an "accumulator" to add up our values as we go along and eventually compute our partial derivative. Thus we get <sup>∂</sup> / <sub>∂Θ<sub>ij</sub><sup>(l)</sup></sub> J(Θ) =  D<sub>ij</sub><sup>(l)​</sup>