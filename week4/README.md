# Neural Networks: Representation

## Neural Networks

### Model Representation I

Let's examine how we will represent a hypothesis function using neural networks. At a very simple level, neurons are basically computational units that take inputs (<b>dendrites</b>) as electrical inputs (called "spikes") that are channeled to outputs (<b>axons</b>). In our model, our dendrites are like the input features x<sub>1</sub> ... x<sub>n</sub>, and the output is the result of our hypothesis function. In this model our x<sub>0</sub> input node is sometimes called the "bias unit." It is always equal to 1. In neural networks, we use the same logistic function as in classification, 1 / ( 1 + e ^ (θ<sup>T</sup>x) ) yet we sometimes call it a sigmoid (logistic) <b>activation</b> function. In this situation, our "theta" parameters are sometimes called "weights".

Visually, a simplistic representation looks like:

![IMG](img/img1.png)


Our input nodes (layer 1), also known as the "input layer", go into another node (layer 2), which finally outputs the hypothesis function, known as the "output layer".

We can have intermediate layers of nodes between the input and output layers called the "hidden layers."

In this example, we label these intermediate or "hidden" layer nodes a<sub>0</sub><sup>2</sup> ... a<sub>n</sub><sup>2</sup> and call them "activation units."

![IMG](img/img2.png)

If we had one hidden layer, it would look like:

![IMG](img/img3.png)


The values for each of the "activation" nodes is obtained as follows:


![IMG](img/img4.png)


This is saying that we compute our activation nodes by using a 3×4 matrix of parameters. We apply each row of the parameters to our inputs to obtain the value for one activation node. Our hypothesis output is the logistic function applied to the sum of the values of our activation nodes, which have been multiplied by yet another parameter matrix θ<sup>(2)</sup> containing the weights for our second layer of nodes.

Each layer gets its own matrix of weights, θ<sup>(j)</sup>.

The dimensions of these matrices of weights is determined as follows:

If network has <i>s<sub>j</sub></i> units in layer <i>j</i> and <i>s<sub>j+1</sub></i> units in layer j+1, then θ<sup>(j)</sup> will be of dimension <i>s<sub>j+1</sub></i> X (<i>s<sub>j</sub></i> + 1)


![IMG](img/img5.png)


### Model Representation II


To re-iterate, the following is an example of a neural network:

![IMG](img/img6.png)

In this section we'll do a vectorized implementation of the above functions. We're going to define a new variable z<sub>k</sub><sup>(j)</sup> that encompasses the parameters inside our g function. In our previous example if we replaced by the variable z for all the parameters we would get:


![IMG](img/img7.png)


In other words, for layer j=2 and node k, the variable z will be:

![IMG](img/img8.png)

The vector representation of x and z<sup>j</sup> is:

![IMG](img/img9.png)

Setting x = a<sup>(1)</sup>, we can rewrite the equation as:

![IMG](img/img10.png)


We are multiplying our matrix Θ<sup>(j−1)</sup> with dimensions s<sub>j</sub> X (n+1)s (where s<sub>j</sub> is the number of our activation nodes) by our vector a<sup>(j−1)</sup> with height (n+1). This gives us our vector z<sup>(j)</sup> with height s<sub>j</sub>.
Now we can get a vector of our activation nodes for layer j as follows:

![IMG](img/img11.png)

Where our function g can be applied element-wise to our vector z<sup>(j)</sup>.

We can then add a bias unit (equal to 1) to layer j after we have computed a<sup>(j)</sup>. This will be element a<sub>0</sub><sup>(j)</sup> and will be equal to 1. To compute our final hypothesis, let's first compute another z vector:

![IMG](img/img12.png)


We get this final z vector by multiplying the next theta matrix after Θ<sup>(j−1)</sup> with the values of all the activation nodes we just got. This last theta matrix Θ<sup>(j)</sup> will have only <b>one row</b> which is multiplied by one column a<sup>(j)</sup> so that our result is a single number. We then get our final result with:

![IMG](img/img13.png)

Notice that in this <b>last step</b>, between layer j and layer j+1, we are doing <b>exactly the same thing</b> as we did in logistic regression. 
Adding all these intermediate layers in neural networks allows us to more elegantly produce interesting and more complex non-linear hypotheses.