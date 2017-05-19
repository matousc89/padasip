r"""
.. versionadded:: 0.3

In this module is stored everything related to Multi-layer perceptron (MLP).
This neural network can be used for classification and regression.


Minimal Working Example
************************

.. code-block:: python

    import numpy as np
    import padasip as pa

    # data creation
    x = np.array([
            [0,0,0,0], [1,0,0,0], [0,1,0,0], [1,1,0,0],
            [0,0,1,0], [1,0,1,0], [0,1,1,0], [1,1,1,0],
            [0,0,0,1], [1,0,0,1], [0,1,0,1], [1,1,0,1],
            [0,0,1,1], [1,0,1,1], [0,1,1,1], [1,1,1,1]
        ])
    d = np.array([0,1,1,0,0,1,0,0,1,0,1,0,1,1,1,0])
    N = len(d)
    n = 4

    # creation of neural network
    nn = pa.ann.NetworkMLP([5,6], n, outputs=1, activation="tanh", mu="auto")    

    # training
    e, mse = nn.train(x, d, epochs=200, shuffle=True)    

    # get results
    y = nn.run(x)

And the result (pairs: target, output) can look like

>>> for i in zip(d, y): print i
... 
(0, 0.0032477183193071906)
(1, 1.0058082383308447)
(1, 1.0047503447788306)
(0, 0.0046026142618665845)
(0, 0.0003037425037410007)
(1, 1.0017672193832869)
(0, 0.0015817734995124679)
(0, 0.0019115885715706904)
(1, 0.99342117275580499)
(0, 0.00069114178424850147)
(1, 1.0021789943501729)
(0, 0.0021355836851727717)
(1, 0.99809312951378826)
(1, 1.0071488717506856)
(1, 1.0067500768423701)
(0, -0.0045962250501771244)
>>> 



Learning Rate Selection
**************************

If you select the learning rate (:math:`\mu` in equations,
or `mu` in code) manually, it will be used the same value for all nodes,
otherwise it is selected automatically :cite:`lecun2012efficient` as follows

:math:`\mu_{ij} = m^{-0.5}`

where the :math:`m` is the amount of nodes on input of given node.
The automatic selection is recomended and default option.


Default Values of Weights
****************************

The distribution from what the weights are taken is chosen automatically
:cite:`lecun2012efficient`, it has zero mean and
the standard derivation estimated as follows

:math:`\sigma_{w} = m^{-0.5}`

where the :math:`m` is the amount of nodes on input of given node.


References
***************

.. bibliography:: mlp.bib
    :style: plain

Code Explanation
******************
"""
import numpy as np

class Layer():
    """
    This class represents a single hidden layer of the MLP.

    Args:

    * `n_layer` : size of the layer (int)

    * `n_input` : how many inputs the layer have (int)

    * `activation_f` : what function should be used as activation function (str)

    * `mu` : learning rate (float or str), it can be directly the float value,
        or string `auto` for automatic selection of learning rate
        :cite:`lecun2012efficient`

    """
    
    def __init__(self, n_layer, n_input, activation_f, mu):
        sigma = n_input**(-0.5)
        if mu == "auto":
            self.mu = sigma
        else:
            self.mu = mu
        self.n_input = n_input
        self.w = np.random.normal(0, sigma, (n_layer, n_input+1))
        self.x = np.ones(n_input+1)
        self.y = np.zeros(n_input+1)
        self.f = activation_f

    def activation(self, x, f="sigmoid", der=False):
        """
        This function process values of layer outputs with activation function.

        **Args:**

        * `x` : array to process (1-dimensional array) 

        **Kwargs:**

        * `f` : activation function

        * `der` : normal output, or its derivation (bool)

        **Returns:**

        * values processed with activation function (1-dimensional array)
        
        """
        if f == "sigmoid":
            if der:
                return x * (1 - x)
            return 1. / (1 + np.exp(-x))
        elif f == "tanh":
            if der:
                return 1 - x**2 
            return (2. / (1 + np.exp(-2*x))) - 1       
                
    def predict(self, x):
        """
        This function make forward pass through this layer (no update).

        **Args:**

        * `x` : input vector (1-dimensional array)

        **Returns:**
        
        * `y` : output of MLP (float or 1-diemnsional array).
            Size depends on number of nodes in this layer.
            
        """
        self.x[1:] = x
        self.y = self.activation(np.sum(self.w*self.x, axis=1), f=self.f)
        return self.y
    
    def update(self, w, e):
        """
        This function make update according provided target
        and the last used input vector.

        **Args:**

        * `d` : target (float or 1-dimensional array).
            Size depends on number of MLP outputs.

        **Returns:**

        * `w` : weights of the layers (2-dimensional layer).
            Every row represents one node.
        
        * `e` : error used for update (float or 1-diemnsional array).
            Size correspond to size of input `d`.
        """
        if len(w.shape) == 1:
            e = self.activation(self.y, f=self.f, der=True) * e * w
            dw = self.mu * np.outer(e, self.x)
        else:
            e = self.activation(self.y, f=self.f, der=True) * (1 - self.y) * np.dot(e, w)
            dw = self.mu * np.outer(e, self.x)
        w = self.w[:,1:]
        self.w += dw
        return w, e
        
        
class NetworkMLP():
    """
    This class represents a Multi-layer Perceptron neural network.

    *Args:**

    * `layers` : array describing hidden layers of network
        (1-dimensional array of integers). Every number in array represents
        one hidden layer. For example [3, 6, 2] create
        network with three hidden layers. First layer will have 3 nodes,
        second layer will have 6 nodes and the last hidden layer
        will have 2 nodes.

    * `n_input` : number of network inputs (int). 

    **Kwargs:**

    * `outputs` : number of network outputs (int). Default is 1.

    * `activation` : activation function (str)

        * "sigmoid" - sigmoid
    
        * "tanh" : hyperbolic tangens

    * `mu` : learning rate (float or str), it can be:
        * float value - value is directly used as `mu`

        * "auto" - this will trigger automatic selection of learning rate
        according to :cite:`lecun2012efficient`

    """

    def __init__(self, layers, n_input, outputs=1, activation="sigmoid", mu="auto"):
        sigma = layers[-1]**(-0.5)
        # set learning rate
        if mu == "auto":
            self.mu = sigma
        else:
            try:
                param = float(mu)            
            except:
                raise ValueError(
                    'Parameter mu is not float or similar'
                    )
            self.mu = mu
        self.n_input = n_input
        # create output layer
        self.outputs = outputs
        if self.outputs == 1:
            self.w = np.random.normal(0, sigma, layers[-1]+1)
        else:
            self.w = np.random.normal(0, sigma, (outputs, layers[-1]+1))
        self.x = np.ones(layers[-1]+1)
        self.y = 0
        # create hidden layers
        self.n_layers = len(layers)
        self.layers = []
        for n in range(self.n_layers):
            if n == 0:
                l = Layer(layers[n], n_input, activation, mu)
                self.layers.append(l)
            else:
                l = Layer(layers[n], layers[n-1], activation, mu)
                self.layers.append(l)
 
    def train(self, x, d, epochs=10, shuffle=False):
        """
        Function for batch training of MLP.

        **Args:**

        * `x` : input array (2-dimensional array).
            Every row represents one input vector (features).

        * `d` : input array (n-dimensional array).
            Every row represents target for one input vector.
            Target can be one or more values (in case of multiple outputs).

        **Kwargs:**
        
        * `epochs` : amount of epochs (int). That means how many times
            the MLP will iterate over the passed set of data (`x`, `d`).

        * `shuffle` : if true, the order of inputs and outpust are shuffled (bool).
            That means the pairs input-output are in different order in every epoch.

        **Returns:**
        
        * `e`: output vector (m-dimensional array). Every row represents
            error (or errors) for an input and output in given epoch.
            The size of this array is length of provided data times
            amount of epochs (`N*epochs`).

        * `MSE` : mean squared error (1-dimensional array). Every value
            stands for MSE of one epoch.
            
        """
        # measure the data and check if the dimmension agree
        N = len(x)
        if not len(d) == N:
            raise ValueError('The length of vector d and matrix x must agree.')  
        if not len(x[0]) == self.n_input:
            raise ValueError('The number of network inputs is not correct.')
        if self.outputs == 1:
            if not len(d.shape) == 1:
                raise ValueError('For one output MLP the d must have one dimension')
        else:
            if not d.shape[1] == self.outputs:
                raise ValueError('The number of outputs must agree with number of columns in d')
        try:    
            x = np.array(x)
            d = np.array(d)
        except:
            raise ValueError('Impossible to convert x or d to a numpy array')
        # create empty arrays
        if self.outputs == 1:
            e = np.zeros(epochs*N)
        else:
            e = np.zeros((epochs*N, self.outputs))
        MSE = np.zeros(epochs)
        # shuffle data if demanded
        if shuffle:
            randomize = np.arange(len(x))
            np.random.shuffle(randomize)
            x = x[randomize]
            d = d[randomize]
        # adaptation loop
        for epoch in range(epochs):
            for k in range(N):
                self.predict(x[k])
                e[(epoch*N)+k] = self.update(d[k])
            MSE[epoch] = np.sum(e[epoch*N:(epoch+1)*N-1]**2) / N
        return e, MSE

    def run(self, x):
        """
        Function for batch usage of already trained and tested MLP.

        **Args:**

        * `x` : input array (2-dimensional array).
            Every row represents one input vector (features).

        **Returns:**
        
        * `y`: output vector (n-dimensional array). Every row represents
            output (outputs) for an input vector.
            
        """
        # measure the data and check if the dimmension agree
        try:    
            x = np.array(x)
        except:
            raise ValueError('Impossible to convert x to a numpy array')
        N = len(x)   
        # create empty arrays     
        if self.outputs == 1:
            y = np.zeros(N)
        else:
            y = np.zeros((N, self.outputs))
        # predict data in loop        
        for k in range(N):
            y[k] = self.predict(x[k])
        return y

    def test(self, x, d):
        """
        Function for batch test of already trained MLP.

        **Args:**

        * `x` : input array (2-dimensional array).
            Every row represents one input vector (features).

        * `d` : input array (n-dimensional array).
            Every row represents target for one input vector.
            Target can be one or more values (in case of multiple outputs).

        **Returns:**
        
        * `e`: output vector (n-dimensional array). Every row represents
            error (or errors) for an input and output.
            
        """
        # measure the data and check if the dimmension agree
        N = len(x)
        if not len(d) == N:
            raise ValueError('The length of vector d and matrix x must agree.')  
        if not len(x[0]) == self.n_input:
            raise ValueError('The number of network inputs is not correct.')
        if self.outputs == 1:
            if not len(d.shape) == 1:
                raise ValueError('For one output MLP the d must have one dimension')
        else:
            if not d.shape[1] == self.outputs:
                raise ValueError('The number of outputs must agree with number of columns in d')
        try:    
            x = np.array(x)
            d = np.array(d)
        except:
            raise ValueError('Impossible to convert x or d to a numpy array')
        # create empty arrays
        if self.outputs == 1:
            y = np.zeros(N)
        else:
            y = np.zeros((N, self.outputs))
        # measure in loop        
        for k in range(N):
            y[k] = self.predict(x[k])
        return d - y    

    def predict(self, x):
        """
        This function make forward pass through MLP (no update).

        **Args:**

        * `x` : input vector (1-dimensional array)

        **Returns:**
        
        * `y` : output of MLP (float or 1-diemnsional array).
            Size depends on number of MLP outputs.
            
        """
        # forward pass to hidden layers
        for l in self.layers:
            x = l.predict(x)
        self.x[1:] = x
        # forward pass to output layer
        if self.outputs == 1:
            self.y = np.dot(self.w, self.x)
        else: 
            self.y = np.sum(self.w*self.x, axis=1)
        return self.y
    
    def update(self, d):
        """
        This function make update according provided target
        and the last used input vector.

        **Args:**

        * `d` : target (float or 1-dimensional array).
            Size depends on number of MLP outputs.

        **Returns:**
        
        * `e` : error used for update (float or 1-diemnsional array).
            Size correspond to size of input `d`.
            
        """
        # update output layer
        e = d - self.y
        error = np.copy(e)
        if self.outputs == 1:
            dw = self.mu * e * self.x   
            w = np.copy(self.w)[1:]
        else:
            dw = self.mu * np.outer(e, self.x)
            w = np.copy(self.w)[:,1:]
        self.w += dw
        # update hidden layers
        for l in reversed(self.layers):
            w, e = l.update(w, e) 
        return error


