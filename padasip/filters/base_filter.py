"""
.. versionadded:: 0.1
.. versionchanged:: 0.7

"""
import numpy as np

from padasip.misc import get_mean_error

class AdaptiveFilter():
    """
    Base class for adaptive filter classes. It puts together some functions
    used by all adaptive filters.
    """

    def init_weights(self, w, n=-1):
        """
        This function initialises the adaptive weights of the filter.

        **Args:**

        * `w` : initial weights of filter. Possible values are:
        
            * array with initial weights (1 dimensional array) of filter size
        
            * "random" : create random weights
            
            * "zeros" : create zero value weights

       
        **Kwargs:**

        * `n` : size of filter (int) - number of filter coefficients.

        **Returns:**

        * `y` : output value (float) calculated from input array.

        """
        if n == -1:
            n = self.n
        if type(w) == str:
            if w == "random":
                w = np.random.normal(0, 0.5, n)
            elif w == "zeros":
                w = np.zeros(n)
            else:
                raise ValueError('Impossible to understand the w')
        elif len(w) == n:
            try:
                w = np.array(w, dtype="float64")
            except:
                raise ValueError('Impossible to understand the w')
        else:
            raise ValueError('Impossible to understand the w')
        self.w = w    

    def predict(self, x):
        """
        This function calculates the new output value `y` from input array `x`.

        **Args:**

        * `x` : input vector (1 dimension array) in length of filter.

        **Returns:**

        * `y` : output value (float) calculated from input array.

        """
        y = np.dot(self.w, x)
        return y

    def pretrained_run(self, d, x, ntrain=0.5, epochs=1):
        """
        This function sacrifices part of the data for few epochs of learning.
        
        **Args:**

        * `d` : desired value (1 dimensional array)

        * `x` : input matrix (2-dimensional array). Rows are samples,
          columns are input arrays.
       
        **Kwargs:**

        * `ntrain` : train to test ratio (float), default value is 0.5
          (that means 50% of data is used for training)
          
        * `epochs` : number of training epochs (int), default value is 1.
          This number describes how many times the training will be repeated
          on dedicated part of data.

        **Returns:**
        
        * `y` : output value (1 dimensional array).
          The size corresponds with the desired value.

        * `e` : filter error for every sample (1 dimensional array).
          The size corresponds with the desired value.

        * `w` : vector of final weights (1 dimensional array).    
        """
        Ntrain = int(len(d)*ntrain)
        # train
        for epoch in range(epochs):
            self.run(d[:Ntrain], x[:Ntrain])
        # test
        y, e, w = self.run(d[Ntrain:], x[Ntrain:])
        return y, e, w

    def explore_learning(self, d, x, mu_start=0, mu_end=1., steps=100,
            ntrain=0.5, epochs=1, criteria="MSE", target_w=False):
        """
        Test what learning rate is the best.

        **Args:**

        * `d` : desired value (1 dimensional array)

        * `x` : input matrix (2-dimensional array). Rows are samples,
          columns are input arrays.
       
        **Kwargs:**
        
        * `mu_start` : starting learning rate (float)
        
        * `mu_end` : final learning rate (float)
        
        * `steps` : how many learning rates should be tested between `mu_start`
          and `mu_end`.

        * `ntrain` : train to test ratio (float), default value is 0.5
          (that means 50% of data is used for training)
          
        * `epochs` : number of training epochs (int), default value is 1.
          This number describes how many times the training will be repeated
          on dedicated part of data.
          
        * `criteria` : how should be measured the mean error (str),
          default value is "MSE".
          
        * `target_w` : target weights (str or 1d array), default value is False.
          If False, the mean error is estimated from prediction error.
          If an array is provided, the error between weights and `target_w`
          is used.

        **Returns:**
        
        * `errors` : mean error for tested learning rates (1 dimensional array).

        * `mu_range` : range of used learning rates (1d array). Every value
          corresponds with one value from `errors`

        """
        mu_range = np.linspace(mu_start, mu_end, steps)
        errors = np.zeros(len(mu_range))
        for i, mu in enumerate(mu_range):
            # init
            self.init_weights("zeros")
            self.mu = mu
            # run
            y, e, w = self.pretrained_run(d, x, ntrain=ntrain, epochs=epochs)
            if type(target_w) != bool:
                errors[i] = get_mean_error(w[-1]-target_w, function=criteria)
            else:
                errors[i] = get_mean_error(e, function=criteria)
        return errors, mu_range            

    def check_float_param(self, param, low, high, name):
        """
        Check if the value of the given parameter is in the given range
        and a float.
        Designed for testing parameters like `mu` and `eps`.
        To pass this function the variable `param` must be able to be converted
        into a float with a value between `low` and `high`.

        **Args:**

        * `param` : parameter to check (float or similar)

        * `low` : lowest allowed value (float), or None

        * `high` : highest allowed value (float), or None

        * `name` : name of the parameter (string), it is used for an error message
            
        **Returns:**

        * `param` : checked parameter converted to float

        """
        try:
            param = float(param)            
        except:
            raise ValueError(
                'Parameter {} is not float or similar'.format(name)
                )
        if low != None or high != None:
            if not low <= param <= high:
                raise ValueError('Parameter {} is not in range <{}, {}>'
                    .format(name, low, high))    
        return param 
        
    def check_int(self, param, error_msg):
        """
        This function check if the parameter is int.
        If yes, the function returns the parameter,
        if not, it raises error message.
        
        **Args:**
        
        * `param` : parameter to check (int or similar)

        * `error_ms` : lowest allowed value (int), or None        
        
        **Returns:**
        
        * `param` : parameter (int)
        """
        if type(param) == int:
            return int(param)
        else:
            raise ValueError(error_msg)   

    def check_int_param(self, param, low, high, name):
        """
        Check if the value of the given parameter is in the given range
        and an int.
        Designed for testing parameters like `mu` and `eps`.
        To pass this function the variable `param` must be able to be converted
        into a float with a value between `low` and `high`.

        **Args:**

        * `param` : parameter to check (int or similar)

        * `low` : lowest allowed value (int), or None

        * `high` : highest allowed value (int), or None

        * `name` : name of the parameter (string), it is used for an error message
            
        **Returns:**

        * `param` : checked parameter converted to float

        """
        try:
            param = int(param)            
        except:
            raise ValueError(
                'Parameter {} is not int or similar'.format(name)
                )
        if low != None or high != None:
            if not low <= param <= high:
                raise ValueError('Parameter {} is not in range <{}, {}>'
                    .format(name, low, high))    
        return param      

