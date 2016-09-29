"""
.. versionchanged:: 0.4
"""
import numpy as np
import padasip.consts as co

class AdaptiveFilter():
    """
    Base class for adaptive filter classes. It puts together some functions
    used by all adaptive filters.
    """

    def init_weights(self, w, n):
        """
        This function initialises the adaptive weights of the filter.

        Args:

        * `w` : initial weights of filter. Possible values are:
        
            * array with initial weights (1 dimensional array) of filter size
        
            * "random" : create random weights
            
            * "zeros" : create zero value weights

        * `n` : size of filter (int) - number of filter coefficients.

        Returns:

        * `y` : output value (float) calculated from input array.

        """
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
        return w    

    def predict(self, x):
        """
        This function calculates the new output value `y` from input array `x`.

        Args:

        * `x` : input vector (1 dimension array) in length of filter.

        Returns:

        * `y` : output value (float) calculated from input array.

        """
        y = np.dot(self.w, x)
        return y

    def check_float_param(self, param, low, high, name):
        """
        Check if the value of the given parameter is in the given range
        and a float.
        Designed for testing parameters like `mu` and `eps`.
        To pass this function the variable `param` must be able to be converted
        into a float with a value between `low` and `high`.

        Args:

        * `param` : parameter to check (float or similar)

        * `low` : lowest allowed value (float), or None

        * `high` : highest allowed value (float), or None

        * `name` : name of the parameter (string), it is used for an error message
            
        Returns:

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
        """
        if type(param) == int:
            return param
        else:
            raise ValueError(error_msg)   

    def check_int_param(self, param, low, high, name):
        """
        Check if the value of the given parameter is in the given range
        and an int.
        Designed for testing parameters like `mu` and `eps`.
        To pass this function the variable `param` must be able to be converted
        into a float with a value between `low` and `high`.

        Args:

        * `param` : parameter to check (int or similar)

        * `low` : lowest allowed value (int), or None

        * `high` : highest allowed value (int), or None

        * `name` : name of the parameter (string), it is used for an error message
            
        Returns:

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

