# arc_sinc.py
# Tracy Whelen, Microwave Remote Sensing Lab, University of Massachusetts
# June 4, 2015

# This is the python version of arc_sinc.m, which calculates the inverse sinc function.

#!/usr/bin/python
from numpy import *
from math import *
from scipy import interpolate
import numpy as np

# define arc_sinc function
def arc_sinc(x, c_param):
    # Get rid of extreme values by set all values where x > 1 equal to 1, and x < 0 equal to 0 
    x[(x > 1)] = 1
    x[(x < 0)] = 0

    # Create array of increments between 0 and pi of size pi/100
    XX = linspace(0, math.pi, num=100, endpoint=True)

    # Set the first value of XX to eps to avoid division by zero issues -> Paul's suggestion
    XX[0] = spacing(1)

    # Calculate sinc for XX and save it to YY
    ## YY = sinc(XX / math.pi)
    YY = np.sin(XX) / XX

    # Reset the first value of XX to zero and the first value of YY to the corresponding output
    XX[0] = 0
    YY[0] = 1
    
    # Set the last value of YY to 0 to avoid NaN issues
    YY[-1] = 0

    # Flip XX and YY left to right
    XX = XX[::-1]
    YY = YY[::-1]
    
    # Run interpolation
    # XX and YY are your original values, x is the query values, and y is the interpolated values that correspond to x
    interp_func = interpolate.interp1d(YY, XX * c_param, kind='slinear') 
    y = interp_func(x)

    # Set all values in y less than 0 equal to 0
    y[(y < 0)] = 0
    # return y
    return y
