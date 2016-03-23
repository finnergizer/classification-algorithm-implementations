import math
from scipy import optimize
import numpy as np

def fh(x,y):
    return math.pow(x[0],y[0])

xmin = optimize.fmin(func=fh, x0=1, a)