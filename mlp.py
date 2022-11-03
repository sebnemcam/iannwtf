import numpy as np
from numpy.random import default_rng

# Task 01 - Building our data set

# our input values x
x = np.array(default_rng(42).random((1,100)))

print(x)

#our targets t
t = x**3 - x**2

print(t)
