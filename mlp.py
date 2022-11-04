import numpy as np
from numpy.random import default_rng
import tensorflow as tf

'''
# Task 01 - Building our data set

# our input values x
x = np.array(default_rng(42).random((1,100)))

print(x)

#our targets t
t = x**3 - x**2

print(t)
'''
# Task 02 - Perceptrons


class Layer:

    def __init__(self, n_units, input_units):
        self.n_units = n_units
        self.input_units = input_units
        self.biases = np.zeros(n_units)
        self.weights = np.random.random((input_units, n_units))
        #self.input = np.array([])
        self.input = np.array([0.5, 3, 7])
        self.preactivation = np.empty(n_units)
        #self.activation = np.empty(n_units)
        self.activation = np.array([1, 2, 3])

    def forward_step(self, prev_activation):
        self.preactivation = (prev_activation * self.weights) + self.biases
        print(np.maximum(0, self.preactivation))
        return np.maximum(0, self.preactivation)

    def backward_step(self, targets):
        diff = self.activation - targets
        #print("{} = {}".format("diff", diff))
        diff_squared = diff**2
        #print("{} = {}".format("diff_squared", diff_squared))
        loss = diff_squared.mean()
        #print("{} = {}".format("loss", loss))
        gradients_weights = tf.transpose(self.input)*(tf.nn.sigmoid(self.preactivation) * (1 - tf.nn.sigmoid(self.preactivation))) * loss
        #print(gradients_weights)
        gradients_biases = tf.nn.sigmoid(self.preactivation) * (1 - tf.nn.sigmoid(self.preactivation)) * loss
        #print(gradients_biases)
        gradients_input = gradients_biases * tf.transpose(self.weights)
        #print(gradients_input)
        #print(self.weights)
        self.weights = self.weights - (0.03 * gradients_weights)
        #print(self.weights)
        #print(self.biases)
        self.biases = self.biases - (0.03 * gradients_biases)
        #print(self.biases)
        return


layer1 = Layer(3, 3)

#layer1.forward_step(np.array([2, 3, 4]))

targets = [1, 4, 5]

layer1.backward_step(targets)


