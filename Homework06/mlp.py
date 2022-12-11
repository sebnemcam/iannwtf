import numpy as np
from numpy.random import default_rng
import tensorflow as tf


# Task 01 - Building our data set
'''
# our input values x
x = np.array(default_rng(42).random((1,100)))

print(x)

#our targets t
t = x**2
'''

# Task 02 - Perceptrons


class Layer:

    def __init__(self, n_units, input_units):
        self.n_units = n_units
        self.input_units = input_units
        self.biases = np.zeros(n_units)
        self.weights = np.random.random((input_units, n_units))
        self.input = np.array([])
        self.preactivation = np.empty(n_units)
        self.activation = np.empty(n_units)
        #self.activation = np.array([1, 2, 3])

    def forward_step(self, previous):
        self.input = previous
        self.preactivation = (self.input * self.weights) + self.biases
        #print(np.maximum(0, self.preactivation))
        self.activation = np.maximum(0, self.preactivation)
        return self.activation

    def backward_step(self, dLda):
        deriv_preactivation = [1 if self.preactivation[i] > 0 else 0 for i in self.preactivation]
        gradients_weights = tf.transpose(self.input) * deriv_preactivation * dLda
        #print(gradients_weights)
        gradients_biases = (self.preactivation > 0) * dLda
        #print(gradients_biases)
        #print(self.weights)
        self.weights = self.weights - (0.03 * gradients_weights)
        #print(self.weights)
        #print(self.biases)
        self.biases = self.biases - (0.03 * gradients_biases)
        #print(self.biases)
        return dLda * (self.preactivation > 0) * tf.transpose(self.weights)

class MLP:

    def __init__(self, layers):
        self.layers = np.array(layers)
        self.output = []
        

    def forward_step(self, input):
        print(input)
        for i in range(0, len(self.layers)-1):
            print(self.layers[i].weights)
            if i == 0:
                self.layers[i].forward_step(input)
            else:
                self.layers[i].forward_step(self.layers[i-1].activation)
            print(self.layers[i].activation)
        print(self.layers[-1].activation)
        return self.layers[-1].activation

    def backpropagation(self, dLda):
        for layer in np.flip(self.layers):
            dLda = layer.backward_step(dLda)
            #print(layer.weights)
            #print(layer.biases)
        return



layer1 = Layer(10, 1)
layer2 = Layer(1, 10)


mlp = MLP([layer1, layer2])
#mlp.forward_step(4)
mlp.backpropagation(mlp.forward_step(4) - 16)
#layer1.forward_step(np.array([2, 3, 4]))

#targets = [1, 4, 5]
