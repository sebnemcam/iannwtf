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

class MLP:

    def __init__(self, layers):
        self.layers = np.array(layers)
        self.output = []

    def feed_forward(self, input):
        self.layers[0].forward_step(input)
        print(self.layers[0].activation)
        for i in range(1, len(self.layers)-1):
            self.layers[i+1].forward_step(self.layers[i].activation)
            print(self.layers[i+1].activation)
        return self.layers[-1].activation

    def backpropagation(self, loss):
        
        return



layer1 = Layer(3, 1)
layer2 = Layer(3, 3)
layer3 = Layer(3, 3)
layer4 = Layer(1, 3)

mlp = MLP([layer1, layer2, layer3, layer4])
mlp.feed_forward(4)

#layer1.forward_step(np.array([2, 3, 4]))

#targets = [1, 4, 5]

#layer1.backward_step(targets)

