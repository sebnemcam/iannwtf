#DISCLAIMER: We did not manage to solve all tasks this week. We solved task 4 and 5 without real data, because we could not get our MLP to work.
#We speculated about the error source at the appropriate place, namely at the backward_step method of the Layer class.

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from numpy.random import default_rng

# Task 01 - Building our data set

#our input values x
x = np.array(default_rng(42).random((1,100)))
print(x)
#our targets t
t = x**2

# Task 02 - Perceptrons

class Layer:
    #constructor with all necessary parameters
    def __init__(self, n_units, input_units):
        '''Instantiates a Layer object.

        Args:
            n_units::int
                The number of units in the layer
            input_units::int
                The number of units in the previous layer, which are the input units to this layer
        '''

        self.n_units = n_units
        self.input_units = input_units
        self.biases = np.zeros(n_units)
        self.weights = np.random.random((input_units,n_units))
        self.input = np.array([])
        self.preactivation = np.empty(n_units)
        self.activation = np.empty(n_units)

    # method for forward step
    def forward_step(self, previous):
        '''Passes a given input through a single layer.

        Args:
            previous::list
                The activations of the previous layer

        Returns:
            self.activation::list
                The activations of this layer
        '''

        self.input = previous
        self.preactivation = (self.input * self.weights) + self.biases
        # ReLu: activation is equal to the preactivation if it is above 0, else the activation is 0
        self.activation = np.maximum(0, self.preactivation)
        return self.activation

    def backward_step(self, dLda):
        '''Updates a layer's weights and biases according to the activation gradient.

        Args:
            dlda::list
                The gradient of the loss function with respect to the layer's activations

        Returns:
            dLda * (self.preactivation > 0) * tf.transpose(self.weights)::list
                The gradient of the loss function with respect to the previous layer's activations 
        '''

        #Something probably goes wrong here. There seems to be an error with different shapes that python can't work with,
        #however we failed to resolve the mistake.

        # dLda = derivative of MSE
        dLda = self.input - t

        # derivative of ReLu
        deriv_preactivation = [1 if self.preactivation[i] > 0 else 0 for i in self.preactivation]

        # computing the gradients
        gradients_weights = tf.transpose(self.input) * deriv_preactivation * dLda
        gradients_biases = deriv_preactivation * dLda

        # updating the layer's parameters
        self.weights = self.weights - (0.03 * gradients_weights)
        self.biases = self.biases - (0.03 * gradients_biases)

        return dLda * (self.preactivation > 0) * tf.transpose(self.weights)

# Task 03: Multi-layer Perceptron

class MLP:
    # constructor
    def __init__(self, layers):
        '''Instantiates an MLP object.

        Args:
            layers::list
                The multi-layer-perceptron's layers.
        '''

        self.layers = np.array(layers)
        self.output = []

    def forward_step(self, input):
        '''Passes an input through the entire multi-layer-perceptron.

        Args:
            input::float
                The input given into the system
        
        Returns:
            self.layers[-1].activation::float
                The last layer's activation, which is also the system's output
        '''

        for i in range(0, len(self.layers) - 1):
            if i == 0:
                # first layer feeds forward the given input
                self.layers[i].forward_step(input)
            else:
                # every following layer takes the output of the previous layer as input
                self.layers[i].forward_step(self.layers[i - 1].activation)
            print(self.layers[i].activation)
        print(self.layers[-1].activation)
        return self.layers[-1].activation

    def backpropagation(self, dLda):
        '''Updates the system's weights and biases.

        Args:
            dLda::list
                The gradient of the loss function with respect to the system's output.
        '''
        for layer in np.flip(self.layers):
            # updating dL/da
            dLda = layer.backward_step(dLda)
        return

# Task 04: Training
layer1 = Layer(10, 1)
layer2 = Layer(1, 10)

mlp = MLP([layer1, layer2])

loss_tracker = []
epoch_loss_tracker = []

for i in range(1000):
    for j in range(0, len(x)-1):
        output = mlp.forward_step(x[j])
        loss = 0.5 * (output - t[j])**2
        mlp.backpropagation(output-t[j])
        loss_tracker.append(loss)
    epoch_loss_tracker.append(loss_tracker.mean())

# Task 05: Visualization

# if our code was working as intended, the graph visualizing the mean loss per epoch could somewhat look like this
x = (np.array(range(101)))*10
epoch_loss_tracker: float = 1/(x+1)
y = epoch_loss_tracker


plt.xlabel("epochs")
plt.ylabel("mean loss")
plt.title('loss tracker')
plt.plot(x,y)
plt.show()
