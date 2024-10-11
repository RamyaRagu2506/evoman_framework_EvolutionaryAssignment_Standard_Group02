# memetic_controller.py

from evoman.controller import Controller
import numpy as np


def sigmoid_activation(x):
    return 1. / (1. + np.exp(-x))

# Controller for the player agent
class player_controller(Controller):
    def __init__(self, n_hidden, predefined_weights=None):
        self.n_hidden = [n_hidden]
        # Predefined weights for the controller, if provided
        self.predefined_weights = predefined_weights

    def set(self, controller, n_inputs):
        # Ensure the controller has the expected number of weights
        assert len(controller) == 265, f"Controller has {len(controller)} weights, expected 265."

        # Number of hidden neurons
        if self.n_hidden[0] > 0:
            # Biases for the n hidden neurons
            self.bias1 = controller[:self.n_hidden[0]].reshape(1, self.n_hidden[0])
            # Weights for the connections from the inputs to the hidden nodes
            weights1_slice = n_inputs * self.n_hidden[0] + self.n_hidden[0]
            self.weights1 = controller[self.n_hidden[0]:weights1_slice].reshape((n_inputs, self.n_hidden[0]))

            # Preparing the weights and biases from the controller of layer 2
            self.bias2 = controller[weights1_slice:weights1_slice + 5].reshape(1, 5)
            self.weights2 = controller[weights1_slice + 5:].reshape((self.n_hidden[0], 5))
    def control(self, inputs, controller):
        inputs = (inputs - min(inputs)) / float((max(inputs) - min(inputs)))

        if self.n_hidden[0] > 0:
            output1 = sigmoid_activation(inputs.dot(self.weights1) + self.bias1)
            output = sigmoid_activation(output1.dot(self.weights2) + self.bias2)[0]
        else:
            bias = controller[:5].reshape(1, 5)
            weights = controller[5:].reshape((len(inputs), 5))
            output = sigmoid_activation(inputs.dot(weights) + bias)[0]

        # Map activations to actions
        left = 1 if output[0] > 0.5 else 0
        right = 1 if output[1] > 0.5 else 0
        jump = 1 if output[2] > 0.5 else 0
        shoot = 1 if output[3] > 0.5 else 0
        release = 1 if output[4] > 0.5 else 0

        return [left, right, jump, shoot, release]