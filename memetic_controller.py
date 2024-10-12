# memetic_controller.py

from evoman.controller import Controller
import numpy as np


def sigmoid_activation(x):
    return 1. / (1. + np.exp(-x))


# Controller for the player agent
class player_controller(Controller):
    def __init__(self, n_hidden):
        self.n_hidden = [n_hidden]

    class player_controller(Controller):
        def __init__(self, n_hidden):
            self.n_hidden = [n_hidden]

        def set(self, controller, n_inputs):
            # Ensure that the number of parameters matches the expected size
            assert len(controller) == 265, f"Expected 265 parameters, but got {len(controller)}"

            if self.n_hidden[0] > 0:
                # Biases for hidden neurons
                self.bias1 = controller[:self.n_hidden[0]].reshape(1, self.n_hidden[0])

                # Weights from inputs to hidden layer
                weights1_start = self.n_hidden[0]
                weights1_end = weights1_start + n_inputs * self.n_hidden[0]
                self.weights1 = controller[weights1_start:weights1_end].reshape((n_inputs, self.n_hidden[0]))

                # Biases for output neurons
                bias2_start = weights1_end
                bias2_end = bias2_start + 5
                self.bias2 = controller[bias2_start:bias2_end].reshape(1, 5)

                # Weights from hidden layer to output layer
                weights2_start = bias2_end
                self.weights2 = controller[weights2_start:].reshape((self.n_hidden[0], 5))

        def control(self, inputs, controller):
            # Normalize the inputs using min-max scaling
            inputs = (inputs - min(inputs)) / float((max(inputs) - min(inputs)))

            if self.n_hidden[0] > 0:
                # Forward pass through the input-to-hidden layer
                output1 = sigmoid_activation(inputs.dot(self.weights1) + self.bias1)

                # Forward pass through the hidden-to-output layer
                output = sigmoid_activation(output1.dot(self.weights2) + self.bias2)[0]
            else:
                # If no hidden neurons, we directly map inputs to outputs
                bias = controller[:5].reshape(1, 5)
                weights = controller[5:].reshape((len(inputs), 5))
                output = sigmoid_activation(inputs.dot(weights) + bias)[0]

            # Convert the output activations into actions
            left = 1 if output[0] > 0.5 else 0
            right = 1 if output[1] > 0.5 else 0
            jump = 1 if output[2] > 0.5 else 0
            shoot = 1 if output[3] > 0.5 else 0
            release = 1 if output[4] > 0.5 else 0

            return [left, right, jump, shoot, release]