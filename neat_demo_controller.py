from evoman.controller import Controller
import numpy as np


def sigmoid_activation(x):
    return 1. / (1. + np.exp(-x))


# implements controller structure for player
class player_controller(Controller):
    def __init__(self, _n_hidden):
        self.n_hidden = [_n_hidden]

    def control(self, inputs, controller):
        # Normalises the input using min-max scaling
        inputs = (inputs - min(inputs)) / float((max(inputs) - min(inputs)))

        # Use the NEAT network (controller) to generate output actions
        output = controller.activate(inputs)

        # takes decisions about sprite actions based on the network output
        left = 1 if output[0] > 0.5 else 0
        right = 1 if output[1] > 0.5 else 0
        jump = 1 if output[2] > 0.5 else 0
        shoot = 1 if output[3] > 0.5 else 0
        release = 1 if output[4] > 0.5 else 0

        return [left, right, jump, shoot, release]

# implements controller structure for enemy
class enemy_controller(Controller):
    def __init__(self, _n_hidden):
        self.n_hidden = [_n_hidden]

    def control(self, inputs, controller):
        # Normalises the input using min-max scaling
        inputs = (inputs - min(inputs)) / float((max(inputs) - min(inputs)))

        # Use the NEAT network (controller) to generate output actions
        output = controller.activate(inputs)

        # takes decisions about sprite actions based on the network output
        attack1 = 1 if output[0] > 0.5 else 0
        attack2 = 1 if output[1] > 0.5 else 0
        attack3 = 1 if output[2] > 0.5 else 0
        attack4 = 1 if output[3] > 0.5 else 0

        return [attack1, attack2, attack3, attack4]