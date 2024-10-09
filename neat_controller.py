from evoman.controller import Controller
import numpy as np


# Controller for the player agent
class NEATController(Controller):
    def __init__(self, neat_network):
        """
        The constructor accepts the NEAT neural network object.
        """
        self.neat_network = neat_network  # Store the NEAT neural network

    def control(self, inputs, controller=None):
        """
        Use the NEAT neural network to generate actions from inputs.

        Inputs:
        - inputs: Sensor readings from the game environment
        - controller: Ignored because NEAT handles the network internally

        Returns:
        A list of actions for left, right, jump, shoot, and release.
        """
        # Normalize inputs (if required by NEAT or the environment)
        inputs = (inputs - min(inputs)) / float((max(inputs) - min(inputs)))

        # Use NEAT network's `activate` method to get the outputs
        output = self.neat_network.activate(inputs)

        # Convert output into discrete actions (binary decisions)
        left = 1 if output[0] > 0.5 else 0
        right = 1 if output[1] > 0.5 else 0
        jump = 1 if output[2] > 0.5 else 0
        shoot = 1 if output[3] > 0.5 else 0
        release = 1 if output[4] > 0.5 else 0

        # Return the actions
        return [left, right, jump, shoot, release]