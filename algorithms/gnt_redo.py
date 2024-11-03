import sys
import torch
from math import sqrt
import torch.nn.functional as F
from algorithms.adamgnt import AdamGnT
from torch import nn
import numpy as np

class GnTREDO(object):
    """
    Generate-and-Test algorithm for feed forward neural networks, based on maturity-threshold based replacement
    """
    def __init__(
            self,
            hidden_layers,
            hidden_activations,
            opt,
            threshold=0.01,
            reset_period=1000,
            device="cpu",
    ):
        super(GnTREDO, self).__init__()
        self.device = device
        assert len(hidden_layers) == len(hidden_activations) + 1

        self.hidden_layers = hidden_layers
        self.hidden_activations = hidden_activations

        self.opt = opt
        self.opt_type = 'sgd'
        if isinstance(self.opt, (AdamGnT, torch.optim.AdamW, torch.optim.Adam)):
            self.opt_type = 'adam'

        """
        Define the hyper-parameters of the algorithm
        """
        self.threshold = threshold
        self.reset_period = reset_period
        self.steps_since_last_redo = 0

        self.bounds = self.compute_bounds()

    def compute_bounds(self):
        bounds = [sqrt(1 / hidden_layer.in_features) for hidden_layer in self.hidden_layers[:-1]]
        return bounds

    def test_features(self, features):
        """
        Args:
            features: Activation values in the neural network, mini-batch * layer-idx * feature-idx
        Returns:
            Features to replace in each layer, Number of features to replace in each layer
        """
        features = features.mean(dim=0)
        features_to_replace = [None]*(len(self.hidden_layers)-1)
        num_features_to_replace = [None]*(len(self.hidden_layers)-1)
        for i in range(len(self.hidden_layers)-1):
            # Find features to replace
            feature_utility = features[i] / features[i].mean()
            new_features_to_replace = (feature_utility <= self.threshold).nonzero().reshape(-1)
            # Initialize utility for new features
            features_to_replace[i] = new_features_to_replace
            num_features_to_replace[i] = new_features_to_replace.shape[0]

        return features_to_replace, num_features_to_replace

    def gen_new_features(self, features_to_replace, num_features_to_replace):
        """
        Generate new features: Reset input and output weights for low utility features
        """
        with torch.no_grad():
            for i in range(len(self.hidden_layers)-1):
                if num_features_to_replace[i] == 0:
                    continue
                current_layer = self.hidden_layers[i]
                next_layer = self.hidden_layers[i+1]
                current_layer.weight.data[features_to_replace[i], :] *= 0.0
                current_layer.weight.data[features_to_replace[i], :] += \
                    torch.empty(num_features_to_replace[i], current_layer.in_features).uniform_(
                        -self.bounds[i], self.bounds[i]).to(self.device)
                nn.init.orthogonal_(
                    current_layer.weight.data[features_to_replace[i], :],
                    gain=nn.init.calculate_gain(self.hidden_activations[i])
                )
                current_layer.bias.data[features_to_replace[i]] *= 0

                next_layer.weight.data[:, features_to_replace[i]] = 0


    def update_optim_params(self, features_to_replace, num_features_to_replace):
        """
        Update Optimizer's state
        """
        if self.opt_type == 'adam':
            for i in range(len(self.hidden_layers)-1):
                # input weights
                if num_features_to_replace == 0:
                    continue
                self.opt.state[self.hidden_layers[i].weight]['exp_avg'][features_to_replace[i], :] = 0.0
                self.opt.state[self.hidden_layers[i].bias]['exp_avg'][features_to_replace[i]] = 0.0
                self.opt.state[self.hidden_layers[i].weight]['exp_avg_sq'][features_to_replace[i], :] = 0.0
                self.opt.state[self.hidden_layers[i].bias]['exp_avg_sq'][features_to_replace[i]] = 0.0
                self.opt.state[self.hidden_layers[i].weight]['step'].zero_()
                self.opt.state[self.hidden_layers[i].bias]['step'].zero_()
                # output weights
                self.opt.state[self.hidden_layers[i+1].weight]['exp_avg'][:, features_to_replace[i]] = 0.0
                self.opt.state[self.hidden_layers[i+1].weight]['exp_avg_sq'][:, features_to_replace[i]] = 0.0
                self.opt.state[self.hidden_layers[i+1].weight]['step'].zero_()

    def gen_and_test(self, features, only_test = False):
        """
        Perform generate-and-test
        :param features: activation of hidden units in the neural network
        """
        self.steps_since_last_redo += 1
        if self.steps_since_last_redo < self.reset_period:
            return 0, 0, 0

        features_to_replace, num_features_to_replace = self.test_features(features=features.abs())
        if not only_test:
            self.gen_new_features(features_to_replace, num_features_to_replace)
            self.update_optim_params(features_to_replace, num_features_to_replace)
        self.steps_since_last_redo = 0

        fraction_to_replace = sum(num_features_to_replace) / features.numel()

        return fraction_to_replace, sum(num_features_to_replace), fraction_to_replace
