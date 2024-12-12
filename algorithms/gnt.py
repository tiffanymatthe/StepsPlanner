import sys
import torch
from math import sqrt
import torch.nn.functional as F
from algorithms.adamgnt import AdamGnT
from torch import nn
import numpy as np

class GnT(object):
    """
    Generate-and-Test algorithm for feed forward neural networks, based on maturity-threshold based replacement
    """
    def __init__(
            self,
            hidden_layers,
            hidden_activations, # TODO how to use
            opt,
            decay_rate=0.99,
            replacement_rate=1e-4,
            device="cpu",
            maturity_threshold=20,
            util_type='contribution',
            accumulate=False,
    ):
        super(GnT, self).__init__()
        self.device = device
        assert len(hidden_layers) == len(hidden_activations) + 1
        self.accumulate = accumulate

        self.hidden_layers = hidden_layers
        self.hidden_activations = hidden_activations

        self.opt = opt
        self.opt_type = 'sgd'
        if isinstance(self.opt, (AdamGnT, torch.optim.AdamW, torch.optim.Adam)):
            self.opt_type = 'adam'

        """
        Define the hyper-parameters of the algorithm
        """
        self.replacement_rate = replacement_rate
        self.decay_rate = decay_rate
        self.maturity_threshold = maturity_threshold
        self.util_type = util_type

        """
        Utility of all features/neurons
        """
        self.reset()

        self.bounds = self.compute_bounds()

    def reset(self):
        self.util = [torch.zeros(hidden_layer.out_features).to(self.device) for hidden_layer in self.hidden_layers]
        self.bias_corrected_util = [torch.zeros(hidden_layer.out_features).to(self.device) for hidden_layer in self.hidden_layers]
        self.ages = [torch.zeros(hidden_layer.out_features).to(self.device) for hidden_layer in self.hidden_layers]
        self.m = torch.nn.Softmax(dim=1)
        self.mean_feature_act = [torch.zeros(hidden_layer.out_features).to(self.device) for hidden_layer in self.hidden_layers]
        self.accumulated_num_features_to_replace = [0 for hidden_layer in self.hidden_layers]

    def compute_bounds(self):
        bounds = [sqrt(1 / hidden_layer.in_features) for hidden_layer in self.hidden_layers]
        return bounds

    def update_utility(self, layer_idx=0, features=None, next_features=None):
        with torch.no_grad():
            self.util[layer_idx] *= self.decay_rate
            """
            Adam-style bias correction
            """
            bias_correction = 1 - self.decay_rate ** self.ages[layer_idx]

            self.mean_feature_act[layer_idx] *= self.decay_rate
            self.mean_feature_act[layer_idx] -= - (1 - self.decay_rate) * features.mean(dim=0)
            bias_corrected_act = self.mean_feature_act[layer_idx] / bias_correction

            current_layer = self.hidden_layers[layer_idx]
            next_layer = self.hidden_layers[layer_idx + 1]
            output_wight_mag = next_layer.weight.data.abs().mean(dim=0)
            input_wight_mag = current_layer.weight.data.abs().mean(dim=1)

            if self.util_type == 'weight':
                new_util = output_wight_mag
            elif self.util_type == 'contribution':
                new_util = output_wight_mag * features.abs().mean(dim=0)
            elif self.util_type == 'adaptation':
                new_util = 1/input_wight_mag
            elif self.util_type == 'zero_contribution':
                new_util = output_wight_mag * (features - bias_corrected_act).abs().mean(dim=0)
            elif self.util_type == 'adaptable_contribution':
                new_util = output_wight_mag * (features - bias_corrected_act).abs().mean(dim=0) / input_wight_mag
            elif self.util_type == 'feature_by_input':
                input_wight_mag = self.net[layer_idx*2].weight.data.abs().mean(dim=1)
                new_util = (features - bias_corrected_act).abs().mean(dim=0) / input_wight_mag
            else:
                new_util = 0

            self.util[layer_idx] += (1 - self.decay_rate) * new_util

            """
            Adam-style bias correction
            """
            self.bias_corrected_util[layer_idx] = self.util[layer_idx] / bias_correction

            if self.util_type == 'random':
                self.bias_corrected_util[layer_idx] = torch.rand(self.util[layer_idx].shape)

    def test_features(self, features):
        """
        Args:
            features: Activation values in the neural network
        Returns:
            Features to replace in each layer, Number of features to replace in each layer
        """
        # -1 since we don't want to replace features in outgoing layer
        features_to_replace = [torch.empty(0, dtype=torch.long).to(self.device) for _ in range(len(self.hidden_layers)-1)]
        num_features_to_replace = [0 for _ in range(len(self.hidden_layers) - 1)]
        num_eligible_features = [0 for _ in range(len(self.hidden_layers) - 1)]
        if self.replacement_rate == 0:
            return features_to_replace, num_features_to_replace, num_eligible_features, 0, 0

        dead_neurons = torch.zeros(len(features), dtype=torch.float32)
        total_number = 0
        threshold = 1e-2
        max_bias = 0

        for i in range(len(self.hidden_layers)-1):
            self.ages[i] += 1
            """
            Update feature utility
            """
            self.update_utility(layer_idx=i, features=features[i])
            """
            Find the no. of features to replace
            """

            eligible_feature_indices = torch.where(self.ages[i] > self.maturity_threshold)[0]
            num_eligible_features[i] = eligible_feature_indices.shape[0]

            score = features[i].abs().mean(dim=0)
            normalized_score = score / (score.mean() + 1e-9)
            dead_neurons[i] = (normalized_score < threshold).sum()
            total_number += features[i].shape[1]

            if eligible_feature_indices.shape[0] == 0:
                continue
            num_new_features_to_replace = self.replacement_rate*eligible_feature_indices.shape[0]
            self.accumulated_num_features_to_replace[i] += num_new_features_to_replace

            """
            Case when the number of features to be replaced is between 0 and 1.
            """
            if self.accumulate:
                num_new_features_to_replace = int(self.accumulated_num_features_to_replace[i])
                self.accumulated_num_features_to_replace[i] -= num_new_features_to_replace
            else:
                if num_new_features_to_replace < 1:
                    if torch.rand(1) <= num_new_features_to_replace:
                        num_new_features_to_replace = 1
                num_new_features_to_replace = int(num_new_features_to_replace)
    
            if num_new_features_to_replace == 0:
                continue

            """
            Find features to replace in the current layer
            """
            neg_bias, new_features_to_replace = torch.topk(-self.bias_corrected_util[i][eligible_feature_indices],
                                                 num_new_features_to_replace)
            new_features_to_replace = eligible_feature_indices[new_features_to_replace]

            max_bias = max(-neg_bias[-1], max_bias)

            """
            Initialize utility for new features
            """
            self.util[i][new_features_to_replace] = 0
            self.mean_feature_act[i][new_features_to_replace] = 0.

            features_to_replace[i] = new_features_to_replace
            num_features_to_replace[i] = num_new_features_to_replace
        
        dormant_fraction = dead_neurons.sum().item() / total_number

        return features_to_replace, num_features_to_replace, num_eligible_features, dormant_fraction, max_bias

    def gen_new_features(self, features_to_replace, num_features_to_replace):
        """
        Generate new features: Reset input and output weights for low utility features
        """
        with torch.no_grad():
            for i in range(len(self.hidden_layers) - 1):
                if num_features_to_replace[i] == 0:
                    continue
                current_layer = self.hidden_layers[i]
                next_layer = self.hidden_layers[i + 1]

                # print(f"Before replacement for layer {i}")
                # print(current_layer.weight.data[features_to_replace[i], :])
                # print(current_layer.bias.data[features_to_replace[i]])
                # print(next_layer.weight.data[:, features_to_replace[i]])
                # print(next_layer.bias.data)

                current_layer.weight.data[features_to_replace[i], :] *= 0.0
                current_layer.weight.data[features_to_replace[i], :] += \
                    torch.empty(num_features_to_replace[i], current_layer.in_features).uniform_(
                        -self.bounds[i], self.bounds[i]).to(self.device)
                
                # print(current_layer.weight.data[features_to_replace[i], :])
                current_layer.weight.data[features_to_replace[i], :] = nn.init.orthogonal_(
                    current_layer.weight.data[features_to_replace[i], :],
                    gain=nn.init.calculate_gain(self.hidden_activations[i])
                )
                current_layer.bias.data[features_to_replace[i]] = 0

                """
                # Update bias to correct for the removed features and set the outgoing weights and ages to zero
                """
                next_layer.bias.data += (next_layer.weight.data[:, features_to_replace[i]] * \
                                                self.mean_feature_act[i][features_to_replace[i]] / \
                                                (1 - self.decay_rate ** self.ages[i][features_to_replace[i]])).sum(dim=1)
                next_layer.weight.data[:, features_to_replace[i]] = 0
                self.ages[i][features_to_replace[i]] = 0

                # print("After replacement")
                # print(current_layer.weight.data[features_to_replace[i], :])
                # print(current_layer.bias.data[features_to_replace[i]])
                # print(f"next layer {i + 1} replaced")
                # print(next_layer.weight.data[:, features_to_replace[i]])
                # print(next_layer.bias.data)


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

    def compute_average_weight_magnitude(self):
        total_magnitude = 0
        total_weights = 0
        
        for layer in self.hidden_layers:
            # Check if the layer has parameters
            if isinstance(layer, torch.nn.Module):
                for param in layer.parameters():
                    if param.requires_grad:  # Only consider trainable parameters
                        total_magnitude += param.abs().sum().item()
                        total_weights += param.numel()
        
        return total_magnitude / total_weights if total_weights > 0 else 0

    def gen_and_test(self, features, only_test = False):
        """
        Perform generate-and-test
        :param features: activation of hidden units in the neural network
        """
        if not isinstance(features, list):
            print('features passed to generate-and-test should be a list')
            sys.exit()
        features_to_replace, num_features_to_replace, num_eligible_features, dormant_fraction, max_bias = self.test_features(features=features)
        if not only_test:
            self.gen_new_features(features_to_replace, num_features_to_replace)
            self.update_optim_params(features_to_replace, num_features_to_replace)

        num_features_to_replace = np.sum(np.array(num_features_to_replace))
        num_eligible_features = np.sum(np.array(num_eligible_features))

        return dormant_fraction, self.compute_average_weight_magnitude()
