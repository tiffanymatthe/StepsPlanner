import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


class FixedNormal(Normal):
    def __init__(self, loc, scale, validate_args=False):
        self.loc, self.scale = loc, scale
        batch_shape = self.loc.size()
        super(Normal, self).__init__(batch_shape, validate_args=validate_args)

    def log_probs(self, actions):
        return super().log_prob(actions).sum(-1, keepdim=True)

    def entropy(self):
        return super().entropy().sum(-1)

    def mode(self):
        return self.mean


class DiagGaussian(nn.Module):
    def __init__(self, num_outputs):
        super(DiagGaussian, self).__init__()
        self.logstd = AddBias(torch.zeros(num_outputs))

    def forward(self, action_mean):
        #  An ugly hack for my KFAC implementation.
        zeros = torch.zeros_like(action_mean)
        # condition = self.logstd._bias.mean() >= -2.5
        action_std = self.logstd(zeros).clamp(-2.5).exp()
        return FixedNormal(action_mean, action_std)


class AddBias(nn.Module):
    def __init__(self, bias):
        super(AddBias, self).__init__()
        self._bias = nn.Parameter(bias.unsqueeze(0))

    def forward(self, x):
        return x + self._bias


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


init_s_ = lambda m: init(
    m,
    nn.init.orthogonal_,
    lambda x: nn.init.constant_(x, 0),
    nn.init.calculate_gain("sigmoid"),
)
init_r_ = lambda m: init(
    m,
    nn.init.orthogonal_,
    lambda x: nn.init.constant_(x, 0),
    nn.init.calculate_gain("relu"),
)
init_t_ = lambda m: init(
    m,
    nn.init.orthogonal_,
    lambda x: nn.init.constant_(x, 0),
    nn.init.calculate_gain("tanh"),
)


class Policy(nn.Module):
    def __init__(self, controller):
        super(Policy, self).__init__()
        self.actor = controller
        self.dist = DiagGaussian(controller.action_dim)

        state_dim = controller.state_dim
        h_size = 256
        self.critic = nn.Sequential(
            init_r_(nn.Linear(state_dim, h_size)),
            nn.ReLU(),
            init_r_(nn.Linear(h_size, h_size)),
            nn.ReLU(),
            init_r_(nn.Linear(h_size, h_size)),
            nn.ReLU(),
            init_r_(nn.Linear(h_size, h_size)),
            nn.ReLU(),
            init_r_(nn.Linear(h_size, 1)),
        )

        self.setup_feature_logging()

    def setup_feature_logging(self) -> None:
        self.to_log_features = False
        # Prepare for logging
        self.activations = {}
        self.feature_keys = [self.critic[i] for i in [1,3,5]]
        self.layers_to_check = [self.critic[i] for i in [0,2,4,6]]

        def hook_fn(m, i, o):
            if self.to_log_features:
                self.activations[m] = o

        for feature in self.feature_keys:
            feature.register_forward_hook(hook_fn)

    def get_activations(self,):
        return [self.activations[key] for key in self.feature_keys]

    def forward(self, inputs, states, masks):
        raise NotImplementedError

    def act(self, inputs, deterministic=False):
        action = self.actor(inputs)
        dist = self.dist(action)

        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()

        # action.clamp_(-1.0, 1.0)
        action_log_probs = dist.log_probs(action)

        value = self.critic(inputs)

        return value, action, action_log_probs

    def get_value(self, inputs):
        value = self.critic(inputs)
        return value

    def evaluate_actions(self, inputs, action, to_log_features=False):
        self.to_log_features = to_log_features
        self.actor.to_log_features = to_log_features

        value = self.critic(inputs)
        mode = self.actor(inputs)
        dist = self.dist(mode)

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        self.to_log_features = False
        self.actor.to_log_features = False

        return value, action_log_probs, dist_entropy


class SoftsignPolicy(Policy):
    def __init__(self, controller):
        super(SoftsignPolicy, self).__init__(controller)

        state_dim = controller.state_dim
        h_size = 256
        self.critic = nn.Sequential(
            init_s_(nn.Linear(state_dim, h_size)),
            nn.Softsign(),
            init_s_(nn.Linear(h_size, h_size)),
            nn.Softsign(),
            init_s_(nn.Linear(h_size, h_size)),
            nn.Softsign(),
            init_r_(nn.Linear(h_size, h_size)),
            nn.ReLU(),
            init_r_(nn.Linear(h_size, h_size)),
            nn.ReLU(),
            init_s_(nn.Linear(h_size, 1)),
        )


class SoftsignActor(nn.Module):
    """ Simple neural net actor that takes observation as input and outputs torques """

    def __init__(self, env):
        super(SoftsignActor, self).__init__()
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]

        h_size = 256
        self.net = nn.Sequential(
            init_s_(nn.Linear(self.state_dim, h_size)),
            nn.Softsign(),
            init_s_(nn.Linear(h_size, h_size)),
            nn.Softsign(),
            init_s_(nn.Linear(h_size, h_size)),
            nn.Softsign(),
            init_r_(nn.Linear(h_size, h_size)),
            nn.ReLU(),
            init_r_(nn.Linear(h_size, h_size)),
            nn.ReLU(),
            nn.Linear(h_size, self.action_dim),
        )

        self.setup_feature_logging()

    def setup_feature_logging(self) -> None:
        self.to_log_features = False
        # Prepare for logging
        self.activations = {}
        self.feature_keys = [self.net[i] for i in [1,3,5,7]]
        self.layers_to_check = [self.net[i] for i in [0,2,4,6,8]]

        def hook_fn(m, i, o):
            if self.to_log_features:
                self.activations[m] = o

        for feature in self.feature_keys:
            feature.register_forward_hook(hook_fn)

    def get_activations(self,):
        return [self.activations[key] for key in self.feature_keys]

    def forward(self, x):
        return torch.tanh(self.net(x))


class MixedActor(nn.Module):
    def __init__(
        self,
        env,
        num_experts=6,
    ):
        super().__init__()

        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        self.num_experts = num_experts

        self.robot_state_dim = env.robot.observation_space.shape[0]

        expert_input_size = self.robot_state_dim
        gate_input_size = self.state_dim
        output_size = self.action_dim
        hidden_size = 256

        self.layers = [
            (
                nn.Parameter(torch.empty(num_experts, expert_input_size, hidden_size)),
                # Avoid unsqueeze in the future, otherwise 1 has no meaning.
                nn.Parameter(torch.zeros(num_experts, 1, hidden_size)),
                F.softsign,
            ),
            (
                nn.Parameter(torch.empty(num_experts, hidden_size, hidden_size)),
                nn.Parameter(torch.zeros(num_experts, 1, hidden_size)),
                F.softsign,
            ),
            (
                nn.Parameter(torch.empty(num_experts, hidden_size, hidden_size)),
                nn.Parameter(torch.zeros(num_experts, 1, hidden_size)),
                F.softsign,
            ),
            (
                nn.Parameter(torch.empty(num_experts, hidden_size, hidden_size)),
                nn.Parameter(torch.zeros(num_experts, 1, hidden_size)),
                torch.relu,
            ),
            (
                nn.Parameter(torch.empty(num_experts, hidden_size, hidden_size)),
                nn.Parameter(torch.zeros(num_experts, 1, hidden_size)),
                torch.relu,
            ),
            (
                nn.Parameter(torch.empty(num_experts, hidden_size, output_size)),
                nn.Parameter(torch.zeros(num_experts, 1, output_size)),
                torch.tanh,
            ),
        ]

        for index, (weight, bias, activation) in enumerate(self.layers):

            # Initialize each expert separately
            if "softsign" in activation.__name__:
                gain = nn.init.calculate_gain("sigmoid")
            elif "relu" in activation.__name__:
                gain = nn.init.calculate_gain("relu")
            elif "tanh" in activation.__name__:
                gain = nn.init.calculate_gain("tanh")
            else:
                gain = 1.0

            for w in weight:
                nn.init.orthogonal_(w, gain=gain)

            # bias.data.fill_(0)
            self.register_parameter(f"w{index}", weight)
            self.register_parameter(f"b{index}", bias)

        # Gating network
        self.gate = nn.Sequential(
            init_r_(nn.Linear(gate_input_size, hidden_size)),
            nn.ELU(),
            init_r_(nn.Linear(hidden_size, hidden_size)),
            nn.ELU(),
            init_r_(nn.Linear(hidden_size, hidden_size)),
            nn.ELU(),
            init_r_(nn.Linear(hidden_size, num_experts)),
        )

    def forward(self, x):
        coefficients = F.softmax(self.gate(x), dim=1).t().unsqueeze(-1)
        out = x[:, : self.robot_state_dim]

        for (weight, bias, activation) in self.layers:
            out = activation(
                out.matmul(weight)  # (N, B, H), B = Batch, H = hidden
                .add(bias)  # (N, B, H)
                .mul(coefficients)  # (B, H)
                .sum(dim=0)
            )

            # Option 2: Need too much CUDA memory during backprop
            # But is slightly faster in inference mode
            # out = activation(
            #     out.unsqueeze(dim=1)
            #     .bmm((coefficients.unsqueeze(-1) * weight.unsqueeze(1)).sum(dim=0))
            #     .add((coefficients * bias).sum(dim=0).unsqueeze(1))
            #     .squeeze(1)
            # )

        return out

    def forward_with_activations(self, x):
        coefficients = F.softmax(self.gate(x), dim=1).t().unsqueeze(-1)
        out = x[:, : self.robot_state_dim]

        for (weight, bias, activation) in self.layers:
            out = activation(
                out.matmul(weight)  # (N, B, H), B = Batch, H = hidden
                    .add(bias)  # (N, B, H)
                    .mul(coefficients)  # (B, H)
                    .sum(dim=0)
            )

        return out, coefficients