import torch


class RolloutStorage(object):
    def __init__(self, num_steps, num_processes, obs_shape, action_dim):
        self.observations = torch.zeros(num_steps + 1, num_processes, *obs_shape)
        self.rewards = torch.zeros(num_steps, num_processes, 1)
        self.value_preds = torch.zeros(num_steps + 1, num_processes, 1)
        self.returns = torch.zeros(num_steps + 1, num_processes, 1)
        self.action_log_probs = torch.zeros(num_steps, num_processes, 1)
        self.actions = torch.zeros(num_steps, num_processes, action_dim)
        self.masks = torch.ones(num_steps + 1, num_processes, 1)
        self.bad_masks = torch.ones(num_steps + 1, num_processes, 1)

        self.num_steps = num_steps
        self.step = 0

    def to(self, device):
        self.observations = self.observations.to(device)
        self.rewards = self.rewards.to(device)
        self.value_preds = self.value_preds.to(device)
        self.returns = self.returns.to(device)
        self.action_log_probs = self.action_log_probs.to(device)
        self.actions = self.actions.to(device)
        self.masks = self.masks.to(device)
        self.bad_masks = self.bad_masks.to(device)

    def insert(
        self, current_obs, action, action_log_prob, value_pred, reward, mask, bad_mask
    ):
        self.observations[self.step + 1].copy_(current_obs)
        self.actions[self.step].copy_(action)
        self.action_log_probs[self.step].copy_(action_log_prob)
        self.value_preds[self.step].copy_(value_pred)
        self.rewards[self.step].copy_(reward)
        self.masks[self.step + 1].copy_(mask)
        self.bad_masks[self.step + 1].copy_(bad_mask)

        self.step = (self.step + 1) % self.num_steps

    def after_update(self):
        self.observations[0].copy_(self.observations[-1])
        self.masks[0].copy_(self.masks[-1])
        self.bad_masks[0].copy_(self.bad_masks[-1])

    def compute_returns(self, next_value, use_gae, gamma, gae_lambda):
        if use_gae:
            self.value_preds[-1] = next_value
            scaled_deltas = self.bad_masks[1:] * (
                self.rewards
                + gamma * self.value_preds[1:] * self.masks[1:]
                - self.value_preds[:-1]
            )
            scaled_masks = gamma * gae_lambda * self.masks[1:] * self.bad_masks[1:]
            gae = 0
            for step in reversed(range(self.rewards.size(0))):
                gae = scaled_deltas[step] + scaled_masks[step] * gae
                self.returns[step] = gae + self.value_preds[step]
        else:
            self.returns[-1] = next_value
            for step in reversed(range(self.rewards.size(0))):
                self.returns[step] = (
                    self.returns[step + 1] * gamma * self.masks[step + 1]
                    + self.rewards[step]
                ) * self.bad_masks[step + 1] + (
                    1 - self.bad_masks[step + 1]
                ) * self.value_preds[
                    step
                ]

    def feed_forward_generator(self, advantages, num_mini_batch):
        num_steps, num_processes = self.rewards.size()[0:2]
        obs_dim = self.observations.size(-1)
        act_dim = self.actions.size(-1)

        batch_size = num_processes * num_steps
        mini_batch_size = batch_size // num_mini_batch
        N = mini_batch_size * num_mini_batch

        device = self.rewards.device
        shuffled_indices = torch.randperm(N, generator=None, device=device)
        shuffled_indices_batch = shuffled_indices.view(num_mini_batch, -1)

        observations_shaped = self.observations.view(-1, obs_dim)
        actions_shaped = self.actions.view(-1, act_dim)
        value_preds_shaped = self.value_preds.view(-1, 1)
        returns_shaped = self.returns.view(-1, 1)
        masks_shaped = self.masks.view(-1, 1)
        action_log_probs_shaped = self.action_log_probs.view(-1, 1)
        advantages_shaped = advantages.view(-1, 1)

        for indices in shuffled_indices_batch:
            yield (
                observations_shaped[indices],
                actions_shaped[indices],
                value_preds_shaped[indices],
                returns_shaped[indices],
                masks_shaped[indices],
                action_log_probs_shaped[indices],
                advantages_shaped[indices],
            )
