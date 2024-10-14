import os

current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
os.sys.path.insert(0, parent_dir)

import torch
import torch.nn as nn
import torch.optim as optim

from algorithms.gnt import GnT
# from algorithms.adamgnt import AdamGnT

def clip_grad_norm_(parameters, max_norm):
    total_norm = torch.cat([p.grad.detach().view(-1) for p in parameters]).norm()
    clip_coef = (max_norm / (total_norm + 1e-6)).clamp(max=1.0)
    for p in parameters:
        p.grad.detach().mul_(clip_coef)
    return total_norm


class PPO(object):
    def __init__(
        self,
        actor_critic,
        clip_param,
        ppo_epoch,
        num_mini_batch,
        value_loss_coef,
        entropy_coef,
        lr=None,
        eps=None,
        max_grad_norm=None,
        use_clipped_value_loss=True,
        mirror_function=None,
    ):
        self.actor_critic = actor_critic

        self.clip_param = clip_param
        self.ppo_epoch = ppo_epoch
        self.num_mini_batch = num_mini_batch

        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef

        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

        self.mirror_function = mirror_function

        self.optimizer = optim.AdamW(
            actor_critic.parameters(),
            lr=lr,
            weight_decay=5e-4,
            eps=eps,
            betas=(0.99,0.99)
        )

        # settings based on https://github.com/shibhansh/loss-of-plasticity/blob/7bf3dfe6723a43a543fa1057a38eaf4b480f2ff3/lop/rl/cfg/walker/cbp.yml

        self.actor_gnt = GnT(
            hidden_layers=self.actor_critic.actor.layers_to_check,
            hidden_activations=["relu"],
            opt=self.optimizer,
            replacement_rate=1e-4,
            decay_rate=0.99,
            maturity_threshold=10000,
            util_type="contribution",
            device="cuda:0" if torch.cuda.is_available() else "cpu",
            # accumulate=accumulate,
        )

        self.critic_gnt = GnT(
            hidden_layers=self.actor_critic.layers_to_check,
            hidden_activations=["relu", "relu", "relu"],
            opt=self.optimizer,
            replacement_rate=1e-4,
            decay_rate=0.99,
            maturity_threshold=10000,
            util_type="contribution",
            device="cuda:0" if torch.cuda.is_available() else "cpu",
            # accumulate=accumulate,
        )

    def update(self, rollouts):
        advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)

        device = advantages.device
        value_loss_epoch = torch.tensor(0.0).to(device)
        action_loss_epoch = torch.tensor(0.0).to(device)
        dist_entropy_epoch = torch.tensor(0.0).to(device)
        critic_fraction_to_replace_epoch = torch.tensor(0.0).to(device)
        actor_fraction_to_replace_epoch = torch.tensor(0.0).to(device)

        clip_param = self.clip_param

        parameters = [
            p for p in self.actor_critic.parameters() if p.requires_grad is not None
        ]
        assert len(parameters) != 0, "No trainable parameters"

        for e in range(self.ppo_epoch):
            data_generator = rollouts.feed_forward_generator(
                advantages, self.num_mini_batch
            )

            for sample in data_generator:
                if self.mirror_function is not None:
                    (
                        observations_batch,
                        actions_batch,
                        value_preds_batch,
                        return_batch,
                        masks_batch,
                        old_action_log_probs_batch,
                        adv_targ,
                    ) = self.mirror_function(sample)
                else:
                    (
                        observations_batch,
                        actions_batch,
                        value_preds_batch,
                        return_batch,
                        masks_batch,
                        old_action_log_probs_batch,
                        adv_targ,
                    ) = sample

                (
                    values,
                    action_log_probs,
                    dist_entropy,
                ) = self.actor_critic.evaluate_actions(
                    observations_batch, actions_batch, to_log_features=True
                )

                ratio = (action_log_probs - old_action_log_probs_batch).exp()
                surr1 = ratio * adv_targ
                surr2 = ratio.clamp(1.0 - clip_param, 1.0 + clip_param) * adv_targ
                action_loss = -torch.min(surr1, surr2).mean()

                if self.use_clipped_value_loss:
                    value_pred_clipped = value_preds_batch + (
                        values - value_preds_batch
                    ).clamp(-clip_param, clip_param)
                    value_losses = (values - return_batch).pow(2)
                    value_losses_clipped = (value_pred_clipped - return_batch).pow(2)
                    value_loss = (
                        0.5 * torch.max(value_losses, value_losses_clipped).mean()
                    )
                else:
                    value_loss = 0.5 * (return_batch - values).pow(2).mean()

                self.optimizer.zero_grad()
                (
                    value_loss * self.value_loss_coef
                    + action_loss
                    - dist_entropy * self.entropy_coef
                ).backward()
                clip_grad_norm_(parameters, self.max_grad_norm)
                self.optimizer.step()

                # continual backprop (wipe dormant neurons)
                self.optimizer.zero_grad()
                critic_fraction_to_replace = self.critic_gnt.gen_and_test(features=self.actor_critic.get_activations() + [None])
                actor_fraction_to_replace = self.actor_gnt.gen_and_test(features=self.actor_critic.actor.get_activations() + [None])
                critic_fraction_to_replace = 0
                actor_fraction_to_replace = 0

                value_loss_epoch.add_(value_loss.detach())
                action_loss_epoch.add_(action_loss.detach())
                dist_entropy_epoch.add_(dist_entropy.detach())
                critic_fraction_to_replace_epoch.add_(critic_fraction_to_replace)
                actor_fraction_to_replace_epoch.add_(actor_fraction_to_replace)

        num_updates = self.ppo_epoch * self.num_mini_batch

        value_loss_epoch.div_(num_updates)
        action_loss_epoch.div_(num_updates)
        dist_entropy_epoch.div_(num_updates)
        critic_fraction_to_replace_epoch.div_(num_updates)
        actor_fraction_to_replace_epoch.div_(num_updates)

        return (
            value_loss_epoch.item(),
            action_loss_epoch.item(),
            dist_entropy_epoch.item(),
            critic_fraction_to_replace_epoch.item(),
            actor_fraction_to_replace_epoch.item()
        )
