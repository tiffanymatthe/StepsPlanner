import torch
import time
import numpy as np
from collections import deque
import copy
import torch.nn.functional as F
from bottleneck import nanmean
from common.envs_utils import make_env, make_vec_envs
from common.controller import SoftsignActor, Policy
import itertools
from common.csv_utils import CSVLogger

def list_string(array):
    return ', '.join(f"{num:.2f}" for num in array)

class Distiller:
    def __init__(self, env_name, base_env_kwargs, seed, device, num_processes, num_experts, num_epochs, dummy_env, log_dir):
        env_kwargs = {
            **base_env_kwargs,
            "determine": True,
        }

        self.csv_logger = CSVLogger(log_dir=log_dir, filename="plaid.csv")
        self.log_dir = log_dir

        num_processes = 10 # overwrite
        
        self.num_experts = num_experts

        self.envs_per_task = [
            # make_env(env_name, seed=seed, **env_kwargs)
            make_vec_envs(
                env_name, seed, num_processes, None, **env_kwargs
            )
            for i in range(self.num_experts)
        ]

        obs_shape = self.envs_per_task[0].observation_space.shape
        obs_shape = (obs_shape[0], *obs_shape[1:])
        obs_dim = obs_shape[0]
        act_dim = self.envs_per_task[0].action_space.shape[0]

        # order: current, previous

        self.num_steps_per_task = [5000 for _ in range(self.num_experts)]
        self.num_epochs = num_epochs
    
        self.buffer_observations_per_task = [torch.zeros(self.num_steps_per_task[i] * num_epochs + 1, num_processes, *obs_shape, device="cpu").pin_memory() for i in range(self.num_experts)]
        self.buffer_expert_actions_per_task = [torch.zeros(self.num_steps_per_task[i] * num_epochs, num_processes, act_dim, device="cpu").pin_memory() for i in range(self.num_experts)]
        self.buffer_expert_values_per_task = [torch.zeros(self.num_steps_per_task[i] * num_epochs, num_processes, act_dim, device="cpu").pin_memory() for i in range(self.num_experts)]

        self.dummy_env = dummy_env

        self.device = device
        self.num_processes = num_processes

    def distill_policies(self, policies, behavior_curriculums):

        assert len(policies) == len(behavior_curriculums) and len(policies) == self.num_experts

        env_kwargs_per_task = [
            {
                "start_curriculum": 9,
                "start_behavior_curriculum": behavior_curriculums[i],
                "curriculum": 9,
                "behavior_curriculum": behavior_curriculums[i],
                "determine": False,
            } for i in range(self.num_experts)
        ]

        for i, env in enumerate(self.envs_per_task):
            env.set_env_params(env_kwargs_per_task[i])

        return self.train(
            policies,
            self.envs_per_task,
            env_kwargs_per_task,
            num_epochs=self.num_epochs,
            device=self.device,
            num_processes=self.num_processes,
        )

    def train(
        self,
        expert_policies,
        envs_per_task,
        env_per_task_kwargs, # list of kwargs
        num_epochs=20,
        mini_batch_size=512,
        num_processes=4,
        device="cuda:0",
    ) -> None:
        with torch.no_grad():
            controller = SoftsignActor(self.dummy_env)
            student_policy = Policy(controller)
            student_policy.load_state_dict(copy.deepcopy(expert_policies[-1].state_dict()))
            student_policy.to(device)
               
        optimizer = torch.optim.Adam(student_policy.parameters(), lr=3e-4)

        obs_shape = envs_per_task[0].observation_space.shape
        obs_shape = (obs_shape[0], *obs_shape[1:])
        obs_dim = obs_shape[0]
        act_dim = envs_per_task[0].action_space.shape[0]

        prev_ep_action_loss = 0
        same_action_loss_count = 0

        BC_epochs = 50 # behavior cloning only epochs

        start = time.time()
        for epoch in range(num_epochs):
            observations_shaped_per_task = [None for _ in range(self.num_experts)]
            expert_actions_shaped_per_task = [None for _ in range(self.num_experts)]
            expert_values_shaped_per_task = [None for _ in range(self.num_experts)]
            shuffled_indices_batch_per_task = [None for _ in range(self.num_experts)]
            curriculum_metric_per_task = [None for _ in range(self.num_experts)]
            timing_met_per_task = [None for _ in range(self.num_experts)]
            dist_err_per_task = [None for _ in range(self.num_experts)]
            heading_err_per_task = [None for _ in range(self.num_experts)]
            use_expert_min_threshold = 0 # 0 if epoch < BC_epochs else min((epoch-BC_epochs) / 100, 1)
            deterministic_max_threshold = max(0, (epoch - BC_epochs) / (num_epochs - BC_epochs))
            for task_i in range(self.num_experts):
                max_episodes = int(self.num_processes * self.num_steps_per_task[task_i])
                episode_rewards = deque(maxlen=max_episodes)
                curriculum_metrics = [deque(maxlen=max_episodes) for _ in range(4)]
                avg_heading_errs = [deque(maxlen=max_episodes) for _ in range(4)]
                avg_dist_errs = [deque(maxlen=max_episodes) for _ in range(4)]
                avg_timing_mets = [deque(maxlen=max_episodes) for _ in range(4)]
                if epoch == 0:
                    obs = envs_per_task[task_i].reset()
                    self.buffer_observations_per_task[task_i][epoch * self.num_steps_per_task[task_i]].copy_(torch.from_numpy(obs))
                with torch.no_grad():
                    for step in range(self.num_steps_per_task[task_i]):
                        buffer_index = step + epoch * self.num_steps_per_task[task_i]
                        expert_value, expert_action, _ = expert_policies[task_i].act(
                            self.buffer_observations_per_task[task_i][buffer_index].to(device), deterministic=True
                        )

                        use_expert = True # np.random.rand() > use_expert_min_threshold

                        if not use_expert:
                            # determines if we get observations from the student or teacher, but reference data is from teacher for MSE loss calc
                            _, student_action, _ = student_policy.act(self.buffer_observations_per_task[task_i][buffer_index].to(device), deterministic=(np.random.rand() < deterministic_max_threshold))

                        if use_expert:
                            cpu_actions = expert_action.cpu().numpy()
                        else:
                            cpu_actions = student_action.cpu().numpy()
                        obs, _, dones, infos = envs_per_task[task_i].step(cpu_actions)

                        masks = torch.FloatTensor(~dones).unsqueeze(1)
                        # num_dones += (num_processes - torch.count_nonzero(masks))
                        bad_masks = torch.ones((self.num_processes, 1))
                        for p_index, info in enumerate(infos):
                            # This information is added by common.envs_utils.TimeLimitMask
                            if "bad_transition" in info:
                                bad_masks[p_index] = 0.0
                            # This information is added by common.envs_utils.Monitor
                            if "episode" in info:
                                episode_rewards.append(info["episode"]["r"])
                            if "curriculum_metric" in info:
                                curriculum_metrics[info["mask_combo_id"]].append(info["curriculum_metric"])
                            if "avg_heading_err" in info:
                                avg_heading_errs[info["mask_combo_id"]].append(info["avg_heading_err"])
                            if "avg_timing_met" in info:
                                avg_timing_mets[info["mask_combo_id"]].append(info["avg_timing_met"])
                            if "avg_dist_err" in info:
                                avg_dist_errs[info["mask_combo_id"]].append(info["avg_dist_err"])


                        self.buffer_observations_per_task[task_i][buffer_index + 1].copy_(torch.from_numpy(obs))
                        self.buffer_expert_actions_per_task[task_i][buffer_index].copy_(expert_action.to(device))
                        self.buffer_expert_values_per_task[task_i][buffer_index].copy_(expert_value.to(device))

                batch_size = self.num_steps_per_task[task_i] * (epoch + 1) * num_processes
                num_mini_batch = batch_size // mini_batch_size
                shuffled_indices = torch.randperm(
                    num_mini_batch * mini_batch_size, generator=None, device="cpu"
                )
                shuffled_indices_batch_per_task[task_i] = shuffled_indices.view(num_mini_batch, -1)

                observations_shaped_per_task[task_i] = self.buffer_observations_per_task[task_i].view(-1, obs_dim)
                expert_actions_shaped_per_task[task_i] = self.buffer_expert_actions_per_task[task_i].view(-1, act_dim)
                expert_values_shaped_per_task[task_i] = self.buffer_expert_values_per_task[task_i].view(-1, 1)

                curriculum_metric_per_task[task_i] = [nanmean(x) for x in curriculum_metrics[0:4]]
                timing_met_per_task[task_i] = [nanmean(x) for x in avg_timing_mets[0:4]]
                dist_err_per_task[task_i] = [nanmean(x) for x in avg_dist_errs[0:4]]
                heading_err_per_task[task_i] = [nanmean(x) for x in avg_heading_errs[0:4]]

                print(
                    (
                        f"Epoch {epoch+1:4d}/{num_epochs:4d} | "
                        f"env {task_i} | "
                        f"curriculum_metric {list_string(curriculum_metric_per_task[task_i])} | "
                        f"avg_heading_err {list_string(heading_err_per_task[task_i])} | "
                        f"avg_timing_met {list_string(timing_met_per_task[task_i])} | "
                        f"avg_dist_err {list_string(dist_err_per_task[task_i])} | "
                    )
                )

            ep_action_loss = torch.tensor(0.0, device=device).float()
            ep_value_loss = torch.tensor(0.0, device=device).float()

            for batch_tuples in itertools.zip_longest(*shuffled_indices_batch_per_task):
                optimizer.zero_grad()
                for task_i, indices in enumerate(batch_tuples):
                    if indices is None:  # This task has no more batches
                        continue

                    observations_batch = observations_shaped_per_task[task_i][indices].to(device, non_blocking=True)
                    actions_batch = expert_actions_shaped_per_task[task_i][indices].to(device, non_blocking=True)
                    values_batch = expert_values_shaped_per_task[task_i][indices].to(device, non_blocking=True)

                    pred_actions = student_policy.actor(observations_batch)
                    pred_values = student_policy.get_value(observations_batch)

                    action_loss = F.mse_loss(pred_actions, actions_batch)
                    value_loss = F.mse_loss(pred_values, values_batch)

                    (action_loss + value_loss).backward()

                    ep_action_loss.add_(action_loss.detach())
                    ep_value_loss.add_(value_loss.detach())

                optimizer.step()

            L = shuffled_indices_batch_per_task[0].shape[0] + shuffled_indices_batch_per_task[1].shape[0]
            ep_action_loss.div_(L)
            ep_value_loss.div_(L)

            elapsed_time = time.time() - start

            if epoch % 50 == 0:
                torch.save(student_policy.state_dict(), f"{self.log_dir}/plaid_distilled_epoch_{epoch}.pt")

            self.csv_logger.log_epoch({
                "prev_expert_task": f"{env_per_task_kwargs[1]['behavior_curriculum']}",
                "prev_expert_curriculum": f"{env_per_task_kwargs[1]['curriculum']}",
                "expert_task": f"{env_per_task_kwargs[0]['behavior_curriculum']}",
                "expert_curriculum": f"{env_per_task_kwargs[0]['curriculum']}",
                "epoch": epoch + 1,
                "elapsed_time": elapsed_time,
                "action_loss": ep_action_loss.item(),
                "value_loss": ep_value_loss.item(),
                "curriculum_metric": curriculum_metric_per_task[0],
                "curriculum_metric_prev": curriculum_metric_per_task[1],
                "timing_met": timing_met_per_task[0],
                "timing_met_prev": timing_met_per_task[1],
                "dist_err": dist_err_per_task[0],
                "dist_err_prev": dist_err_per_task[1],
                "heading_err": heading_err_per_task[0],
                "heading_err_prev": heading_err_per_task[1],
                "use_expert_min_threshold": use_expert_min_threshold,
                "deterministic_max_threshold": deterministic_max_threshold,
            })

            print(
                (
                    f"Epoch {epoch+1:4d}/{num_epochs:4d} | "
                    f"Elapsed Time {elapsed_time:8.2f} |"
                    f"Action Loss: {ep_action_loss.item():8.5f} | "
                    f"Value Loss: {ep_value_loss.item():8.4f} | "
                    f"Use Expert: rand > {use_expert_min_threshold:.4f} | "
                    f"Deterministic Student: rand < {deterministic_max_threshold:.4f} | "
                )
            )

            if ep_action_loss.item() <= 0.01 and epoch > 100:
                print("Quitting early.")
                break

            if epoch > 20 and abs(ep_action_loss.item() - prev_ep_action_loss) <= 0.00001:
                # do not update prev action loss
                same_action_loss_count += 1
                if same_action_loss_count > 10:
                    print("Quitting early.")
                    break
            else:
                prev_ep_action_loss = ep_action_loss.item()
                same_action_loss_count = 0

        return student_policy