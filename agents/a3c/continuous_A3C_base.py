"""
"""

import torch
import torch.nn as nn
from .utils import (normalized_columns_initializer, set_init, push_and_pull,
                    record)
import torch.nn.functional as F
from torch.autograd import Variable
import torch.multiprocessing as mp
from .shared_adam import SharedAdam
import math
import numpy as np
from ray.tune import Trainable
from ray.tune.registry import _global_registry, ENV_CREATOR

# Env related
import os
# sys.path.append(os.path.join(os.getcwd(),"../../"))

import datetime
import time
from tensorboardX import SummaryWriter


# Works only with Carla env with continous action space
class Net(nn.Module):
    def __init__(self, state_space, action_space):
        super(Net, self).__init__()

        self.action_space = action_space.shape[0]
        # Observation space for CarlaEnv is:
        # gym.spaces.Tuple(Box(84,84,6), Discrete(5), Box(2,))
        self.input_image_shape = state_space.spaces[0].shape  # 84x84x6
        self.input_image_size = np.product(self.input_image_shape)
        # 1 is for the discrete space in state_space.spaces[1]
        self.input_measurements_shape = 1 + state_space.spaces[2].shape[0]
        self.conv1 = nn.Conv2d(
            np.int(self.input_image_shape[2]), 32, 4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(32, 32, 2, stride=1, padding=1)
        self.linear = nn.Linear(32 * 12 * 12, 256)
        # Measurements will be concatenated with the input_image
        self.critic_linear = nn.Linear(256 + self.input_measurements_shape, 1)
        self.actor_mu = nn.Linear(256 + self.input_measurements_shape,
                                  self.action_space)
        self.actor_sigma = nn.Linear(256 + self.input_measurements_shape,
                                     self.action_space)
        # Init weights
        # self.apply(weights_init)
        set_init([
            self.conv1, self.conv2, self.conv3, self.conv4, self.linear,
            self.critic_linear, self.actor_mu, self.actor_sigma
        ])

        self.actor_mu.weight.data = normalized_columns_initializer(
            self.actor_mu.weight.data, 0.01)
        self.actor_mu.bias.data.fill_(0)

        self.actor_sigma.weight.data = normalized_columns_initializer(
            self.actor_sigma.weight.data, 0.1)
        self.actor_sigma.bias.data.fill_(0)

        self.critic_linear.weight.data = normalized_columns_initializer(
            self.critic_linear.weight.data, 1.0)
        self.critic_linear.bias.data.fill_(0)

        self.distribution = torch.distributions.Normal
        self.train()

    def forward(self, inputs):
        obs = inputs
        if len(inputs.shape) < 2:  # Single input; batch_size == 1
            obs = obs.unsqueeze(0)
        input_image = obs[:, :self.input_image_size].contiguous().view(
            -1, self.input_image_shape[2], self.input_image_shape[0],
            self.input_image_shape[1])
        input_measurements = obs[:, self.input_image_size:]

        x = F.relu(self.conv1(input_image))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.linear(x.view(-1, 32 * 12 * 12)))
        x = torch.cat((x, input_measurements), 1)

        actor_mu = F.tanh(self.actor_mu(x))  # Or clip to -1 & +1?
        actor_sigma = F.softplus(self.actor_sigma(x)) + 1e5
        critic_value = self.critic_linear(x)

        return actor_mu, actor_sigma, critic_value

    def choose_action(self, s):
        self.training = False
        mu, sigma, _ = self.forward(s)
        m = self.distribution(mean=mu.data, std=sigma.data)
        return m.sample().numpy()

    def loss_func(self, s, a, v_t):
        self.train()
        mu, sigma, values = self.forward(s)
        td = v_t - values
        c_loss = td.pow(2)

        m = self.distribution(mean=mu, std=sigma)
        log_prob = m.log_prob(a)
        entropy = 0.5 + 0.5 * math.log(2 * math.pi) + torch.log(m.std)
        exp_v = log_prob * td.detach() + 0.005 * entropy
        a_loss = -exp_v
        total_loss = (a_loss + c_loss).mean()
        return total_loss


class Worker(mp.Process):
    def __init__(self, gnet, opt, global_ep, global_ep_r, res_queue, name,
                 env_creator, config, state_dim, action_dim):
        super(Worker, self).__init__()
        self.name = 'w%i' % name
        self.config = config
        self.g_ep, self.g_ep_r, self.res_queue = (global_ep, global_ep_r,
                                                  res_queue)
        self.gnet, self.opt = gnet, opt
        self.lnet = Net(state_dim, action_dim)  # local network
        self.env = env_creator(self.config["env_config"])
        self.successful_episodes = 0
        self.mean_episode_len = 0
        self.total_step = 1
        now = datetime.datetime.now()
        self.summary_writer = SummaryWriter(
            os.path.expanduser(
                "~/tensorboard_log/continuous_a3c/{}_{}_{}_{}_{}_{}_{}".format(
                    now.year, now.month, now.day, now.hour, now.minute,
                    now.second, self.name)))

    def run(self):
        while self.g_ep.value < self.config["MAX_EP"]:
            s = self.env.reset()
            s = torch.from_numpy(
                np.concatenate((s[0].flatten(), np.array([s[1]]),
                                np.array(s[2]).flatten())))
            buffer_s, buffer_a, buffer_r = [], [], []
            ep_r = 0.
            times_sum = 0.0
            episode_len = self.config["MAX_EP_STEP"]
            for t in range(self.config["MAX_EP_STEP"]):
                start_time = time.time()
                a = self.lnet.choose_action(Variable(s).float())
                s_, r, done, py_measurements = self.env.step(a.squeeze().clip(
                    -1, 1))
                times_sum += time.time() - start_time
                # TensorboardX freezing here
                # self.summary_writer.add_scalar("Current Reward",
                # torch.DoubleTensor([r]),self.total_step)
                # self.summary_writer.add_scalar("Distance to Goal",
                # torch.DoubleTensor(
                # [py_measurements["dist_to_goal_euclidean"]]), self.total_step)
                s_ = torch.from_numpy(
                    np.concatenate((s_[0].flatten(), np.array([s_[1]]),
                                    np.array(s_[2]).flatten())))
                if t == self.config["MAX_EP_STEP"] - 1:
                    done = True
                ep_r += r
                buffer_a.append(a)
                buffer_s.append(s)
                buffer_r.append((r + 8.1) / 8.1)  # normalize

                if self.total_step % self.config["UPDATE_GLOBAL_ITER"] == 0 or \
                        done:  # update global and assign to local net
                    # sync
                    push_and_pull(self.opt, self.lnet, self.gnet, done, s_,
                                  buffer_s, buffer_a, buffer_r,
                                  self.config["gamma"])
                    buffer_s, buffer_a, buffer_r = [], [], []

                    if done:  # done and print information
                        record(self.g_ep, self.g_ep_r, ep_r, self.res_queue,
                               self.name)
                        episode_len = t
                        break
                s = s_
                self.total_step += 1

            if py_measurements["distance_to_goal_euclidean"] < 2.0:
                self.successful_episodes += 1

            self.summary_writer.add_scalar(
                "Num success episodes",
                torch.DoubleTensor(
                    [self.successful_episodes / self.g_ep.value]),
                self.g_ep.value)
            self.summary_writer.add_scalar(
                "Mean Reward", torch.DoubleTensor([ep_r / (t + 1)]),
                self.g_ep.value)
            self.summary_writer.add_scalar(
                "Mean Time Per Iteration in Seconds",
                torch.DoubleTensor([times_sum / (t + 1)]), self.g_ep.value)

            self.mean_episode_len += (
                (episode_len - self.mean_episode_len) / self.g_ep.value)
            self.summary_writer.add_scalar(
                "Mean Ep Len", torch.DoubleTensor([self.mean_episode_len]),
                self.total_step)
            self.res_queue.put(None)
        self.summary_writer.close()


class ContinuousA3CTune(Trainable):
    def _setup(self):
        self.env_creator = _global_registry.get(ENV_CREATOR, "carla_env")
        assert self.env_creator is not None
        self.env = self.env_creator(self.config["env_config"])
        self.N_S = self.env.observation_space
        self.N_A = self.env.action_space
        self.gnet = Net(self.N_S, self.N_A)  # global network
        if "load_checkpoint_path" in self.config and self.config[
                "load_checkpoint_path"] is not None:
            self.restore(self.config["load_checkpoint_path"])
        # share the global parameters in multiprocessing
        self.gnet.share_memory()
        # global optimizer
        self.opt = SharedAdam(self.gnet.parameters(), lr=0.0002)
        self.global_ep, self.global_ep_r, self.res_queue = (mp.Value('i', 0),
                                                            mp.Value('d', 0.),
                                                            mp.Queue())
        self.training_iteration = 0
        # parallel training
        self.workers = [
            Worker(self.gnet, self.opt, self.global_ep, self.global_ep_r,
                   self.res_queue, i, self.env_creator, self.config, self.N_S,
                   self.N_A) for i in range(self.config["num_local_workers"])
        ]  # mp.cpu_count())]
        [w.start() for w in self.workers]

    def _train(self):
        # res = []  # record episode reward to plot
        # while True:
        #    r = self.res_queue.get()
        #    if r is not None:
        #        res.append(r)
        #    else:
        #        break
        # [w.join() for w in workers]
        r = self.res_queue.get()
        if r is None:
            r = 0
        self.training_iteration += 1
        self._save(self.config["save_checkpoint_path"])
        # import matplotlib.pyplot as plt
        # plt.plot(res)
        # plt.ylabel('Moving average ep reward')
        # plt.xlabel('Step')
        # plt.show()
        return {"episode_reward_mean": r, "timesteps_this_iter": 1}

    def _save(self, checkpoint_dir):
        current_time = int(round(time.time() * 1000))
        torch.save(
            self.gnet.state_dict(), "{}global/{}_{}.pth".format(
                checkpoint_dir, self.training_iteration, current_time))

    def _restore(self, path):
        self.gnet.load_state_dict(torch.load(path))
