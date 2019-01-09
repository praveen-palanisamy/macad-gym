"""
TDAC agent for Carla cont.
"""
import os
import glob
import datetime
import time
import math
import json

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.multiprocessing as mp
from tensorboardX import SummaryWriter

from .utils import normalized_columns_initializer, set_init, push_and_pull, \
    record
from .shared_adam import SharedAdam
from env.carla.multi_env import MultiCarlaEnv, DEFAULT_MULTIENV_CONFIG
from env.carla.scenarios import update_scenarios_parameter

# TODO: Move to actor config or argps
LOG_DIR = os.path.expanduser("~/tensorboard_logs")
MODEL_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "saved_models")

env_config = DEFAULT_MULTIENV_CONFIG
config_update = update_scenarios_parameter(
    json.load(open("agents/TDAC/env_config.json")))
env_config.update(config_update)

vehicle_name = next(iter(env_config['actors'].keys()))

env = MultiCarlaEnv(env_config)
N_S = env.observation_space
N_A = env.action_space

UPDATE_GLOBAL_ITER = 5
GAMMA = 0.9
MAX_EP = 10000000  # 10M
SAVE_STEP = 100000

save_model_dir = os.path.expanduser(MODEL_DIR)
if not os.path.exists(os.path.join(save_model_dir, "global")):
    os.makedirs(os.path.join(save_model_dir, "global"))
if not os.path.exists(os.path.join(save_model_dir, "local")):
    os.makedirs(os.path.join(save_model_dir, "local"))


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
        m = self.distribution(mu.data, sigma.data)
        return m.sample().numpy()

    def loss_func(self, s, a, v_t):
        self.train()
        mu, sigma, values = self.forward(s)
        td = v_t - values
        c_loss = td.pow(2)

        m = self.distribution(mu, sigma)
        log_prob = m.log_prob(a)
        entropy = 0.5 + 0.5 * math.log(2 * math.pi) + torch.log(m.scale)
        exp_v = log_prob * td.detach() + 0.005 * entropy
        a_loss = -exp_v
        total_loss = (a_loss + c_loss).mean()
        return total_loss


class Worker(mp.Process):
    def __init__(self, gnet, opt, global_ep, global_ep_r, res_queue, name):
        super(Worker, self).__init__()
        self.name = 'w%i' % name
        self.g_ep, self.g_ep_r, self.res_queue = (global_ep, global_ep_r,
                                                  res_queue)
        self.gnet, self.opt = gnet, opt
        self.lnet = Net(N_S, N_A)  # local network
        self.env = MultiCarlaEnv(env_config)

    def run(self):
        last_checkpoint = max(
            glob.glob(save_model_dir + "/local/*"),
            key=os.path.getctime,
            default=None)
        if last_checkpoint:
            self.lnet.load_state_dict(torch.load(last_checkpoint))
            print("Loaded saved local model:", last_checkpoint)

        successful_episodes = 0
        mean_episode_len = 0
        total_step = 1
        now = datetime.datetime.now()
        summary_writer = SummaryWriter(
            os.path.join(
                LOG_DIR, "continuous_a3c/{}_{}_{}_{}_{}_{}_{}".format(
                    now.year, now.month, now.day, now.hour, now.minute,
                    now.second, self.name)))
        while self.g_ep.value < MAX_EP:
            state = self.env.reset()[vehicle_name]
            state = torch.from_numpy(
                np.concatenate((state[0].flatten(), np.array([state[1]]),
                                np.array(state[2]).flatten())))
            buffer_s, buffer_a, buffer_r = [], [], []
            ep_r = 0.
            times_sum = 0.0
            done = False
            t = 0
            while not done:
                # for t in range(MAX_EP_STEP):
                t += 1  # Step num
                start_time = time.time()
                action = self.lnet.choose_action(Variable(state).float())
                next_state, reward, done, py_measurements = self.env.step({
                    vehicle_name:
                    action.squeeze().clip(-1, 1)
                })
                reward = reward[vehicle_name]
                next_state = next_state[vehicle_name]
                done = done[vehicle_name]
                py_measurements = py_measurements[vehicle_name]

                times_sum += time.time() - start_time
                summary_writer.add_scalar("Current_Reward",
                                          torch.DoubleTensor([reward]),
                                          total_step)
                summary_writer.add_scalar(
                    "Distance_to_Goal",
                    torch.DoubleTensor(
                        [py_measurements["distance_to_goal_euclidean"]]),
                    total_step)
                next_state = torch.from_numpy(
                    np.concatenate((next_state[0].flatten(),
                                    np.array([next_state[1]]),
                                    np.array(next_state[2]).flatten())))
                # if t == MAX_EP_STEP - 1:
                #     done = True
                ep_r += reward
                buffer_a.append(action)
                buffer_s.append(state)
                buffer_r.append((reward + 8.1) / 8.1)  # normalize

                # update global and assign to local net
                if total_step % UPDATE_GLOBAL_ITER == 0 or done:
                    # sync
                    push_and_pull(self.opt, self.lnet, self.gnet, done,
                                  next_state, buffer_s, buffer_a, buffer_r,
                                  GAMMA)
                    buffer_s, buffer_a, buffer_r = [], [], []

                    if done:  # done and print information
                        record(self.g_ep, self.g_ep_r, ep_r, self.res_queue,
                               self.name)
                        episode_len = t
                        break
                state = next_state
                if total_step % SAVE_STEP == 0:
                    current_time = int(round(time.time() * 1000))
                    torch.save(
                        self.lnet.state_dict(), "{}/local/{}_{}.pt".format(
                            save_model_dir, total_step, current_time))
                    torch.save(
                        self.gnet.state_dict(), "{}/global/{}_{}.pt".format(
                            save_model_dir, total_step, current_time))
                total_step += 1

            if py_measurements["distance_to_goal_euclidean"] < 2.0:
                successful_episodes += 1

            summary_writer.add_scalar(
                "Num_of_Successfully_Completed_Episodes",
                torch.DoubleTensor([successful_episodes / self.g_ep.value]),
                self.g_ep.value)
            summary_writer.add_scalar("Mean_Reward",
                                      torch.DoubleTensor([ep_r / (t + 1)]),
                                      self.g_ep.value)
            summary_writer.add_scalar(
                "Mean_Time_Per_Iteration_in_Seconds",
                torch.DoubleTensor([times_sum / (t + 1)]), self.g_ep.value)

            mean_episode_len += (
                (episode_len - mean_episode_len) / self.g_ep.value)
            summary_writer.add_scalar("Mean_Episode_Length",
                                      torch.DoubleTensor([mean_episode_len]),
                                      total_step)

        self.res_queue.put(None)
        summary_writer.close()


if __name__ == "__main__":
    gnet = Net(N_S, N_A)  # global network

    last_checkpoint = max(
        glob.glob(save_model_dir + "/global/*"),
        key=os.path.getctime,
        default=None)
    if last_checkpoint:
        gnet.load_state_dict(torch.load(last_checkpoint))
        print("Loaded saved global model:", last_checkpoint)
    gnet.share_memory()  # share the global parameters in multiprocessing
    opt = SharedAdam(gnet.parameters(), lr=0.0002)  # global optimizer
    global_ep, global_ep_r, res_queue = (mp.Value('i', 0), mp.Value('d', 0.),
                                         mp.Queue())

    # parallel training
    workers = [
        Worker(gnet, opt, global_ep, global_ep_r, res_queue, i)
        for i in range(3)
    ]  # mp.cpu_count())]
    [w.start() for w in workers]
    res = []  # record episode reward to plo8
    while True:
        r = res_queue.get()
        if r is not None:
            res.append(r)
        else:
            break
    [w.join() for w in workers]

    import matplotlib.pyplot as plt

    plt.plot(res)
    plt.ylabel('Moving average ep reward')
    plt.xlabel('Step')
    plt.show()
