import pickle
from collections import namedtuple

import gym
from gym import spaces
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal

import matplotlib.pyplot as plt
import numpy as np

# if gpu is to be used
device = torch.device("cpu")


# Check out action and state spaces

env = gym.make("Pendulum-v0")
observation_space = env.observation_space.shape[0]
action_space = env.action_space.shape[0]

env.reset()
print("observation_space = {} action_space = {}".format(observation_space, action_space))
env.close()


# ## Setup memory buffer and hyper-params

TrainingRecord = namedtuple('TrainingRecord', ['ep', 'reward', 'MI'])
Transition = namedtuple('Transition', ['s', 'a', 'r', 's_'])

class Memory():

    data_pointer = 0
    isfull = False

    def __init__(self, capacity):
        self.memory = np.empty(capacity, dtype=object)
        self.capacity = capacity

    def update(self, transition):
        self.memory[self.data_pointer] = transition
        self.data_pointer += 1
        if self.data_pointer == self.capacity:
            self.data_pointer = 0
            self.isfull = True

    def sample(self, batch_size):
        return np.random.choice(self.memory, batch_size)


# Hyper parameters

GAMMA = 0.9
LOG_INTERVAL = 10

BATCH_SIZE = 128

EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10
PATH = "saved_weights_{}"


class Dynamics():
    def __init__(self, g=10.0):
        self.max_speed=8
        self.max_torque=2.
        self.dt=.05
        self.g = g
        self.m = 1.
        self.l = 1.
        high = np.array([1., 1., self.max_speed])
    def angle_normalize(self, x):
        return (((x + np.pi) % (2 * np.pi)) - np.pi)

    def _get_obs(self):
        sin_theta, cos_theta, thetadot = self.state
        return np.array([sin_theta, cos_theta, thetadot],dtype=float)

    def step(self,u, state):
        u = u.numpy()[0]
        th = np.arctan2(state[1], state[0])
        thdot = state[2]
        g = self.g
        m = self.m
        l = self.l
        dt = self.dt
        u = np.clip(u, -self.max_torque, self.max_torque)[0]
        costs = self.angle_normalize(th)**2 + .1*thdot**2 + .001*(u**2)

        newthdot = thdot + (-3*g/(2*l) * np.sin(th + np.pi) + 3./(m*l**2)*u) * dt
        newth = th + newthdot*dt
        newthdot = np.clip(newthdot, -self.max_speed, self.max_speed) #pylint: disable=E1111

        self.state = np.array([np.cos(newth), np.sin(newth), newthdot],dtype=float)
        return self._get_obs(), -costs, False, {}


# # 3 Setup networks

# ## Baseline A2C
# Lets create our network:

class Source(nn.Module):
    def __init__(self, n_actions, n_states):
        super(Source, self).__init__()
        self.fc = nn.Linear(n_states, 100)
        self.mu_head = nn.Linear(100, 1)
        self.var = nn.Linear(100, 1)

    def forward(self, s):
        s = torch.from_numpy(s).float().unsqueeze(0)
        x = F.relu(self.fc(s))
        mu = 2.0 * torch.tanh(self.mu_head(x))
        sig = F.elu(self.var(x))+1
        return mu, sig

    def select_action(self, state):
        mean, var = self.forward(state)
        dist = Normal(mean, var)
        action = dist.sample()
        action.clamp(-2.0, 2.0)
        return action, dist.log_prob(action)


class Planning(nn.Module):

    def __init__(self, n_actions, n_states):
        super(Planning, self).__init__()
        self.fc1 = nn.Linear(n_states, 100)
        self.fc2 = nn.Linear(n_states, 100)
        self.fc = nn.Linear(200, 100)
        self.mu_head = nn.Linear(100, 1)
        self.var = nn.Linear(100, 1)

    def forward(self, s, s_next):
        s = torch.from_numpy(s).float().unsqueeze(0)
        s = F.relu(self.fc1(s))
        s_next = F.relu(self.fc2(s_next))
        s_cat = torch.cat([s, s_next], dim=-1)
        x = F.relu(self.fc(s_cat))
        mu = 2.0 * torch.tanh(self.mu_head(x))
        sig = F.elu(self.var(x)) + 1
        return mu, sig

    def select_action(self, state, state_next):
        state_next = torch.from_numpy(state_next).float().unsqueeze(0)
        mean, variance = self.forward(state, state_next)
        dist = Normal(mean, variance)
        action = dist.sample()
        action.clamp(-2.0, 2.0)
        return action, dist.log_prob(action)


class ActorNet(nn.Module):

    def __init__(self, n_actions, n_states):
        super(ActorNet, self).__init__()
        self.fc = nn.Linear(n_states, 100)
        self.mu_head = nn.Linear(100, n_actions)

    def forward(self, s):
        x = F.relu(self.fc(s))
        u = 2.0 * torch.tanh(self.mu_head(x))
        return u

class CriticNet(nn.Module):

    def __init__(self, n_actions, n_states):
        super(CriticNet, self).__init__()
        self.fc = nn.Linear(n_actions + n_states, 100)
        self.v_head = nn.Linear(100, n_actions)

    def forward(self, s, a):
        x = F.relu(self.fc(torch.cat([s, a], dim=1)))
        state_value = self.v_head(x)
        return state_value

class Agent():

    max_grad_norm = 0.5

    def __init__(self, n_actions, n_states):
        self.training_step = 0
        self.var = 1.
        self.eval_cnet, self.target_cnet = CriticNet(n_actions, n_states).float(), CriticNet(n_actions, n_states).float()
        self.eval_anet, self.target_anet = ActorNet(n_actions, n_states).float(), ActorNet(n_actions, n_states).float()
        self. source_distribution = Source(n_actions = 1, n_states=3)
        self.planning_distribution = Planning(n_actions = 1, n_states=3)
        self.forward_dynamics = Dynamics()
        self.optimizer_c = optim.Adam(self.eval_cnet.parameters(), lr=1e-3)
        self.optimizer_a = optim.Adam(self.eval_anet.parameters(), lr=3e-4)
        self.source_dist_optimizer = optim.Adam(self.source_distribution.parameters(), lr=1e-3)
        self.planning_dist_optimizer = optim.Adam(self.source_distribution.parameters(), lr=1e-3)
        self.avg_MI = 0
    def select_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        mu = self.eval_anet(state)
        dist = Normal(mu, torch.tensor(self.var, dtype=torch.float))
        action = dist.sample()
        action.clamp(-2.0, 2.0)
        return (action.item(),)

    def update(self, transitions):
        self.training_step += 1
        #transitions = self.memory.sample(32)
        s = torch.tensor([t.s for t in transitions], dtype=torch.float)
        a = torch.tensor([t.a for t in transitions], dtype=torch.float).view(-1, 1)
        r = torch.tensor([t.r for t in transitions], dtype=torch.float).view(-1, 1)
        s_ = torch.tensor([t.s_ for t in transitions], dtype=torch.float)

        with torch.no_grad():
            q_target = r + GAMMA * self.target_cnet(s_, self.target_anet(s_))
        q_eval = self.eval_cnet(s, a)


        # Update planning and source         # update critic net        # update actor net
        source_sample, source_log_prob = self.source_distribution.select_action(state)
        next_state = self.forward_dynamics.step(source_sample, state)[0]
        planning_sample, planning_log_prob = self.planning_distribution.select_action(state, next_state)
        self.optimizer_c.zero_grad()
        self.optimizer_a.zero_grad()
        self.source_dist_optimizer.zero_grad()
        self.planning_dist_optimizer.zero_grad()

        extra_loss = -planning_log_prob + source_log_prob
        c_loss = F.smooth_l1_loss(q_eval, q_target)
        a_loss = -self.eval_cnet(s, self.eval_anet(s)).mean() + extra_loss
        c_loss.backward(retain_graph=True)
        nn.utils.clip_grad_norm_(self.eval_cnet.parameters(), self.max_grad_norm)
        self.optimizer_c.step()
        a_loss.backward(retain_graph=True )
        nn.utils.clip_grad_norm_(self.eval_anet.parameters(), self.max_grad_norm)
        self.optimizer_a.step()
        nn.utils.clip_grad_norm_(self.source_distribution.parameters(), 0.5)
        self.source_dist_optimizer.step()
        nn.utils.clip_grad_norm_(self.planning_distribution.parameters(), 0.5)
        self.planning_dist_optimizer.step()


        if self.training_step % 200 == 0:
            self.target_cnet.load_state_dict(self.eval_cnet.state_dict())
        if self.training_step % 201 == 0:
            self.target_anet.load_state_dict(self.eval_anet.state_dict())
        self.var = max(self.var * 0.999, 0.01)

        return q_eval.mean().item(), -extra_loss.item()

from PIL import Image
import torchvision.transforms as T
resize = T.Compose([T.ToPILImage(),
                    T.Resize(40, interpolation=Image.CUBIC),
                    T.ToTensor()])

def get_screen():
    screen = env.render(mode='rgb_array').transpose((2, 0, 1))
    # Cart is in the lower half, so strip off the top and bottom of the screen
    _, screen_height, screen_width = screen.shape
    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
    screen = torch.from_numpy(screen)
    # Resize, and add a batch dimension (BCHW)
    return resize(screen).unsqueeze(0).to(device)

agent = Agent(n_actions=1, n_states=3)

state = env.reset()
action = agent.select_action(state)
state, reward, done, _ = env.step(action)

plt.figure()
plt.imshow(get_screen().cpu().squeeze(0).permute(1, 2, 0).numpy(),
           interpolation='none')
env.close()
plt.title('Example extracted screen, state = x: {:.3f} y: {:.3f} angle vel: {:.3f}, action: {:.3f}'.format(state[0], state[1], state[2], action[0]))
plt.show()


training_records = []
running_reward, running_q = -1000, 0
memory = Memory(2000)

agent = Agent(n_actions=1, n_states=3)
MI = 0
for i_ep in range(500):
    score = 0
    state = env.reset()
    for t in range(200):
        action = agent.select_action(state)
        state_, reward, done, _ = env.step(action)
        score += reward
        memory.update(Transition(state, action, (reward + 8) / 8, state_))
        state = state_
        env.render()
        if memory.isfull:
            transitions = memory.sample(16)
            q, MI = agent.update(transitions)
            running_q = 0.99 * running_q + 0.01 * q
    running_reward = running_reward * 0.9 + score * 0.1
    training_records.append(TrainingRecord(i_ep, running_reward, MI))

    if i_ep % LOG_INTERVAL == 0:
        print('Step {}\tAverage score: {:.2f}\tAverage Q: {:.2f}\tMI: {:.2f}'.format(
                i_ep, running_reward, running_q, MI))

    if running_reward > -200:
        print("Solved! Running reward is now {}!".format(running_reward))
        env.close()
        torch.save(agent.eval_anet.state_dict(), 'ddpg_anet_params.pkl')
        torch.save(agent.eval_cnet.state_dict(), 'ddpg_cnet_params.pkl')
        with open('ddpg_training_records.pkl', 'wb') as f:
            pickle.dump(training_records, f)
        break

env.close()


plt.plot([r.ep for r in training_records], [r.reward for r in training_records])
plt.title('EmpoweredDDPG')
plt.xlabel('Episode')
plt.ylabel('Moving averaged episode reward')
plt.savefig("reward.png")
plt.show()
plt.plot

plt.plot([r.ep for r in training_records], [r.MI for r in training_records])
plt.title('EmpoweredDDPG')
plt.xlabel('Episode')
plt.ylabel('Instantaneous MI')
plt.savefig("mi.png")
plt.show()
plt.plot