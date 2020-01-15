import pickle
from collections import namedtuple
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np

device = torch.device("cpu")

#Action and state spaces
env = gym.make("Pendulum-v0")
observation_space = env.observation_space.shape[0]
action_space = env.action_space.shape[0]

#Memory buffer and hyper-params
TrainingRecord = namedtuple('TrainingRecord', ['ep', 'reward', 'empowerment'])
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

#Hyper-parameters
GAMMA = 0.9
LOG_INTERVAL = 10
BATCH_SIZE = 128
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10
M = 10
T = 100
BETA = 5
PATH = "saved_weights_{}"

class Source(nn.Module):
    def __init__(self):
        super(Source, self).__init__()

        self.fc1 = nn.Linear(3, 2)
        self.fc21 = nn.Linear(2, 1)
        self.fc22 = nn.Linear(2, 1)
        self.fc3 = nn.Linear(1, 2)
        self.fc4 = nn.Linear(2, 3)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


class Planning(nn.Module):
    def __init__(self):
        super(Planning, self).__init__()

        self.fc1 = nn.Linear(6, 2)
        self.fc21 = nn.Linear(2, 1)
        self.fc22 = nn.Linear(2, 1)
        self.fc3 = nn.Linear(1, 2)
        self.fc4 = nn.Linear(2, 6)

    def encode(self, x, x_):
        h1 = F.relu(self.fc1(torch.cat([x, x_], dim=-1)))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x, x_):
        mu, logvar = self.encode(x, x_)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD


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
        self.optimizer_c = optim.Adam(self.eval_cnet.parameters(), lr=1e-3)
        self.optimizer_a = optim.Adam(self.eval_anet.parameters(), lr=3e-4)

    def select_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        mu = self.eval_anet(state)
        dist = Normal(mu, torch.tensor(self.var, dtype=torch.float))
        action = dist.sample()
        action.clamp(-2.0, 2.0)
        return (action.item(),), dist

    def update(self, transitions):
        self.training_step += 1
        s = torch.tensor([t.s for t in transitions], dtype=torch.float)
        a = torch.tensor([t.a for t in transitions], dtype=torch.float).view(-1, 1)
        r = torch.tensor([t.r for t in transitions], dtype=torch.float).view(-1, 1)
        s_ = torch.tensor([t.s_ for t in transitions], dtype=torch.float)

        with torch.no_grad():
            q_target = r + GAMMA * self.target_cnet(s_, self.target_anet(s_))
        q_eval = self.eval_cnet(s, a)

        self.optimizer_c.zero_grad()
        self.optimizer_a.zero_grad()

        c_loss = F.smooth_l1_loss(q_eval, q_target)
        a_loss = -self.eval_cnet(s, self.eval_anet(s)).mean()
        c_loss.backward(retain_graph=True)
        nn.utils.clip_grad_norm_(self.eval_cnet.parameters(), self.max_grad_norm)
        self.optimizer_c.step()
        a_loss.backward(retain_graph=True )
        nn.utils.clip_grad_norm_(self.eval_anet.parameters(), self.max_grad_norm)
        self.optimizer_a.step()

        if self.training_step % 200 == 0:
            self.target_cnet.load_state_dict(self.eval_cnet.state_dict())
        if self.training_step % 201 == 0:
            self.target_anet.load_state_dict(self.eval_anet.state_dict())
        self.var = max(self.var * 0.999, 0.01)

        return q_eval.mean().item()

training_records = []
running_reward, running_q = -1000, 0        
memory = Memory(2000)
agent = Agent(n_actions=1, n_states=3)
source_network = Source().double()
planning_network = Planning().double()
source_optimizer = optim.Adam(source_network.parameters(), lr=1e-3)
planning_optimizer = optim.Adam(planning_network.parameters(), lr=1e-3)
epoch = 0

while epoch <= 50:
    empowerment = 0
    for m in range(M):
        score = 0
        state = env.reset()
        for t in range(T):
            action, policy_dist = agent.select_action(state)
            state_, reward, done, _ = env.step(action)
            #print(forward_dynamics.step(torch.tensor(np.asarray(action)).unsqueeze(0),state), state_)
            state = Variable(torch.from_numpy(state).double())
            state_ = Variable(torch.from_numpy(state_).double())
            score += reward
            memory.update(Transition(state, action, (reward + 8) / 8, state_))
            source_lat, source_mu, source_logvar = source_network(state)
            source_dist = Normal(source_mu, source_logvar)
            source_action = source_dist.sample()
            source_action.clamp(-2.0, 2.0)
            source_log_prob = source_dist.log_prob(source_action)

            planning_lat, planning_mu, planning_logvar = planning_network(state, state_)
            planning_dist = Normal(planning_mu, planning_logvar)
            planning_action = planning_dist.sample()
            planning_action.clamp(-2.0, 2.0)
            planning_log_prob = planning_dist.log_prob(planning_action)

            MI = planning_log_prob - source_log_prob
            empowerment += -MI - torch.distributions.kl.kl_divergence(policy_dist, Normal(torch.tensor([[0.0]]),torch.tensor([[1.0]])))
            state = state_
            env.render()
        print("trajectory: ", m)

    if memory.isfull:
        transitions = memory.sample(16)
        q = agent.update(transitions)
        running_q = 0.99 * running_q + 0.01 * q
        empowerment.backward()
        source_optimizer.zero_grad()
        planning_optimizer.zero_grad()
        nn.utils.clip_grad_norm_(source_network.parameters(), 0.5)
        source_optimizer.step()
        nn.utils.clip_grad_norm_(planning_network.parameters(), 0.5)
        planning_optimizer.step()
    '''model.train()
    train_loss = 0
    data = data
    optimizer.zero_grad()
    recon_batch, mu, logvar = model(data)
    loss = loss_function(recon_batch, data, mu, logvar)
    loss.backward()
    train_loss += loss.item()
    optimizer.step()'''
    running_reward = running_reward * 0.9 + score * 0.1
    training_records.append(TrainingRecord(m, running_reward, empowerment.item()))
    print('Epoch {}\tAverage score: {:.2f}\tAverage Q: {:.2f}\tEmpowerment: {:.2f}'.format(
                epoch, running_reward, running_q, empowerment.item()))
    if running_reward > -200:
        print("Solved! Running reward is now {}!".format(running_reward))
        env.close()
        torch.save(agent.eval_anet.state_dict(), 'ddpg_anet_params.pkl')
        torch.save(agent.eval_cnet.state_dict(), 'ddpg_cnet_params.pkl')
        with open('ddpg_training_records.pkl', 'wb') as f:
            pickle.dump(training_records, f)
        break
    epoch += 1
    BETA += (2000-5)/800
env.close()

plt.plot([r.ep for r in training_records], [r.reward for r in training_records])
plt.title('DDPG')
plt.xlabel('Episode')
plt.ylabel('Moving averaged episode reward')
plt.savefig("ddpg.png")
plt.show()

