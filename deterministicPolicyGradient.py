
import numpy as np
import random
import torch
from torch.autograd import Function
from torch.autograd import gradcheck
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as f
import copy
from torch.utils.tensorboard import SummaryWriter
from collections import namedtuple

writer = SummaryWriter()



def fanin_init(size, fanin=None):
    fanin = fanin or size[0]
    v = 1. / np.sqrt(fanin)
    return torch.Tensor(size).uniform_(-v, v)

class Actor(nn.Module):
    def __init__(self, nb_states, nb_actions, hidden1=40, hidden2=30, init_w=3e-4):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(nb_states, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, nb_actions)
        self.relu = nn.LeakyReLU()
        self.tanh = nn.Tanh()
        self.init_weights(init_w)

    def init_weights(self, init_w):
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
        self.fc3.weight.data.uniform_(-init_w, init_w)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.tanh(out)
        return out

class Critic(nn.Module):
    def __init__(self, nb_states, nb_actions, hidden1=40, hidden2=30, init_w=3e-4):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(nb_states, hidden1)
        self.fc2 = nn.Linear(hidden1+nb_actions, hidden2)
        self.fc3 = nn.Linear(hidden2, 1)
        self.relu = nn.LeakyReLU()
        self.init_weights(init_w)

    def init_weights(self, init_w):
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
        self.fc3.weight.data.uniform_(-init_w, init_w)

    def forward(self, x, a):
        out = self.fc1(x)
        out = self.relu(out)
        # debug()
        out = self.fc2(torch.cat([out,a],1))
        out = self.relu(out)
        out = self.fc3(out)
        return out


random.seed(42)



Transition = namedtuple('Transition',
                    ('state', 'action', 'next_state', 'reward', 'done'))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DeepDeterministicPG():

    def __init__(self, n_actions, state_dim, replay_memory_capacity=100000, rhoP=0.95, rhoQ=0.95,  aLow=-1, aHigh=1, epsilon = 0.1, decay = 0.9999,
    		 c_update=1000, n_update=100, batch_size=100, plr=0.0001, lr=0.001, gamma=0.99, verbose=False):

        self.replay_memory = ReplayMemory(replay_memory_capacity)
        self.replay_memory_capacity = replay_memory_capacity
        self.c_update = c_update
        self.n_update = n_update

        self.n_actions = n_actions
        self.plr = plr
        self.lr = lr
        self.rhoP = rhoP
        self.rhoQ = rhoQ
        self.gamma = gamma
        self.epsilon = epsilon
        self.decay = decay
        self.batch_size = batch_size

        self.aLow = aLow
        self.aHigh = aHigh

        self.policy = Actor(state_dim, n_actions).to(device)
        self.policy_target = copy.deepcopy(self.policy).to(device)

        self.Q = Critic(state_dim, n_actions).to(device)
        self.Q_target = copy.deepcopy(self.Q).to(device)

        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=self.plr)
        self.Q_optimizer = optim.Adam(self.Q.parameters(), lr=self.lr, weight_decay=0)
        self.criterion = nn.MSELoss()

        self.lobs = None
        self.laction = None
        self.t = 0
        self.updt_cpt = 0



    def act(self, obs, reward, done):

    	obs = torch.tensor(obs).float().to(device)
    	reward = torch.tensor([reward]).float().to(device)


    	with torch.no_grad():
    		if self.t < self.batch_size:
    			if self.t > 0:
    				self.replay_memory.push(self.lobs, self.laction, obs, reward, done)
    			self.t += 1
    			self.lobs = obs

    			action = self.policy(obs.unsqueeze(0)) + torch.randn(1,self.n_actions).to(device)*self.epsilon
    			action.clamp_(self.aLow,self.aHigh)
    			action = action.squeeze(0)

    			self.epsilon *= self.decay
    			self.laction = action


    			return action.cpu().numpy()

    		# store transitions in replay memory D
    		self.replay_memory.push(self.lobs, self.laction, obs, reward, done)
    		# sample random minibatch of transitions from memory



    		action = self.policy(obs.unsqueeze(0)) + torch.randn(1,self.n_actions).to(device)*self.epsilon
    		action.clamp_(self.aLow,self.aHigh)
    		action = action.squeeze(0)

    		self.epsilon *= self.decay

    		self.lobs = obs
    		self.laction = action

    	# every C step, reset target network
    	if self.t % self.c_update == 0:
    		for i in range(self.n_update):
    			self.update()

    	self.t += 1


    	return action.cpu().numpy()

    def update(self):

        transitions = self.replay_memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        state_batch = torch.stack(batch.state).float().to(device).requires_grad_()
        action_batch = torch.stack(batch.action).float().to(device)
        reward_batch = torch.stack(batch.reward).float().to(device)
        next_state_batch = torch.stack(batch.next_state).float().to(device)
        done_batch = torch.stack([torch.tensor([1.0]) if s else torch.tensor([0.0]) for s in batch.done]).float().to(device)

        y_batch = reward_batch + self.gamma*(1.0-done_batch)*self.Q_target(next_state_batch, self.policy_target(next_state_batch))

        self.policy_optimizer.zero_grad()
        self.Q_optimizer.zero_grad()

        Qloss = self.criterion(self.Q(state_batch, action_batch), y_batch.detach())
        Qloss.backward()
        self.Q_optimizer.step()

        Policyloss = -self.Q(state_batch, self.policy(state_batch)).mean()
        Policyloss.backward()
        self.policy_optimizer.step()

        writer.add_scalar("Qloss",Qloss,self.updt_cpt)
        writer.add_scalar("policyLoss",Policyloss, self.updt_cpt)
        self.updt_cpt += 1

        for param_target, param in zip(self.Q_target.parameters(), self.Q.parameters()):
            param_target.data.copy_(self.rhoQ*param_target.data + (1-self.rhoQ)*param.data)

        for param_target, param in zip(self.policy_target.parameters(), self.policy.parameters()):
            param_target.data.copy_(self.rhoP*param_target.data + (1-self.rhoP)*param.data)
