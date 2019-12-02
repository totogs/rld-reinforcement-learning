
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

class NN(nn.Module):
	def __init__(self, inSize, outSize, layers=[]):
		super(NN, self).__init__()
		self.layers = nn.ModuleList([])
		for x in layers:
			self.layers.append(nn.Linear(inSize, x))
			inSize = x
		self.layers.append(nn.Linear(inSize, outSize))
	def forward(self, x):
		x = self.layers[0](x)
		for i in range(1, len(self.layers)):
			x = torch.nn.functional.leaky_relu(x)
			x = self.layers[i](x)
		return x


class Qnet(nn.Module):
	def __init__(self, state_dim, action_dim, layers=[]):
		super(Qnet, self).__init__()
		self.layers = nn.ModuleList([])
		self.layers.append(nn.Linear(state_dim,layers[0]))
		self.layers.append(nn.Linear(layers[0]+action_dim,layers[1]))

		inSize = layers[1]
		for x in layers[2:]:
			self.layers.append(nn.Linear(inSize, x))
			inSize = x
		self.layers.append(nn.Linear(inSize, 1))

	def forward(self, state, action):
		x = self.layers[0](state)
		x = self.layers[1](torch.cat([x,action]))

		for i in range(2, len(self.layers)):
			x = torch.nn.functional.leaky_relu(x)
			x = self.layers[i](x)

		return torch.nn.tanh(x)


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

	def __init__(self, n_actions, state_dim, replay_memory_capacity=100000, rho=0.8,  aLow=-1, aHigh=1,
				 c_update=1000, n_update=20, layersP=[128], layersQ=[10,10,10], batch_size=100, lr=0.001, gamma=0.999, verbose=False):

		self.replay_memory = ReplayMemory(replay_memory_capacity)
		self.replay_memory_capacity = replay_memory_capacity
		self.c_update = c_update
		self.n_update = n_update

		self.n_actions = n_actions
		self.lr = lr
		self.rho =rho
		self.gamma = gamma
		self.batch_size = batch_size

		self.aLow = aLow
		self.aHigh = aHigh

		self.policy = NN(state_dim, n_actions,layers=layersP).to(device)
		self.policy_target = copy.deepcopy(self.policy).to(device)
		self.Q = Qnet(state_dim, n_actions,layers=layersQ).to(device)
		self.Q_target = copy.deepcopy(self.Q).to(device)

		self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=self.lr)
		self.Q_optimizer = optim.Adam(self.Q.parameters(), lr=self.lr)
		self.criterion = nn.MSELoss()

		self.lobs = None
		self.laction = None
		self.t = 0
		self.decay = 1


	def act(self, obs, reward, done):

		obs = torch.tensor(obs).float().to(device)
		reward = torch.tensor(reward).float().to(device)

		if self.t < self.batch_size:
			if self.t > 0:
				self.replay_memory.push(self.lobs, self.laction, obs, reward, done)
			self.t += 1
			self.lobs = obs

			action = self.policy(obs) + torch.randn(self.n_actions).to(device)*0.01*self.decay
			action._clamp(self.aLow,self.aHigh)

			self.decay *= 0.9999
			self.laction = torch.tensor(action)

			return action

		# store transitions in replay memory D
		self.replay_memory.push(self.lobs, self.laction, obs, reward, done)
		# sample random minibatch of transitions from memory


		writer.add_scalar('loss_per_episode', loss.item(), self.t)
		self.t += 1

		# every C step, reset target network
		if self.t % self.c_update == 0:
			for i in range(self.n_update):
				self.update()

		action = self.policy(obs) + torch.randn(self.n_actions)*0.01*self.decay
		action.clamp_(-1,1)

		self.decay *= 0.9999

		self.lobs = obs
		self.laction = torch.tensor(action)

		return action

		def update(self):

			transitions = self.replay_memory.sample(self.batch_size)
			batch = Transition(*zip(*transitions))

			state_batch = torch.stack(batch.state).float().to(device)
			action_batch = torch.stack(batch.action).float().to(device)
			reward_batch = torch.stack(batch.reward).float().to(device)
			next_state_batch = torch.stack(batch.next_state).float().to(device)
			done_batch = torch.stack(batch.done).to(device)

			y_batch = reward_batch + self.gamma*(1-done_batch)*self.Q_target(next_state_batch, self.policy_target(next_state_batch))

			self.policy_optimizer.zero_grad()
			self.Q_optimizer.zero_grad()

			Qloss = self.criterion(self.Q(state_batch, action_batch), y_batch).mean()
			Qloss.backward()

			Policyloss = self.Q(state_batch, self.policy(state_batch)).mean()
			Policyloss.backward()


			self.policy_optimizer.step()
			self.Q_optimizer.step()

			for param_target, param in zip(self.Q_target.parameters(), self.Q.parameters()):
			    param_target.data = self.rho*param_target.data + (1-self.rho)*param.data

			for param_target, param in zip(self.policy_target.parameters(), self.policy.parameters()):
			    param_target.data = self.rho*param_target.data + (1-self.rho)*param.data
