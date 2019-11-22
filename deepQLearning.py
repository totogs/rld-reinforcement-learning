
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


random.seed(42)


class FeaturesExtractor(object):
	def __init__(self,outSize):
		super().__init__()
		self.outSize=outSize*3
	def getFeatures(self, obs):
		state=np.zeros((3,np.shape(obs)[0],np.shape(obs)[1]))
		state[0]=np.where(obs == 2,1,state[0])
		state[1] = np.where(obs == 4, 1, state[1])
		state[2] = np.where(obs == 6, 1, state[2])
		return state.reshape(1,-1)


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

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


class DeepQlearningAgent():

	def __init__(self, n_actions, state_dim, replay_memory_capacity=100000,
				 ctarget=1000, layers=[], batch_size=100, lr=0.001, gamma=0.999,
				 epsilon=0.01, epsilon_decay=0.99999, verbose=False):

		self.replay_memory = ReplayMemory(replay_memory_capacity)
		self.replay_memory_capacity = replay_memory_capacity
		self.ctarget = ctarget
		self.n_actions = n_actions
		self.lr = lr
		self.epsilon = epsilon
		self.epsilon_decay = epsilon_decay
		self.gamma = gamma
		self.batch_size = batch_size

		self.Q = NN(state_dim, n_actions,layers).to(device)
		self.Q_target = copy.deepcopy(self.Q).to(device)
		self.optimizer = optim.Adam(self.Q.parameters())
		self.criterion = nn.SmoothL1Loss()

		self.lobs = None
		self.laction = None
		self.t = 0


	def act(self, obs, reward, done):

		obs = torch.tensor(obs).float().to(device)
		reward = torch.tensor(reward).float().to(device)

		if self.t < self.batch_size:
			if self.t > 0:
				self.replay_memory.push(self.lobs, self.laction, obs, reward)
			self.t += 1
			self.lobs = obs

			action = random.randint(0, self.n_actions-1)
			self.laction = torch.tensor(action)

			return action

		# store transitions in replay memory D
		self.replay_memory.push(self.lobs, self.laction, obs, reward)
		# sample random minibatch of transitions from memory

		transitions = self.replay_memory.sample(self.batch_size)
		batch = Transition(*zip(*transitions))

		non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device, dtype=torch.bool)
		non_final_next_states = torch.stack([s for s in batch.next_state if s is not None]).float().to(device)

		state_batch = torch.stack(batch.state).float().to(device)
		action_batch = torch.stack(batch.action).to(device)
		reward_batch = torch.stack(batch.reward).float().to(device)

		# get output for batch
		# extract Q values only for played actions
		state_action_values = self.Q(state_batch).gather(1, action_batch.unsqueeze(1))


		next_state_values = torch.zeros(self.batch_size, device=device).float()
		next_state_values[non_final_mask] = self.Q_target(non_final_next_states).max(1)[0].detach()

		expected_state_action_values = (next_state_values * self.gamma) + reward_batch


		loss = self.criterion(state_action_values, expected_state_action_values.unsqueeze(1))

		self.optimizer.zero_grad()
		loss.backward()

		# gradient descent step
		for param in self.Q.parameters():
		    param.grad.data.clamp_(-1, 1)
		self.optimizer.step()

		writer.add_scalar('loss_per_episode', loss.item(), self.t)
		self.t += 1

		# every C step, reset target network
		if self.t % self.ctarget == 0:
			self.Q_target = copy.deepcopy(self.Q)

		# epsilon greedy choice
		if random.random() < self.epsilon:
			action = random.randint(0, self.n_actions-1)
		else:
			_ , action = torch.max(self.Q(obs.unsqueeze(0)),1)
			action = action.item()

		self.epsilon *= self.epsilon_decay

		self.lobs = obs
		self.laction = torch.tensor(action)

		return action
