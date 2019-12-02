#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import matplotlib

matplotlib.use("TkAgg")
import gym
import gridworld
from gym import wrappers, logger
import numpy as np
import copy

from torch.utils.tensorboard import SummaryWriter
from modelFreeAgent import *
from actorCritic import *


class FeaturesExtractor(object):
    def __init__(self):
        super().__init__()
    def getFeatures(self, obs):
        state=np.zeros((3,np.shape(obs)[0],np.shape(obs)[1]))
        state[0]=np.where(obs == 2,1,state[0])
        state[1] = np.where(obs == 4, 1, state[1])
        state[2] = np.where(obs == 6, 1, state[2])
        return state.reshape(1,-1)

class RLAgent(object):
    """The world's simplest agent!"""

    def __init__(self, env, epsilon, gamma):
        self.env = env
        self.epsilon = epsilon
        self.gamma = gamma
        self.Pi = value_iteration(env, epsilon, gamma)
        self.reward = 0

    def act(self, observation, reward, done):

        pos = 0
        self.reward += reward

        return self.Pi[pos]


FE = FeaturesExtractor()


#Pi = policy_iteration(1E-5,0.99)



if __name__ == '__main__':


    writer = SummaryWriter()

    env = gym.make("gridworld-v0")
    env.seed(0)  # Initialise le seed du pseudo-random
    print(env.action_space)  # Quelles sont les actions possibles
    print(env.step(1))  # faire action 1 et retourne l'observation, le reward, et un done un booleen (jeu fini ou pas)
    env.render()  # permet de visualiser la grille du jeu
    env.render(mode="human") #visualisation sur la console
    statedic, mdp = env.getMDP()  # recupere le mdp : statedic


    # Faire un fichier de log sur plusieurs scenarios
    outdir = 'gridworld-v0/random-agent-results'
    env.setPlan("gridworldPlans/plan2.txt", {0: -0.05, 3: 1, 4: 1, 5: -1, 6: -1})
    env.seed()  # Initialiser le pseudo aleatoire

    # Execution avec un Agent
    state_dim = FE.getFeatures(env.reset()).shape[1]
    agent = ActorCritic(env,state_dim, layers=[30,30], lrpi=0.001, lrv=0.001)

    episode_count = 1000
    reward = 0
    done = False
    rsum = 0
    FPS = 0.0001

    for i in range(episode_count):
        obs = env.reset()
        agent.setState(FE.getFeatures(obs))
        env.verbose = (i % 100 == 0 and i > 0)  # afficher 1 episode sur 100
        if env.verbose:
            env.render(FPS)
        j = 0
        rsum = 0
        while True:
            action = agent.act(FE.getFeatures(obs), reward, done)
            obs, reward, done, _ = env.step(action)
            rsum += reward
            j += 1
            if env.verbose:
                env.render(FPS)
            if done:
                print("Episode : " + str(i) + " rsum=" + str(rsum) + ", " + str(j) + " actions")
                writer.add_scalar('Rewards_per_episode',rsum,i)
                break

    print("done")
    env.close()
    writer.close()
