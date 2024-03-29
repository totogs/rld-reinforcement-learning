import matplotlib

matplotlib.use("TkAgg")
import gym
import gridworld
from gym import wrappers, logger
import numpy as np
import copy
from deepQLearning import *
from actorCritic import *
from utils import CheckpointState, EarlyStopper

from torch.utils.tensorboard import SummaryWriter

class RandomAgent(object):
    """The world's simplest agent!"""

    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, observation, reward, done):
        return self.action_space.sample()


if __name__ == '__main__':


    env = gym.make('CartPole-v0')
    writer = SummaryWriter()

    # Enregistrement de l'Agent
    agent = DeepQlearningAgent(env.action_space.n,4)

    #agent = ActorCritic(env,4)

    outdir = 'cartpole-v0/random-agent-results'
    envm = wrappers.Monitor(env, directory=outdir, force=True, video_callable=False)
    env.seed(0)

    best_rsum = -1e10
    i_best_rsum = 0


    episode_count = 8000
    reward = 0
    done = False
    env.verbose = True
    rsum = 0
    lossum = 0
    env._max_episode_steps = 10000
    for i in range(episode_count):
        obs = envm.reset()
        env.verbose = (i % 10 == 0 and i > 0)  # afficher 1 episode sur 100
        if env.verbose:
            env.render()
        j = 0
        rsum = 0
        lossum = 0

        while True:

            action = agent.act(obs, reward, done)
            loss = agent.optimize(done)
            lossum += loss

            obs, reward, done, _ = envm.step(action)
            rsum += reward
            j += 1

            if env.verbose:
                env.render()
            if done:
                print("Episode : " + str(i) + " rsum=" + str(rsum) + ", " + str(j) + " actions")
                writer.add_scalar('Rewards_per_episode',rsum,i)
                writer.add_scalar('Loss_per_episode',lossum/j,i)
                agent.checkpoint.epoch +=1

                if rsum > best_rsum:
                    best_rsum = rsum
                    i_best_rsum = agent.checkpoint.epoch
                    agent.checkpoint.save(suffix='_best')

                agent.checkpoint.save()
                break

    print("done")
    env.close()
