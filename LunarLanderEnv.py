import matplotlib

matplotlib.use("TkAgg")
import gym
import gridworld
from gym import wrappers, logger
import numpy as np
import copy

from actorCritic import *
from deepQLearning import *

from torch.utils.tensorboard import SummaryWriter

class RandomAgent(object):
    """The world's simplest agent!"""

    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, observation, reward, done):
        return self.action_space.sample()


if __name__ == '__main__':


    env = gym.make('LunarLander-v2')

    # Enregistrement de l'Agent
    #agent = ActorCritic(env,8)

    agent = DeepQlearningAgent(env.action_space.n,8, replay_memory_capacity=1, ctarget=1000,layers=[200], batch_size=1, lr=0.0001, gamma=0.99, epsilon=0.1, epsilon_decay=0.99999)

    outdir = 'LunarLander-v2/results'
    envm = wrappers.Monitor(env, directory=outdir, force=True, video_callable=False)
    env.seed(0)

    best_rsum = -1e10
    i_best_rsum = 0

    episode_count = 100000
    reward = 0
    done = False
    env.verbose = True
    np.random.seed(5)
    rsum = 0
    lossum = 0
    env._max_episode_steps = 200
    for i in range(episode_count):
        obs = envm.reset()
        env.verbose = (i % 500 == 0 and i > 0)  # afficher 1 episode sur 100
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
