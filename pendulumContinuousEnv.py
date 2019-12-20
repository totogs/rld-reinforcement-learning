import matplotlib

matplotlib.use("TkAgg")
import gym
import gridworld
from gym import wrappers, logger
import numpy as np
import copy

from deterministicPolicyGradient import *

from torch.utils.tensorboard import SummaryWriter

class RandomAgent(object):
    """The world's simplest agent!"""

    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, observation, reward, done):
        return self.action_space.sample()


if __name__ == '__main__':


    env = gym.make('Pendulum-v0')

    # Enregistrement de l'Agent
    agent = DeepDeterministicPG(1,3, aLow=env.action_space.low[0], aHigh=env.action_space.high[0], rhoP=0.999, rhoQ=0.999, plr=0.0001, lr=0.001, epsilon=0.1, decay=0.9999,  c_update=1000, n_update=100, batch_size=128)


    outdir = 'Pendulum-v0/results'
    envm = wrappers.Monitor(env, directory=outdir, force=True, video_callable=False)
    env.seed(0)


    episode_count = 100000
    reward = 0
    done = False
    env.verbose = True
    np.random.seed(5)
    rsum = 0



    for i in range(episode_count):
        obs = envm.reset()
        env.verbose = (i % 40 == 0 and i > 0)  # afficher 1 episode sur 100
        if env.verbose:
            env.render()
        j = 0
        rsum = 0

        while True:
            action = agent.act(obs, reward, done)

            obs, reward, done, _ = env.step(action)


            rsum += reward
            j += 1
            if env.verbose:
                env.render()
            if done:
                print("Episode : " + str(i) + " rsum=" + str(rsum) + ", " + str(j) + " actions")
                writer.add_scalar('Rewards_per_episode',rsum,i)
                break

    print("done")
    env.close()
