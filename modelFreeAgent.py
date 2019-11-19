import numpy as np
import random
import gym
from gridworld import GridworldEnv


class PolicyAgent():
    """Agent that follows an arbitrary policy, for the Gridworld Gym
    environment"""
    def __init__(self, env):
        self.nb_action = env.action_space.n
        states, P = env.getMDP()
        self.action_space = action_space
        # policy: dictionary of (state -> action)
        self.policy = policy

    def act(self, observation, reward, done):
        # get action for current state
        # obs = str(obs.tolist())
        obs = GridworldEnv.state2str(observation)
        action = self.policy[obs]
        return action




class QlearningAgent():

    def __init__(self, env, lr=0.85, gamma=0.99, epsilon=0.01, verbose=False):

        self.nb_action = env.action_space.n
        self.lr = lr
        self.epsilon = epsilon
        self.gamma = gamma

        self.Q = dict()
        self.lstate = None
        self.laction = None

    def setState(self, observation):

        self.lstate = GridworldEnv.state2str(observation)

        if self.lstate not in self.Q.keys():
        	self.Q[self.lstate] = np.zeros(self.nb_action)


    def act(self, observation, reward, done):

        obs = GridworldEnv.state2str(observation)

        if obs not in self.Q.keys():
            self.Q[obs] = np.zeros(self.nb_action)


        if random.uniform(0,1) < self.epsilon:
            self.laction = np.random.randint(self.nb_action)

        else:
            self.laction = np.argmax([self.Q[obs]])

        self._update_Qvalue(reward,obs,done)

        self.lstate = obs

        return self.laction



    def _update_Qvalue(self,reward,obs,done):

        self.Q[self.lstate][self.laction] = self.Q[self.lstate][self.laction]*(1-self.lr) + self.lr*(reward+self.gamma*np.max(self.Q[obs]))



class Sarsa():

    def __init__(self, env, lr=0.85, gamma=0.99, epsilon=0.01, verbose=False):

        self.nb_action = env.action_space.n
        self.lr = lr
        self.epsilon = epsilon
        self.gamma = gamma

        self.Q = dict()
        self.lstate = None
        self.laction = None
        self.laction_next = None

    def setState(self, observation):

        self.lstate = GridworldEnv.state2str(observation)

        if self.lstate not in self.Q.keys():
        	self.Q[self.lstate] = np.zeros(self.nb_action)

        if random.uniform(0,1) < self.epsilon:
        	self.laction = np.random.randint(self.nb_action)

        else:
        	self.laction = np.argmax([self.Q[self.lstate]])


    def act(self, observation, reward, done):

        obs = GridworldEnv.state2str(observation)

        if obs not in self.Q.keys():
        	self.Q[obs] = np.zeros(self.nb_action)


        if random.uniform(0,1) < self.epsilon:
        	self.laction_next = np.random.randint(self.nb_action)

        else:
        	self.laction_next = np.argmax([self.Q[obs]])


        self._update_Qvalue(reward,obs,done)

        self.lstate = obs
        self.laction = self.laction_next

        return self.laction_next




    def _update_Qvalue(self,reward,obs,done):

        self.Q[self.lstate][self.laction] = self.Q[self.lstate][self.laction]*(1 - self.lr) + self.lr*(reward + self.gamma*self.Q[obs][self.laction_next])








class DynaQ():

    def __init__(self, env, lr=0.8, gamma=0.99, epsilon=0.05, verbose=False,k=20):

        self.nb_action = env.action_space.n
        self.lr = lr
        self.epsilon = epsilon
        self.gamma = gamma
        self.k = k

        self.Q = dict()
        self.lstate = None
        self.laction = None


        self.MDP = dict()

    def setState(self, observation):

        self.lstate = GridworldEnv.state2str(observation)

        if self.lstate not in self.Q.keys():
        	self.Q[self.lstate] = np.zeros(self.nb_action)



    def act(self, observation, reward, done):

        obs = GridworldEnv.state2str(observation)

        if obs not in self.Q.keys():
        	self.Q[obs] = np.zeros(self.nb_action)


        if random.uniform(0,1) < self.epsilon:
        	self.laction = np.random.randint(self.nb_action)

        else:
        	self.laction = np.argmax([self.Q[obs]])


        self.update_Qvalue(reward,obs,done)
        self.update_MDP(reward,obs,done)
        self.sample_couple()

        self.lstate = obs

        return self.laction




    def update_Qvalue(self,reward,obs,done):
        self.Q[self.lstate][self.laction] = self.Q[self.lstate][self.laction]*(1-self.lr) + self.lr*(reward+self.gamma*np.max(self.Q[obs]))

    def update_MDP(self, reward,obs,done):

        if (self.lstate,self.laction) in self.MDP.keys():
            if obs in self.MDP[(self.lstate,self.laction)].keys():
                p = self.MDP[(self.lstate,self.laction)][obs][0]
                r = self.MDP[(self.lstate,self.laction)][obs][1]

                self.MDP[(self.lstate,self.laction)][obs] = (p + self.lr*(1-p), r + self.lr*(reward-r))

            else:
                self.MDP[(self.lstate,self.laction)][obs] = (self.lr, reward)

            for o, (p2, r2) in self.MDP[(self.lstate,self.laction)].items():
                if o != obs:
                    self.MDP[(self.lstate,self.laction)][o] = (p2 - self.lr*p2, r2)
        else:
            self.MDP[(self.lstate,self.laction)] = {}
            self.MDP[(self.lstate,self.laction)][obs] = (1.,reward)


    def sample_couple(self):

        for _ in range(self.k):

            s, a = random.choice(list(self.MDP.keys()))

            temp = 0
            ptot = 0
            for s2, (p,r) in self.MDP[(s,a)].items():
                ptot += p
                temp += p*(r + self.gamma*np.max(self.Q[s2]))

            self.Q[s][a] = self.Q[s][a]*(1-self.lr) + self.lr*temp








def equal_value_functions(v1, v2, eps):
    """Check if two value functions are equal up to a threshold
    """
    # return np.sqrt(sum((v1[s] - v2[s])**2
    #                     for s in v1.keys())) < eps
    return sum(abs(v1[s] - v2[s]) for s in v1.keys()) < eps


def equal_policies(p1, p2):
    """Check if two policies are identical"""
    return all(p1[s] == p2[s] for s in p1.keys())


class BasePolicyPlanner():
    """Abstract base class for policy planners"""

    def fit(self, mdp, gamma=0.99, epsilon=1e-6):
        """
        Learns the optimal policy from the MDP of the
        Gridworld Gym environment.
        """
        raise NotImplementedError()

    def _updated_value(self):
        """Updates the value function using the current policy.
           The update rule depends on the algorithm and this function needs to
           be implemented.
        """
        raise NotImplementedError()

    def _expected_reward(self, state, action):
        """Expectancy of reward using the Bellman equation (Q-value)"""
        transitions = self.mdp[state][action]
        return sum(proba * (reward + self.gamma * self.value.get(s, 0))
                   for proba, s, reward, _ in transitions)


    def _updated_policy(self):
        """Greedy update of the policy, using the current value function"""
        # new_policy = dict()
        # for state in self.mdp.keys():
        #     best_reward = -float('inf')
        #     best_action = None
        #     for action in self.mdp[state].keys():
        #         reward = self._expectancy_of_reward(state, action)
        #         if reward > best_reward:
        #             best_reward = reward
        #             best_action = action

        #     new_policy[state] = best_action
        # return new_policy
        new_policy = dict()
        for state in self.mdp.keys():
            best_action, _ = max(((action, self._expected_reward(state, action))
                                  for action in self.mdp[state].keys()),
                                 key=lambda x: x[1])
            new_policy[state] = best_action
        return new_policy

    def _total_reward(self):
        """Sum of rewards expected for every state"""
        return sum(self.value[state] for state in self.mdp.keys())






        obs = GridworldEnv.state2str(observation)

        if obs in self.Q.keys():
            self.Q[obs] = np.zeros(self.nb_action)


        if random.uniform(0,1) < self.epsilon:
            self.laction = np.random.randint(self.nb_action)

        else:
            self.laction = np.argmax([self.Q[obs]])

        self._update_Qvalue(reward,obs,done)

        self.lstate = obs

        return self.laction

class PolicyIteration(BasePolicyPlanner):
    """Policy Iteration algorithm"""

    def fit(self, mdp, gamma=0.99, epsilon=1e-6, verbose=False):
        """
        Learns the optimal policy from the MDP of the
        Gridworld Gym environment.
        """
        self.mdp = mdp
        self.gamma = gamma
        self.epsilon = epsilon
        # randomly initialize the policy (only for non-terminal states)
        self.policy = {state: random.choice(tuple(mdp[state].keys()))
                       for state in mdp.keys()}
        # for state in mdp.keys():
        #     self.policy[state] = random.choice(tuple(mdp[state].keys()))

        i_policy = 0
        while True:
            # randomly initialize the value function with values in [0,1]
            #self.value = np.random.rand(len(mdp.keys()))
            self.value = {state: random.random() for state in mdp.keys()}

            i_value = 0
            while True:
                #next_value = np.zeros(len(mdp.keys()))
                next_value = self._updated_value()
                i_value += 1
                # if convergence, stop
                if equal_value_functions(next_value, self.value, epsilon):
                    if verbose: print("Policy {} evaluated in {} iterations. "
                                      "Total reward: {}"
                                      .format(i_policy, i_value, self._total_reward()))
                    break
                self.value = next_value

            # update the policy
            new_policy = self._updated_policy()
            i_policy += 1

            if equal_policies(new_policy, self.policy):
                # we have converged to the optimal stationary policy
                break
            self.policy = new_policy

        if verbose: print("Optimal policy found after {} policy iterations"
                          .format(i_policy))
        return self.policy

    def _updated_value(self):
        """Updates the value function using the current policy"""
        next_value = {state: self._expected_reward(state, self.policy[state])
                      for state in self.mdp.keys()}
        return next_value


class ValueIteration(BasePolicyPlanner):
    """Value iteration algorithm"""

    def fit(self, mdp, gamma=0.99, epsilon=1e-9, verbose=False):
        """
        Learns the optimal policy from the MDP of the
        Gridworld Gym environment.
        """
        self.mdp = mdp
        self.gamma = gamma
        self.epsilon = epsilon

        # randomly initialize the value function with values in [0,1]
        #self.value = np.random.rand(len(mdp.keys()))
        self.value = {state: random.random() for state in mdp.keys()}
        i_value = 0

        while True:
            next_value = self._updated_value()
            i_value += 1
            # if convergence, stop
            if equal_value_functions(next_value, self.value, epsilon):
                break
            if verbose:
                print(" Iteration {} of value. Total reward: {}".format(i_value, self._total_reward()))

            self.value = next_value

        # get the optimal policy
        self.policy = self._updated_policy()

        if verbose:
            print("Optimal policy found after {} value iterations".format(i_value))
        return self.policy

    def _updated_value(self):
        """Updates the value function using the current policy"""
        # next_value = dict()
        # # update the value function
        # for state in self.mdp.keys():
        #     best_reward = max(self._expected_reward(state, action)
        #                       for action in self.mdp[state].keys())
        next_value = {state: max(self._expected_reward(state, action)
                                 for action in self.mdp[state].keys())
                      for state in self.mdp.keys()}
        return next_value
