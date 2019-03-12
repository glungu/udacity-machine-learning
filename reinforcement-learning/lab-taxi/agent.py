import numpy as np
from collections import defaultdict

class Agent:

    # Best result: 9.354

    def __init__(self, nA=6):
        """ Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        """
        self.nA = nA
        self.Q = defaultdict(lambda: np.zeros(self.nA))
        self.epsilon = 1
        self.gamma = 0.9
        self.alpha = 0.1
        self.episode_num = 1


    def select_action(self, state):
        """ Given the state, select an action.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        # return np.random.choice(self.nA)

        # epsilon-greedy policy
        action_values = self.Q[state]
        # divide epsilon among all actions
        probs = np.ones(self.nA) * self.epsilon / self.nA
        best_action = np.argmax(action_values)
        # favor best action with (1-epsilon)
        probs[best_action] += 1 - self.epsilon
        # randomly sample according to probabilities
        return np.random.choice(np.arange(self.nA), p=probs)


    def step(self, state, action, reward, next_state, done):
        """ Update the agent's knowledge, using the most recently sampled tuple.

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """
        # self.Q[state][action] += 1

        # implementing Q-learning (Sarsamax)
        # self.Q[state][action] = (1-self.alpha)*self.Q[state][action] + self.alpha*(reward + self.gamma*max(self.Q[next_state]))

        # implementing Sarsa (collects more rewards in runtime)
        next_action = self.select_action(next_state)
        self.Q[state][action] = (1-self.alpha)*self.Q[state][action] + self.alpha*(reward + self.gamma*self.Q[next_state][next_action])

        # Expected Sarsa 
        # action_values = self.Q[state]
        # policy = np.ones(self.nA) * self.epsilon / self.nA
        # best_action = np.argmax(action_values)
        # policy[best_action] += 1 - self.epsilon
        # Q_expected = np.dot(self.Q[next_state], policy)
        # self.Q[state][action] = (1-self.alpha)*self.Q[state][action] + self.alpha*(reward + self.gamma*Q_expected)

        if done:
            self.episode_num += 1
            self.epsilon = 1.0/self.episode_num if 1.0/self.episode_num > 0.001 else 0.001