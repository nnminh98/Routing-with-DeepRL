import gym
import sys
import numpy as np


class DijkstraAgent(object):

    def __init__(self, openAI_env="Single Packet Routing Environment"):
        self.env_name = openAI_env
        self.env = gym.make(self.env_name, )
        self.nodes = self.env.graph.nodes

        for i in range(100):
            self.env.reset()
            self.initial_state = self.env.state
            self.src, self.dst = self.initial_state[1], self.initial_state[1]

            self.spt_set = self.env.graph.nodes.copy()
            for node in self.spt_set.keys():
                if node == self.src.id:
                    self.spt_set[node] = True
                else:
                    self.spt_set[node] = False

            self.distances = self.env.graph.nodes.copy()
            for node in self.distances.keys():
                if node == self.src.id:
                    self.distances[node] = 0
                else:
                    self.distances[node] = np.inf

    def __call__(self):
        done = False
        current_node = self.src
        current_state = self.initial_state

        while not done:
            pass
