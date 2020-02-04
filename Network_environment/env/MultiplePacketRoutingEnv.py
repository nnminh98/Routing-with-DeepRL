from gym import error, spaces, utils
import numpy as np
from Architecture import Network, Node, Link
from RoutingControllers import RoutingAlgorithm, RandomRouting, Dijkstra
from SimComponents import Packet
import random
from BaseEnvironment import BaseEnv


class MultiplePacketRoutingEnv(BaseEnv):

    def __init__(self, nodes, edges):
        self.__version__ = "1.0.0"
        self.name = "Multiple Packet Routing Environment"
        super().__init__()

        self.graph = self.create_graph(nodes=nodes, edges=edges)

        self.finished = False
        self.step_number = -1
        self.episode_number = -1
        self.num_nodes = len(self.graph.nodes.values())

    def step(self, action):
        pass

    def reset(self):
        pass

    def get_reward(self):
        pass
