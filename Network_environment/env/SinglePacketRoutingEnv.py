from gym import error, spaces, utils
import numpy as np
from Architecture import Network, Node, Link
from RoutingControllers import RoutingAlgorithm, RandomRouting, Dijkstra
from SimComponents import Packet
import random
from BaseEnvironment import BaseEnv


class SinglePacketRoutingEnv(BaseEnv):

    def __init__(self, nodes, edges, packet=None):
        self.__version__ = "1.0.0"
        self.name = "Single Packet Routing Environment"
        super().__init__()

        self.graph = self.create_network(nodes=nodes, edges=edges)

        self.finished = False
        self.step_number = -1
        self.episode_number = -1
        self.num_nodes = len(self.graph.nodes.values())

        self.max_action = self.get_max_action_integer()
        self.action_space = spaces.Discrete(self.max_action)
        self.observation_space = None

        # State = [current node, source node, next node]
        self.state = self.initial_state(packet=packet)
        self.past_state = None
        [self.state_np] = self.convert_state([self.state])
        [self.current_node] = self.get_current_nodes_from_state([self.state])
        [self.end_node] = self.get_end_nodes_from_state([self.state])

        self.reward = None

    def initial_state(self, packet=None):
        if packet is None:
            src = random.choice(list(self.graph.nodes.keys()))
            dst = random.choice(list(self.graph.nodes.keys()))
            while dst == src:
                dst = random.choice(list(self.graph.nodes.keys()))

        else:
            src = packet[0]
            dst = packet[1]

        pkt = Packet(time=self.graph.env.now, size=1, id=1, src=src, dst=dst)
        self.graph.add_packet(pkt=pkt)

        return [
            self.graph.nodes[src],
            self.graph.nodes[src],
            self.graph.nodes[dst],
        ]

    def reset(self):
        self.graph.clear_packets()
        self.episode_number += 1
        self.step_number = -1
        self.finished = False
        self.state = self.initial_state()
        self.state_np = self.convert_state([self.state])
        [self.current_node] = self.get_current_nodes_from_state([self.state])
        [self.end_node] = self.get_end_nodes_from_state([self.state])

        return self.state_np

    def reset_episode_count(self):
        self.episode_number = 0

    def step(self, action):
        self.step_number += 1
        print(" ")
        print("Step" + str(self.step_number))
        try:
            selected_action, selected_link = self.current_node.routing_algorithm.set(action=action)

            """for node in self.graph.nodes.values():
                if node.id == self.current_node.id:
                    selected_action, selected_link = node.routing_algorithm.set(action=action)
                else:
                    pass
                    #node.routing_algorithm.set(action=action)"""

            self.env.run(until=self.env.now+1)

            self.past_state = self.state
            [self.state] = self.get_state()
            [self.state_np] = self.convert_state([self.state])
            [self.current_node] = self.get_current_nodes_from_state([self.state])
            self.reward = self.get_reward(action=selected_action, state=self.past_state, link=selected_link)
            self.finished = self.is_finished([self.state])

        except IndexError as e:
            print("index error")
            self.reward = -10

        if self.graph.packets[0].ttl <= self.graph.packets[0].ttl_safety:
            self.finished = True
            self.reward = -10

        return self.state_np, self.reward, self.finished, {}

    @staticmethod
    def get_reward(action, state, link):
        reward = None
        if action.id == state[2].id and action is not None:
            reward = 1
        elif link is not None:
            reward = -float(link.cost)/100
        return reward
