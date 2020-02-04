import gym
import numpy as np
from Architecture import Network, Node, Link
import simpy
from numpy.random import RandomState
from abc import abstractmethod

RANDOM = 0
DIJKSTRA = 1
RL = 2


class BaseEnv(gym.Env):

    def __init__(self):
        self.env = simpy.Environment()
        self.gen = RandomState(2)

    @abstractmethod
    def step(self, action):
        raise NotImplementedError()

    @abstractmethod
    def reset(self):
        raise NotImplementedError()

    @abstractmethod
    def get_reward(self, action, state):
        raise NotImplementedError()

    def create_network(self, nodes, edges):
        """Create a network for the environment
        :param nodes: Array of strings representing node ids
        :param edges:
        :return: Network object
        """
        myNetwork = Network(env=self.env, gen=self.gen)
        myNetwork.add_nodes(nodes=nodes)
        myNetwork.set_routing_algorithm(controller="RL")
        myNetwork.add_links(edges)
        return myNetwork

    def get_max_action_integer(self):
        return max([len(node.get_neighbours()) for node in self.graph.nodes.values()])

    def get_state(self):
        """Get state of the packets inside the network
        :return:
        """
        state = []
        for packet in self.graph.packets:
            one_state = [packet.current_node, self.graph.nodes[packet.src], self.graph.nodes[packet.dst]]
            state.append(one_state)
        return state

    @staticmethod
    def is_finished(state):
        """Check if a all packets in the state has reached its destination
        :param state: State
        :return: Boolean
        """
        for pkt_state in state:
            if not pkt_state[0].id == pkt_state[2].id:
                return False
        return True

    @staticmethod
    def convert_state(state):
        """Convert the state of the network packets into a numpy array containing integers denoting node ids
        :param state: state of network, containing nodes -
                [[current_node1, source_node1, destination_node1],
                 [current_node2, source_node2, destination_node2]
                ]
        :return: state of packets represented as a numpy array -
                array([[current_int1, src_int1, dst_int1],
                       [current_int2, src_int2, dst_int2]
                      ])
        """
        np_state = []
        for pkt_state in state:
            add_state = [pkt_state[0].id_str_to_num(), pkt_state[1].id_str_to_num(), pkt_state[2].id_str_to_num()]
            np_state.append(add_state)

        # Find out how to do np.reshape(1,3) on np arrays with multidimensions
        np_state = np.array(np_state)
        return np_state

    @staticmethod
    def get_current_nodes_from_state(state):
        """Obtain the current nodes from the state, where packets are currently present
        :param state: state of network packets
        :return: array of the nodes containing packets
        """
        current_nodes = []
        for pkt_state in state:
            current_nodes.append(pkt_state[0])
        return current_nodes

    @staticmethod
    def get_end_nodes_from_state(state):
        """Obtain the current nodes from the state, where packets are currently present
        :param state: array - state of network packets
        :return: array of destination nodes
        """
        end_nodes = []
        for pkt_state in state:
            end_nodes.append(pkt_state[2])
        return end_nodes

    def render(self):
        pass
