import numpy as np
import math
import gym
import gym_network
from queue import PriorityQueue
import logging
from log_setup import init_logging


class Dijkstras():
    def __init__(
            self,
            env="PathFindingNetworkEnv-v1",
            network="germany50",
            render=False,
            mode="human",
            log_level="DEBUG",
            seed=0):

        init_logging(max_log_files=10, logging_level=log_level)
        # logging.info("Running DDQN for {} episodes.".format(str(num_episodes)))

        # temporarily initialize gym env:
        self.ENV_NAME = env
        self.kwargs = {"network": network, "seed": None}
        self.env = gym.make(self.ENV_NAME, **self.kwargs)
        self.nodes = self.env.G._nodes
        self.durations = []

        for _ in range(10000):
            self.env.reset()
            self.initial_state = self.env.state

            # print(self.initial_state)
            self.source, self.destination = self.initial_state[1], self.initial_state[2]
            self.unvisited = [True if node != self.source else False for node in self.nodes]
            self.distance = [0 if node == self.source else -np.inf for node in self.nodes]
            duration = self.__call__()
            self.durations.append(duration)
            print('Episode Duration', duration)
            print(_)
            print(self.initial_state)
        # self.Q = PriorityQueue()
        print(np.average(self.durations))

    def __call__(self):
        done = False
        current_node = self.source
        current_state = self.initial_state
        print('Current State', current_state)
        while not done:
            if not self.unvisited[self.destination.index] or current_node == self.destination:
                break
            print('Node:', current_node)
            print('Neighbours:', current_node.neighbours)
            for neighbour in current_node.neighbours:
                if self.unvisited[neighbour.index]:
                    # print(self.env._get_reward(neighbour, current_state))
                    cost = self.distance[current_node.index] + self.env._get_reward(neighbour, current_state)
                    if self.distance[neighbour.index] < cost: self.distance[neighbour.index] = cost
            node_distance = [self.distance[node.index] if self.unvisited[node.index] else -np.inf for node in
                             self.nodes]
            if all(x == -np.inf for x in node_distance):
                print('Crashed')
                quit(0)
            # print(neighbour_distance)
            min_node_distance = np.argmax(node_distance)
            self.unvisited[current_node.index] = False
            current_node = self.nodes[min_node_distance]
            current_state = [current_node, self.source, self.destination]

            # print(self.unvisited)
        self.distance = [0 if distance == -np.inf else distance for distance in self.distance]
        return min(self.distance)


import logging
import os
import sys
import random
from time import sleep
from operator import attrgetter

import gym
import numpy as np
from gym import error, spaces, utils
from gym.utils import seeding

sys.path.append(os.path.abspath("./gym_network/envs"))
from Architecture import Edge, Graph, Node
from Parser import Parser
from abc import ABCMeta, abstractmethod


class BaseEnv(gym.Env):
    """
    define a base environment for any sndlib defined network.
    """

    metadata = {"render.modes": ["human", "rgb_array"]}

    def __init__(self, network, seed):
        #define self.__version___ and self.
        self.__version__ = "1.0.0"
        logging.debug("{} OpenAI Gym Env - version {}".format(self.env_name, self.__version__))
        logging.info("Current Network: {}".format(network))

    @abstractmethod
    def step(self, action):
        raise NotImplementedError("step function is an abstract method and must be implemented in the parent class.")

    @abstractmethod
    def reset(self):
        raise NotImplementedError("reset function is an abstract method and must be implemented in the parent class.")

    @abstractmethod
    def _get_reward(self, action, state):
        raise NotImplementedError("_get_reward function is an abstract method and must be implemented in the parent class.")

    def get_link_taken(self, action, state):
        for edge in self.G._edges:
            # print("edge: ", edge._source.index, edge._destination.index)
            # print("matches: ", action.index, state[0].index)
            if (
                edge._source.index == action.index
                and edge._destination.index == state[0].index
            ) or (
                edge._source.index == state[0].index
                and edge._destination.index == action.index
            ):
                return edge

    def _get_state(self, action, state):
        state = self.G.getState(action, state)
        return state

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def createGraph(self, network="germany50"):
        # create parser object from germany50 xml:
        try:
            parser = Parser(network + ".xml")
        except Exception as e:
            logging.critical("ERROR: Likely invalid network filename passed, " + str(e))
            print("ERROR: Likely invalid network filename passed, " + str(e))
            exit()
        # create empty graph:
        graph = Graph()

        node_list = []

        # iterate through each node item gotten in parser from germany50.xml
        # each iteration we get i (the index), and the node_name and the nodes coordinate
        # we create a Node object with all of these, and append the Node object to the node_list
        for i, (node_name, coordinate) in enumerate(parser.nodes.items()):
            node_list.append(Node(i, node_name, coordinate))

        # add all the nodes in node_list to the graph:
        graph.addNodes(node_list)

        def convert(x):
            return list(parser.nodes.keys()).index(x)

        # ok so what happens here is we iterate thru each link obtained by parser, and get the links attributes (e.g. for link in parser.links.values():)
        # then for each link we make a list that has the start and end node numbers [1, 49] with list(map(convert, link[0:2]))
        # then we append the capacity of that link to that list so it looks like [1, 49, '40.0'] with +[link[2:3]]
        # we make a list called edges that consists of the list we made above for each link [[1,49,'40.0],[2,32,'40.0'],...]
        edges = [
            list(map(convert, link[0:2])) + link[2:5] for link in parser.links.values()
        ]

        def getNodes(x):
            return [
                graph._nodes[x.__getitem__(0)],  # this is the source node number
                graph._nodes[x.__getitem__(1)],  # this is the destination node number
                x.__getitem__(2), # this is the index
                x.__getitem__(3),  # this is the capacity
                x.__getitem__(4),  # this is the cost
            ]

        # we use the function above to get the source, destination, and capacity which are needed to create an Edge object
        # each Edge object is added to the graph with the addEdges function which takes a list of edges
        graph.addEdges(
            [
                Edge(source, destination, index, capacity, cost)
                for [source, destination, index, capacity, cost] in list(map(getNodes, edges))
            ]
        )
        return graph

    def getInitialState(self, num_nodes=50, seed=None):
        # by keeping seed parameter as 'None' we generate new random results each time we call the function
        # otherwise if it is kept at a constant integer (e.g. 0), we will obtain the same randomint's each function call.
        if seed == None:
            random.seed()
        else:
            random.seed(seed)
        # built in random library is inclusive for both arguments, randint(a, b) chooses x from a<=x<=b
        start_node = random.randint(0, num_nodes - 1)
        end_node = random.randint(0, num_nodes - 1)
        # reroll if the same:
        while end_node == start_node:
            end_node = random.randint(0, num_nodes - 1)

        logging.debug("Start Node: " + str(start_node) + "| End Node: " + str(end_node))
        # return (self.G._nodes[0], self.G._nodes[37])
        # state is a tuple of (curr_node, start_node, end_node)
        return (
            self.G._nodes[start_node],
            self.G._nodes[start_node],
            self.G._nodes[end_node],
        )

    def getNodeFromState(self, state):
        current_node = state[0]
        return current_node

    def getMaxActions(self):
        return max([len(node.neighbours) for node in self.G._nodes])

    def convertState(self, state):
        state = np.array([state[0].index, state[1].index, state[2].index])
        state = np.reshape(state, newshape=(1, 3))
        # state = state / 49
        [state] = state
        return state

    def checkIfDone(self, np_state):
        if np_state[0] == np_state[2]:
            return True
        else:
            return False

    def render(self, mode):

        from gym.envs.classic_control import rendering

        screen_width = 1200
        screen_height = 1000
        node_radius = 5  # must be int
        node_filled = False  # boolean
        # used for reference to scale longitude/latitude to x,y grid
        min_x_coord = 100.0
        min_y_coord = 100.0
        max_x_coord = 0.0
        max_y_coord = 0.0

        for node in self.G._nodes:
            if float(node.coordinates[0]) < min_x_coord:
                min_x_coord = float(node.coordinates[0])
            if float(node.coordinates[1]) < min_y_coord:
                min_y_coord = float(node.coordinates[1])
            if float(node.coordinates[0]) > max_x_coord:
                max_x_coord = float(node.coordinates[0])
            if float(node.coordinates[1]) > max_y_coord:
                max_y_coord = float(node.coordinates[1])

        x_coord_scale = screen_width / (max_x_coord - min_x_coord)
        y_coord_scale = screen_height / (max_y_coord - min_y_coord)

        if self.viewer is None:
            self.viewer = rendering.Viewer(screen_width, screen_height)
            # self.viewer.set_bounds(min_x_coord,max_x_coord,min_y_coord,max_y_coord)

        for edge in self.G._edges:
            start_coord = edge._source.coordinates  # tuple
            end_coord = edge._destination.coordinates  # tuple
            x_s = float(start_coord[0]) - min_x_coord
            y_s = float(start_coord[1]) - min_y_coord
            x_e = float(end_coord[0]) - min_x_coord
            y_e = float(end_coord[1]) - min_y_coord

            self.viewer.draw_line(
                start=(x_s * x_coord_scale, y_s * y_coord_scale),
                end=(x_e * x_coord_scale, y_e * y_coord_scale),
            )

        for node in self.G._nodes:
            node_color = (0.1, 0.1, 0.1)

            node_x = float(node.coordinates[0]) - min_x_coord
            node_y = float(node.coordinates[1]) - min_y_coord

            node_index = node.index
            node_circle = rendering.make_circle(radius=node_radius, filled=node_filled)
            node_circle.add_attr(
                rendering.Transform(
                    translation=(node_x * x_coord_scale, node_y * y_coord_scale),
                    scale=(1, 1),
                )
            )

            if node.visited == True:
                node_filled = True
            if node.source_node == True:
                node_color = (1.0, 0, 0)
            elif node.dest_node == True:
                node_color = (0, 1.0, 0)

            node_circle.set_color(node_color[0], node_color[1], node_color[2])

            self.viewer.add_onetime(node_circle)

        return self.viewer.render(return_rgb_array=mode == "rgb_array")


# net = NetworkEnv()
# net.__init__()
# net.render()


from functools import total_ordering


class Graph:
    def __init__(self):
        pass

    def addNodes(self, nodes):
        if all(isinstance(node, Node) for node in nodes):
            self._nodes = nodes
        else:
            raise TypeError

    def addPackets(self, packets):
        if all(isinstance(packet, Packet) for packet in packets):
            self._packets = packets
        else:
            raise TypeError

    def _checkEdges(self, edges):
        return (
            True
            if all(
                isinstance(edge, Edge)
                and edge._source
                and edge._destination in self._nodes
                for edge in edges
            )
            else False
        )

    def _processEdges(self, edges):
        #Reverses Edges, i.e. duplicates them
        edges = edges + [edge.__rev__() for edge in edges]
        edges = [
            edge for index, edge in enumerate(edges) if edge not in edges[index + 1 :]
        ]
        return edges

    def _linkNodes(self):
        for node in self._nodes:
            neighbour_list = []
            for edge in self._edges:
                if edge._source == node:
                    neighbour_list.append(edge._destination)
                elif edge._destination == node:
                    neighbour_list.append(edge._source)

            node.neighbours = sorted(neighbour_list)

    def addEdges(self, edges):
        print(len(edges))
        if self._checkEdges(edges):
            #self._edges = self._processEdges(edges)
            self._edges = edges
            print(len(self._edges))
            self._linkNodes()
        else:
            raise TypeError

    def sendPacket(self, packet):
        if all(node in self._nodes for node in [packet._source, packet._destinatioan]):
            return True
        else:
            return False

    def getActions(self, state):
        return state[0].neighbours

    def getState(self, action, past_state):
        state = [action]
        state.extend(past_state[1:])
        #print(state)
        return state

    def getReward(self, action, state):
        return -1 if action != state[2] else -1

    def terminate(self, action, state):
        return False if action != state[2] else True


class Packet:
    def __init__(self, id, state, past_state, size=1, timeToLive=100, reachedEnd=0):
        #define packet headers here:
        #according to http://www.linfo.org/packet_header.html
        self._id = id
        self._state =  state #state=(curr_node, start_node, end_node)
        self._past_state = past_state
        self._size = size #size of packet in arbitrary units
        self._timeToLive = timeToLive #number of hops before packet is allowed to expire
        self._reachedEnd = reachedEnd #0 for false, 1 for true
    def __eq__(self, other):
        return (
            True
            if self._id == other._id
            else False
        )

    def __repr__(self):
        return "Packet: {[self._state, self._size]}"


@total_ordering
class Node:
    def __init__(self, index, name, coordinates):
        if all(
            hasattr(name.__class__, attribute) for attribute in ["__lt__", "__eq__"]
        ):
            self.__name__ = name
        else:
            raise TypeError
        self.neighbours = []
        self.index = index
        self.coordinates = coordinates
        self.visited = False
        self.source_node = False
        self.dest_node = False

    def __lt__(self, other):
        return True if self.__name__ > other.__name__ else False

    def __eq__(self, other):
        return True if self.__name__ == other.__name__ else False

    def __repr__(self):
        return "Node: "+ self.__name__

class Edge:
    def __init__(self, source, destination, index, capacity=None, cost=None, traffic=0):
        if all(isinstance(link, Node) for link in [source, destination]):
            self._source = source
            self._traffic = traffic
            self._destination = destination
            self._index = index
            #defines the pre-installed capacity installed on this link
            self._capacity = capacity
            self._cost = cost
        else:
            raise TypeError

    def __eq__(self, other):
        return (
            True
            if self._source == other._source and self._destination == other._destination
            else False
        )

    def __rev__(self):
        return Edge(self._destination, self._source, self._index)

    def __repr__(self):
        return "Edge: (" +str(self._source) + " -> " +str(self._destination)+")"

    def add_capacity(self, capacity):
        self._capacity = capacity