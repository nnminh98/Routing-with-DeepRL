import sys
import random
from abc import ABCMeta, abstractmethod


class RoutingAlgorithm(object):
    """Base class for the routing controller objects. Each node will have an instance of this object that it can query
    for direction to forward a certain packet"""
    def __init__(self, node, graph):
        self.node = node
        self.graph = graph

    @abstractmethod
    def route_packet(self, packet):
        raise NotImplementedError()


class RLRouting(RoutingAlgorithm):
    """Reinforcement Learning controller, that the agent will modify at each step of the learning process"""
    def __init__(self, node, graph):
        super().__init__(node=node, graph=graph)
        self.route_here = None

    def route_packet(self, packet):
        """This method returns the link referenced in the route_here instance variable if it exists
        :param packet: Packet object
        :return: Link object
        """
        if self.route_here is not None:
            route_here = self.route_here
            self.set_route_here(link=None)
            return route_here
        else:
            return None
            #return random.choice(list(self.node.routes.values()))

    def set_route_here(self, link):
        self.route_here = link

    def set(self, action):
        """Given an action, map it to one of the neighbouring nodes and attached link
        :param action: integer
        :return: neighbouring node, link connecting this node
        """
        actions = self.node.get_neighbours()
        selected_action = actions[action]
        selected_link = self.node.routes["{}_{}".format(self.node.id, selected_action.id)]
        self.set_route_here(link=selected_link)
        return selected_action, selected_link


class RandomRouting(RoutingAlgorithm):
    """A random routing controller that extends the base class."""
    def __init__(self, node, graph, gen):
        super().__init__(node=node, graph=graph)
        self.gen = gen

    def route_packet(self, packet):
        """This method searches for the destination node referenced in the packet's dst. If the destination node is not
        among the neighbours, it returns a random link for the node to forward the packet to
        :param packet: Packet object
        :return: Link
        """
        if not self.node.routes:
            return None
        for link in self.node.routes.values():
            if link.dst.id == packet.dst:
                return link
        link = random.choice(list(self.node.routes.values()))
        while not link.state:
            link = random.choice(list(self.node.routes.values()))
        return link


class Dijkstra(RoutingAlgorithm):
    """Dijkstra routing controller"""
    def __init__(self, node, graph):
        super().__init__(node=node, graph=graph)

    def route_packet(self, packet):
        """Returns the link for the next hop for that path computed using Dijkstra's shortest path algorithm
        :param packet: Packet object
        :return: Link object
        """
        def min_distance(distance, spt_set, self_nodes):
            """Returns the node with the minimum distance value that has not yet been added
            :param distance:
            :param spt_set:
            :param self_nodes:
            :return: Node object
            """
            minimum = sys.maxsize
            minimum_node = None
            for curr_node in self_nodes.values():
                if distance[curr_node.id] < minimum and not spt_set[curr_node.id]:
                    minimum = distance[curr_node.id]
                    minimum_node = curr_node
            return minimum_node

        if self.graph.contains_node(self.node.id) and self.graph.contains_node(packet.dst):
            src = self.graph.nodes[self.node.id]
            dst = self.graph.nodes[packet.dst]

            dist = self.graph.nodes.copy()
            for node in dist.keys():
                dist[node] = sys.maxsize
            dist[src.id] = 0

            sptSet = self.graph.nodes.copy()
            for node in sptSet.keys():
                sptSet[node] = False

            path = self.graph.nodes.copy()
            for node in path.keys():
                path[node] = []

            for count in range(len(self.graph.nodes)):
                current = min_distance(distance=dist, spt_set=sptSet, self_nodes=self.graph.nodes)
                sptSet[current.id] = True
                if current == dst:
                    break

                for v in self.graph.nodes.values():
                    if current.is_neighbour(v) and not sptSet[v.id] and dist[v.id] > dist[current.id] + current.routes["{}_{}".format(current.id, v.id)].cost:
                        if current.routes["{}_{}".format(current.id, v.id)].state:
                            dist[v.id] = dist[current.id] + current.routes["{}_{}".format(current.id, v.id)].cost
                            path[v.id] = path[current.id].copy()
                            path[v.id].append(v)

            #print("The distance between node " + str(src.id) + " and node " + str(dst.id) + " is " + str(dist[dst.id]))
            #print("The path towards destination is " + str(i for i in path[dst.id]))
            #print(path[dst.id][0].id)
            return self.node.routes["{}_{}".format(self.node.id, path[dst.id][0].id)]
        else:
            print("Either destination or source not in the graph")
            return None
