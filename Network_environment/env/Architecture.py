from SimComponents import PacketGenerator, PacketSink, SwitchPort, PortMonitor, Packet
from RoutingControllers import RandomRouting, Dijkstra, RLRouting

import simpy
from functools import partial
import logging
from numpy.random import RandomState
import random
import sys
logger = logging.getLogger(__name__)

RANDOM = 0
DIJKSTRA = 1
RL = 2


class Network(object):

    def __init__(self, env, gen):
        self.env = env
        self.gen = gen
        self.nodes = {}
        self.links = {}
        self.packets = []
        self.routing_algorithm = None

    def add_node(self, node):
        """Add one node into the network
        :param node: Node object to be added to the network
        :return:
        """
        node.set_network(self)
        self.nodes[node.id] = node

    def add_nodes(self, nodes):
        """Add a set of nodes into the network - nodes = ["n1", "n2", "n3", "n4"]
        :param nodes: Set of node names to be added
        :return:
        """
        if all(isinstance(node, str) for node in nodes):
            for node in nodes:
                if not self.contains_node(node):
                    new_node = Node(env=self.env, id=node, gen=self.gen)
                    self.add_node(new_node)

    def add_links(self, links):
        """Function for adding links to the network
        Format of links should be - [["n1" "n2" 5], ["n3" "n4" 1], ["n3" "n6" 3], ["n1" "n3" 2]]
        :param links:
        :return:
        """
        if all(isinstance(edge, list) for edge in links) and all(len(edge) == 3 for edge in links):
            for link in links:
                self.add_link(src=link[0], dst=link[1], rate=500, qlimit=64, monitor_rate=1, propagation_delay=link[2])

    def add_link(self, src, dst, rate, qlimit, monitor_rate, propagation_delay):
        """Function for adding one link to the network
        :param src:
        :param dst:
        :param rate:
        :param qlimit:
        :param monitor_rate:
        :param propagation_delay:
        :return:
        """
        if self.contains_node(src) and self.contains_node(dst):
            links = self.nodes[src].add_connection(dst_node=self.nodes[dst], rate=rate, qlimit=qlimit, monitor_rate=monitor_rate, propagation_delay=propagation_delay, bidirectional=True)
            for link in links:
                self.links[link.id] = link
        else:
            print("Graph does not contain node " + str(src) + " or node " + str(dst))

    def contains_node(self, node):
        """Check if a node is in this network
        :param node: the node we want to look for in the network
        :return: Boolean
        """
        if isinstance(node, str):
            for n in self.nodes.values():
                if n.id == node:
                    return True
        elif isinstance(node, Node):
            for n in self.nodes.values():
                if n.id == node.id:
                    return True
        return False

    def add_packet(self, pkt):
        """Function for adding a packet object into the network, putting it into the correct src node
        :param pkt: Packet object we want to add to the network
        :return: Boolean indicating if the packet insertion was successful
        """
        if isinstance(pkt, Packet) and self.contains_node(pkt.src):
            self.packets.append(pkt)
            self.nodes[pkt.src].put(pkt)
            return True
        return False

    def clear_packets(self):
        """Clear all the packets from all the nodes inside the network
        :return:
        """
        self.packets = []
        for node in self.nodes.values():
            node.clear_packets()

    def set_routing_algorithm(self, controller):
        """Setting a routing controller for all nodes in the network
        :param controller: controller type - "dijkstra", "random" or "RL"
        :return: None
        """
        if not isinstance(controller, str):
            self.routing_algorithm = controller
        else:
            for node in self.nodes.values():
                node.set_routing_algorithm(controller=controller)

    def get_reward(self):
        route_length = 0
        for node in self.nodes.values():
            route_length += node.packet_sink.total_weight
        return 1/(route_length * len(self.packets))

    def run(self):
        pass


class Link(object):

    def __init__(self, env, id, cost, src, dst):
        self.env = env
        self.id = id
        self.cost = cost
        self.src = src
        self.dst = dst
        self.state = True

    def send(self, packet):
        yield self.env.timeout(self.cost)
        if self.state:
            self.dst.put(packet)
        else:
            print("Link is down")

    def put(self, packet):
        self.env.process(self.send(packet))

    def __setstate__(self, state):
        if isinstance(state, bool):
            self.state = state
        else:
            raise TypeError


class Node(object):

    def __init__(self, id, env, gen):
        self.env = env
        self.id = id
        self.packet_generator = None
        self.gen = gen
        self.routes = {}
        self.routing_algorithm = None
        self.network = None
        self.env.process(self.run())
        self.packet_sink = PacketSink(env=self.env, rec_arrivals=True)
        self.packets_to_send = simpy.Store(self.env, capacity=10)
        self.packets_sent = 0

        self.switch_port = SwitchPort(env=self.env, rate=50, qlimit=64, limit_bytes=False, id=self.id)
        self.switch_port.out = self.packets_to_send
        #self.port_monitor = PortMonitor(env, self.switch_port, dist=1)

    def set_network(self, network):
        """ Set the network in which the node is placed in
        :param network: network
        :return: None
        """
        self.network = network
        return

    def set_routing_algorithm(self, controller="RL"):
        """Set routing algorithm for this node
        :param controller:
        :return:
        """
        if controller == "random":
            self.routing_algorithm = RandomRouting(gen=self.gen, node=self, graph=self.network)
        elif controller == "dijkstra":
            self.routing_algorithm = Dijkstra(graph=self.network, node=self)
        elif controller == "RL":
            self.routing_algorithm = RLRouting(node=self, graph=self.network)
        else:
            pass

    def route(self, packet):
        return self.routing_algorithm.route_packet(packet)

    def set_packet_generator(self, lbd, possible_destinations):
        """
        :param lbd:
        :param possible_destinations:
        :return:
        """
        def dst_dist(gen, destinations):
            if destinations is None:
                return None
            else:
                return gen.choice(destinations)

        def next_packet_time(gen, lbd_):
            return gen.exponential(lbd_)

        packet_dst = partial(dst_dist, self.gen, possible_destinations)
        next_pkt_time = partial(next_packet_time, self.gen, lbd)
        self.packet_generator = PacketGenerator(env=self.env, id="{}_pg".format(self.id), adist=next_pkt_time, sdist=100, dstdist=packet_dst)

        ## LOOK AT THIS AGAIN - might want to consider putting it into a switch port
        self.packet_generator.out = self.packets_to_send

    def get_port_id(self, dst_node_id):
        return "{}_{}".format(self.id, dst_node_id)

    def get_link_id(self, dst_node):
        return "{}_{}".format(self.id, dst_node.id)

    def add_connection(self, dst_node, rate, qlimit, monitor_rate, propagation_delay, bidirectional=True):
        """Add a new connection to this node given the following set of parameters
        :param dst_node:
        :param rate:
        :param qlimit:
        :param monitor_rate:
        :param propagation_delay:
        :param bidirectional:
        :return:
        """
        link_id = self.get_link_id(dst_node)
        new_link = Link(self.env, id=link_id, cost=propagation_delay, dst=dst_node, src=self)
        self.routes[link_id] = new_link

        if bidirectional:
            new_link_reverse = dst_node.add_connection(self, rate=rate, qlimit=qlimit, monitor_rate=monitor_rate, propagation_delay=propagation_delay, bidirectional=False)
            return [new_link, new_link_reverse[0]]

        return [new_link]

    def id_str_to_num(self):
        return int(self.id[1])

    def get_neighbours(self):
        """Get neighbours of this node
        :return: list of neighbouring nodes sorted by id
        """
        neighbours = []
        neighbour_id = self.get_neighbour_id()

        for name in neighbour_id:
            link_name = "{}_{}".format(self.id, name)
            neighbours.append(self.routes[link_name].dst)

        return neighbours

    def get_neighbour_id(self):
        """Get all id's of neighbours of this node
        :return: sorted list of of neighbour ids
        """
        neighbour_id = []

        for link in self.routes.values():
            if link.src.id == self.id:
                neighbour = link.dst
            else:
                neighbour = link.src
            neighbour_id.append(neighbour.id)

        neighbour_id.sort()
        return neighbour_id

    def is_neighbour(self, node):
        """
        Check if a certain node is a neighbour of this node
        :param node: Node name (str) to be searched for among the neighbours
        :return: Boolean indicating if the node is found
        """
        if node == self:
            return False

        for link in self.routes.values():
            if link.dst == node or link.src == node:
                return True
        return False

    def get_link(self, node):
        return self.routes["{}_{}".format(self.id, node.id)]

    def clear_packets(self):
        """
        Clear all packets from the node's switch port
        :return: None
        """
        self.switch_port.clear_packets()

    def put(self, packet):
        if not isinstance(packet, Packet):
            return
        packet.set_current_node(self)
        if packet.dst == self.id:
            self.packet_sink.put(packet)
        else:
            self.switch_port.put(packet)
        print("Packet " + str(packet.id) + " put into node " + str(self.id))

    def run(self):
        while True:
            packet = yield (self.packets_to_send.get())
            print("Running")
            outgoing_port = self.route(packet)
            if outgoing_port is not None and self.routes[outgoing_port.id].state:
                packet.increment_hops()
                packet.incrementRouteWeight(self.routes[outgoing_port.id].cost)
                packet.decrement_ttl()

                outgoing_port.put(packet)
                self.packets_sent += 1
            else:
                self.put(packet)


def run_this():
    gen = RandomState(2)
    env = simpy.Environment()

    pkt = Packet(1, size=1, id=1, src="n2", dst="n7")
    pkt2 = Packet(1, size=1, id=2, src="n2", dst="n5")
    pkt3 = Packet(1, size=1, id=3, src="n2", dst="n6")

    myNetwork = Network(env=env, gen=gen)
    myNetwork.add_nodes(["n1", "n2", "n3", "n4", "n5", "n6", "n7"])
    myNetwork.set_routing_algorithm(controller="random")

    links = [["n1", "n2", 1], ["n1", "n3", 1], ["n1", "n4", 1], ["n3", "n2", 1], ["n4", "n6", 1], ["n3", "n6", 2], ["n4", "n7", 1], ["n4", "n5", 1], ["n6", "n7", 1], ["n5", "n7", 10]]
    myNetwork.add_links(links)

    myNetwork.add_packet(pkt)
    myNetwork.add_packet(pkt2)
    myNetwork.add_packet(pkt3)

    def step(env2):
        env2.run(until=env2.now+1)

    for i in range(30):
        print("Queue size is " + str(len(env._queue)))
        step(env2=env)
        #print(pkt.current_node.id)
        if i == 3:
            myNetwork.set_routing_algorithm(controller="dijkstra")
        print("next")

    #print("next")
    #myNetwork.set_routing_algorithm(controller="random")

    #print(myNetwork.get_reward())

#run_this()

