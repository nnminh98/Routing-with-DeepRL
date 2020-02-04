from SimComponents import PacketGenerator, PacketSink, SwitchPort, RandomBrancher, PortMonitor
from minhrun import RoutingAlgorithm
import simpy
from functools import partial
import logging
from numpy.random import RandomState

class Node(object):

    def __init__(self, id, env, gen):
        self.env = env
        self.id = id
        self.packet_generator = None
        self.output_ports = {}
        self.port_monitors = {}
        self.routes = {}
        self.routing_algo = None
        self.env.process(self.run())
        self.packet_sink = PacketSink(env=self.env, rec_arrivals=True, id="{}_ps".format(self.id))
        self.gen = gen

        self.incoming_packets = simpy.Store(self.env, capacity=1)
        self.packets_received = 0
        self.packets_sent = 0

    def set_routing_algo(self):
        # put this router into the controller controller.setNode(self.id)
        self.routing_algo = RoutingAlgorithm(gen=self.gen, node=self)

    def set_packet_generator(self, lbd, possible_destinations):

        def dstdist(gen, possible_destinations):
            if possible_destinations is None:
                return None
            else:
                return gen.choice(possible_destinations)

        def next_packet_time(gen, lbd):
            return gen.exponential(lbd)

        packet_dst = partial(dstdist, self.gen, possible_destinations)
        next_pkt_time = partial(next_packet_time, self.gen, lbd)
        self.packet_generator = PacketGenerator(env=self.env, id="{}_pg".format(self.id), adist=next_pkt_time, sdist=100, dstdist=packet_dst)

        ## LOOK AT THIS AGAIN - might want to consider putting it into a switch port
        self.packet_generator.out = self

    def get_port_id(self, dst_node_id):
        return "{}_{}".format(self.id, dst_node_id)

    def get_link_id(self, dst_node_id):
        return "{}_{}".format(self.id, dst_node_id)

    def add_connection(self, dst_node, rate, qlimit, monitor_rate, propagation_delay, bidirectional=False):
        port_id = self.get_port_id(dst_node.id)
        new_port = SwitchPort(self.env, rate=rate, qlimit=qlimit, limit_bytes=False, id=port_id)
        self.output_ports[port_id] = new_port

        link_id = self.get_link_id(dst_node.id)
        new_link = Link(self.env, id=link_id, delay=propagation_delay, dst=dst_node, src=self.id, switch_port=new_port)
        self.routes[link_id] = new_link

        new_port.out = new_link

        def dist(gen, lbd):
            return gen.exponential(lbd)

        port_monitor_id = new_port.id
        dist_partial = partial(dist, self.gen, monitor_rate)
        port_monitor = PortMonitor(self.env, port=new_port, dist=dist_partial)
        self.port_monitors[port_monitor_id] = port_monitor

        if bidirectional:
            dst_node.add_connection(self, rate, qlimit, monitor_rate, propagation_delay)

    def route(self, packet):
        return self.routing_algo.route_packet(packet)

    def put(self, packet):
        self.packets_received += 1
        if packet.dst == self.id:
            self.packet_sink.put(packet)
        else:
            self.incoming_packets.put(packet)

    def run(self):
        while True:
            packet = yield (self.incoming_packets.get())
            outgoing_port = self.route(packet)
            if self.routes[outgoing_port.id].up:
                # Increment counter in packet
                packet.incrementHops()
                packet.incrementRouteWeight(self.routes[outgoing_port.id].cost)
                packet.decrement_ttl()

                outgoing_port.put(packet)
                self.packets_sent += 1

