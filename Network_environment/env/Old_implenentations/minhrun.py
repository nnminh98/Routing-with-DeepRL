from SimComponents import PacketGenerator, PacketSink, SwitchPort, PortMonitor, Packet
import simpy
from functools import partial
import logging
from numpy.random import RandomState
logger = logging.getLogger(__name__)

class Network(object):
    def __init__(self):
        self.nodes = {}
        self.links = {}

    def get_reward(self):
        total = 0
        for link in self.links:
            pass

    def add_link(self, link):
        self.links[link.id] = link

    def add_node(self, node):
        self.nodes[node.id] = node

class RoutingAlgorithm(object):
    def __init__(self, node, gen):
        self.node = node
        self.gen = gen

    def route_packet(self, packet):
        for port in self.node.output_ports.values():
            if port.id == packet.dst:
                return port
        return self.gen.choice(list(self.node.output_ports.values()))

class Link(object):
    def __init__(self, env, id, cost, src, dst, switch_port):
        self.env = env
        self.id = id
        self.cost = cost
        self.src = src
        self.dst = dst
        self.up = True
        self.switch_port = switch_port

    def send(self, packet):
        yield self.env.timeout(self.cost)
        self.dst.put(packet)

    def put(self, packet):
        self.env.process(self.send(packet))

    def __setstate__(self, state):
        self.up = state

    # Function for returning different capacity parameters
    def get_capacity(self, option):
        if option == 0:
            return self.switch_port.qlimit
        if option == 1:
            return self.switch_port.size_packet
        if option == 2:
            return self.switch_port.qlimit - self.switch_port.size_packet

    def total_cost(self):
        return self.switch_port.size_packet * (self.cost + self.switch_port.rate)

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
        self.packet_sink = PacketSink(env=self.env, rec_arrivals=True)
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
        new_link = Link(self.env, id=link_id, cost=propagation_delay, dst=dst_node, src=self.id, switch_port=new_port)
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

    def get_output_port(self, portid):
        return self.output_ports[portid]

    def get_queue_port(self, portid):
        port = self.get_output_port(portid)
        return len(port.store.items)

    def get_queues(self):
        queues_dict = {k: self.get_queue_port(k) for k in self.output_ports}
        return queues_dict

    def put(self, packet):


        self.packets_received += 1
        if packet.dst == self.id:
            self.packet_sink.put(packet)
        else:
            self.incoming_packets.put(packet)
        print("Packet " + str(packet.id) + " put into node " + str(self.id))

    def run(self):
        while True:
            packet = yield (self.incoming_packets.get())
            outgoing_port = self.route(packet)
            if self.routes[outgoing_port.id].up:
                # Increment counter in packet
                packet.increment_hops()
                packet.incrementRouteWeight(self.routes[outgoing_port.id].cost)
                packet.decrement_ttl()

                outgoing_port.put(packet)
                self.packets_sent += 1

def run():
    gen = RandomState(2)
    env = simpy.Environment()
    n1 = Node(env=env, id="n1", gen=gen)
    n2 = Node(env=env, id="n2", gen=gen)
    n3 = Node(env=env, id="n3", gen=gen)
    n4 = Node(env=env, id="n4", gen=gen)
    n5 = Node(env=env, id="n5", gen=gen)
    n6 = Node(env=env, id="n6", gen=gen)
    n1.set_routing_algo()
    n2.set_routing_algo()
    n3.set_routing_algo()
    n4.set_routing_algo()
    n5.set_routing_algo()
    n6.set_routing_algo()

    n1.add_connection(n2, rate=500, qlimit=64, monitor_rate=1, propagation_delay=5)
    n1.add_connection(n3, rate=500, qlimit=64, monitor_rate=1, propagation_delay=5)
    n2.add_connection(n4, rate=500, qlimit=64, monitor_rate=1, propagation_delay=5)
    n4.add_connection(n6, rate=500, qlimit=64, monitor_rate=1, propagation_delay=5)
    n3.add_connection(n5, rate=500, qlimit=64, monitor_rate=1, propagation_delay=5)
    n5.add_connection(n6, rate=500, qlimit=64, monitor_rate=1, propagation_delay=5)

    pkt = Packet(1, size=1, id=1, src=n1.id, dst=n6.id)
    n1.put(pkt)

    env.run(until=100)

run()