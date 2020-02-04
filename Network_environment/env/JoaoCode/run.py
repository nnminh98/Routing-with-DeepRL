from SimComponents import PacketGenerator, PacketSink, SwitchPort, PortMonitor, Packet
import simpy
from functools import partial
import logging
from numpy.random import RandomState
logger = logging.getLogger(__name__)

class Link:
    """
    Do calculations between throughput and latency to assert that packets can never exceed a certain amount
    """
    def __init__(self, env, id, propagation_delay, dst_node):
        self.env = env
        self.id = id
        self.propagation_delay = propagation_delay
        self.dst_node = dst_node
        self.in_flight = 0

    def _send(self, packet):
        self.in_flight += 1
        yield self.env.timeout(self.propagation_delay)
        self.dst_node.put(packet)
        self.in_flight -= 1

    def put(self, packet):
        logger.debug(f"Packet in {self.__class__.__name__} {self.id} at {self.env.now}: " + str(packet))
        self.env.process(self._send(packet))

class Node:

    def __init__(self, env, id, gen, routing_controller):
        self.env = env
        self.id = id
        self.in_packets = simpy.Store(self.env, capacity =1)
        self.outgoing_ports = {}
        self.port_monitors = {}
        self.links = {}
        self.env.process(self.run())
        self.gen = gen
        self.packet_generator = None
        self.routing_controller = self._connect_to_controller(routing_controller)
        self.packet_sink = PacketSink(env=self.env, rec_arrivals=True, id="{}_ps".format(self.id))
        self.packets_rec = 0
        self.packets_sent_to_port = 0  #Packets sent to a port but still ave to go trough the queue
        self.route_time = 0

    def _connect_to_controller(self, controller):
        controller.register_node(self.id)
        return controller

    def set_packet_generator(self, lbd, possible_destinations=None):
        self.packet_generator = self._create_packet_generator(lbd, possible_destinations)
        self.packet_generator.out = self
        return self.packet_generator

    def _create_packet_generator(self, lbd, possible_destinations):

        def dstdist(gen, possible_destinations):
            if possible_destinations is None:
                return None
            else:
                return gen.choice(possible_destinations)

        def next_packet_time(gen, lbd):
            return gen.exponential(lbd)

        def packet_size():
            return 100

        dstdist_partial = partial(dstdist, self.gen, possible_destinations)
        next_packet_time_partial = partial(next_packet_time, self.gen, lbd)
        return PacketGenerator(env=self.env, id="{}_pg".format(self.id), adist=next_packet_time_partial, sdist=packet_size, dstdist=dstdist_partial)

    def _get_port_id(self, dst_node_id):
        return "{}_{}".format(self.id, dst_node_id)

    def _get_link_id(self, dst_node_id):
        return "link_{}_{}".format(self.id, dst_node_id)

    def _add_outgoing_port(self, dst_node, rate, qlimit):
        outgoing_port_id = self._get_port_id(dst_node.id)
        port = self._create_switchport(rate, qlimit, outgoing_port_id=outgoing_port_id)
        self.outgoing_ports[outgoing_port_id] = port
        return port

    def _create_switchport(self, rate, qlimit, outgoing_port_id):
        return SwitchPort(self.env, rate=rate, qlimit=qlimit, limit_bytes=False, id=outgoing_port_id)

    def _create_link(self, dst_node, propagation_delay, linkid):
        link = Link(env=self.env, id=linkid, propagation_delay=propagation_delay, dst_node=dst_node)
        return link

    def _add_link(self, dst_node, propagation_delay):
        linkid = self._get_link_id(dst_node.id)
        link = self._create_link(dst_node, propagation_delay, linkid)
        self.links[linkid] = link
        return link

    def _add_connection(self, dst_node, rate, qlimit, propagation_delay):
        new_port = self._add_outgoing_port(dst_node, rate, qlimit)
        new_link = self._add_link(dst_node, propagation_delay)
        new_port.out = new_link
        self.routing_controller.register_connection(self.id, new_port.id)

        return new_port

    def add_connection(self, dst_node, rate, qlimit, monitor_rate, propagation_delay):
        new_port = self._add_connection(dst_node=dst_node, rate=rate, qlimit=qlimit, propagation_delay=propagation_delay)
        new_port_monitor = self._add_portmonitor(new_port, monitor_rate)

        return new_port

    def _add_portmonitor(self, port, monitor_rate):
        outgoing_port_id = port.id
        portmonitor = self._create_portmonitor(port, monitor_rate)
        self.port_monitors[outgoing_port_id] = portmonitor
        return portmonitor

    def _create_portmonitor(self, port, monitor_rate):
        pmdist = self._get_pmdist(monitor_rate)
        return PortMonitor(self.env, port, dist=pmdist)

    def _get_pmdist(self, monitor_rate):
        def dist(gen, lbd):
            return gen.exponential(lbd)

        dist_partial = partial(dist, self.gen, monitor_rate)

        return dist_partial

    def _route(self, packet):
        outgoing_port_id = self.routing_controller.route_packet(nodeid = self.id, outgoing_ports=list(self.outgoing_ports.keys()), stats=self._get_stats_to_controller(), packet=packet)
        outgoing_port = self.get_outgoing_port(outgoing_port_id)
        return outgoing_port

    def get_queue_port(self, portid):
        port = self.get_outgoing_port(portid)
        return len(port.store.items)

    def get_queues(self):
        queues_dict = {k: self.get_queue_port(k) for k in self.outgoing_ports}
        return queues_dict

    def get_portmonitor(self, portid):
        return self.port_monitors[portid]

    def get_outgoing_port(self, portid):
        return self.outgoing_ports[portid]

    def _get_stats_to_controller(self):
        stats = {}
        stats["queue_size"] = self.get_queues()
        return stats

    def run(self):
        while True:
            packet = (yield self.in_packets.get())
            outgoing_port = self._route(packet)
            print("Sending packet from " + self.id + " to port " + outgoing_port.id)
            outgoing_port.put(packet)
            self.packets_sent_to_port += 1

    def put(self, packet):
        self.packets_rec += 1
        logger.debug(f"Packet received in {self.__class__.__name__} {self.id} at {self.env.now}: " + str(packet))
        if packet.dst == self.id:
            return self.packet_sink.put(packet)
        else:
            return self.in_packets.put(packet)

class RandomController:

    def __init__(self, gen):
        self.nodes = {}
        self.gen = gen
        self.node = None

    def route_packet(self, nodeid, outgoing_ports, stats, packet):
        return self.gen.choice(outgoing_ports)

    def get_action(self, state):
        return self.agent.get_action(state)

    def signal_end(self):
        pass
    def register_node(self, node):
        self.node = node

    def register_connection(self, nodeid, outgoing_port_id):
        pass


def run():
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    gen =RandomState(2)
    env = simpy.Environment()
    controller = RandomController(gen)
    n1 = Node(env=env, id="n1", gen=gen, routing_controller=controller)
    n1.set_packet_generator(lbd=1, possible_destinations=["n4"])
    n2 = Node(env=env, id="n2", gen=gen, routing_controller=controller)
    n3 = Node(env=env, id="n3", gen=gen, routing_controller=controller)
    n4 = Node(env=env, id="n4", gen=gen, routing_controller=controller)

    n1.add_connection(dst_node=n2, rate=500, qlimit=64, monitor_rate=1, propagation_delay=5)
    n1.add_connection(dst_node=n3, rate=500, qlimit=64, monitor_rate=1, propagation_delay=5)

    n2.add_connection(dst_node=n1, rate=500, qlimit=64, monitor_rate=1, propagation_delay=5)
    n2.add_connection(dst_node=n4, rate=500, qlimit=64, monitor_rate=1, propagation_delay=5)

    n3.add_connection(dst_node=n1, rate=500, qlimit=64, monitor_rate=1, propagation_delay=5)
    n3.add_connection(dst_node=n4, rate=500, qlimit=64, monitor_rate=1, propagation_delay=5)

    n4.add_connection(dst_node=n2, rate=500, qlimit=64, monitor_rate=1, propagation_delay=5)
    n4.add_connection(dst_node=n3, rate=500, qlimit=64, monitor_rate=1, propagation_delay=5)


    env.run(until=100)

run()

