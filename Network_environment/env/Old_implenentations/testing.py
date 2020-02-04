import random
import functools
import simpy

from SimComponents import PacketGenerator, PacketSink, SwitchPort, RandomBrancher, Packet
from Node import NetworkNode

if __name__ == '__main__':

    env = simpy.Environment()
    mean_pkt_size = 100.0  # in bytes
    port_rate = 2.2 * 8 * mean_pkt_size
    adist1 = functools.partial(random.expovariate, 2.0)
    sdist = functools.partial(random.expovariate, 1.0 / mean_pkt_size)
    samp_dist = functools.partial(random.expovariate, 0.50)
    '''switch_port = SwitchPort(env, port_rate*2)
    switch_port2 = SwitchPort(env, port_rate*2)

    for i in range(3):
        packet = Packet(env.now, mean_pkt_size, i)
        switch_port.put(packet)
        print(switch_port.getQueueSize())

    print("something")
    switch_port.out = switch_port2
    switch_port.run()'''

    node1 = NetworkNode(env, "NW1", port_rate, adist1, sdist, samp_dist)
    node2 = NetworkNode(env, "NW2", port_rate, adist1, sdist, samp_dist)
    node3 = NetworkNode(env, "NW3", port_rate, adist1, sdist, samp_dist)
    node4 = NetworkNode(env, "NW4", port_rate, adist1, sdist, samp_dist)
    node5 = NetworkNode(env, "NW5", port_rate, adist1, sdist, samp_dist)
    node1.addPort(node2, True)
    node1.addPort(node3, True)
    node3.addPort(node4, True)
    node2.addPort(node4, True)
    node2.addPort(node5, True)
    node4.addPort(node5, True)
    print(node1.getPorts())
    print(node2.getPorts())
    print(node3.getPorts())
    print(node4.getPorts())
    packet = Packet(env.now, mean_pkt_size, 1, "NW1", "NW4")
    #packet2 = Packet(env.now, mean_pkt_size, 1, "NW2", "NW1")
    node1.put(packet)
    #node2.put(packet2)
    env.run(until=40000)
