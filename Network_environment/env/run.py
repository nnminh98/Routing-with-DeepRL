from SinglePacketRoutingEnv import SinglePacketRoutingEnv

nodes = ["n0", "n1", "n2", "n3", "n4", "n5", "n6", "n7"]
print("Starting")
links = [["n1", "n2", 1], ["n1", "n0", 1], ["n0", "n2", 1], ["n3", "n0", 1], ["n2", "n5", 2], ["n3", "n5", 1], ["n4", "n3", 1], ["n4", "n6", 10], ["n6", "n3", 1], ["n4", "n7", 1],  ["n6", "n7", 1],  ["n5", "n6", 1]]
myEnv = SinglePacketRoutingEnv(nodes=nodes, edges=links)
state = [myEnv.state[0].id, myEnv.state[1].id, myEnv.state[2].id]
print(state)
myEnv.step(0)
for i in range(20):
    myEnv.step(0)
state = [myEnv.state[0].id, myEnv.state[1].id, myEnv.state[2].id]
print(state)

