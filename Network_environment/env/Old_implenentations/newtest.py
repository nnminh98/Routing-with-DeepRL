from multiprocessing import Process

def loop_a():
   for i in range(5):
      print("a")

def loop_b():
   for i in range(5):
      print("b")

Process(target=loop_a).start()
Process(target=loop_b).join()