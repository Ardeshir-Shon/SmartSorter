from Agent import Agent
from Product import Product
from Belt import Belt
from Buffer import Buffer
from Pallet import Pallet

import random

import threading
import time


class Time():
   def __init__(self,time:int) -> None:
      self.time = time
   
   def increaseTime(self):
      self.time += 1

actionAmount = 5 # times to take any action
actionTime = 0 # time took for the action

globalTime = Time(0)

numberOfFrames = 10000

belt = Belt(1)
buffer = Buffer(3,5)
pallet = Pallet(8)
agent = Agent(belt=belt,buffer=buffer,pallet=pallet, globalTime = globalTime)


deltaTStart = 0
deltaTEnd = 5

weightStart = 1
weightEnd = 10

exitFlag = 0

episodes = 4000
episode = 1


class Conveyor (threading.Thread):
   def __init__(self, threadID, name):
      threading.Thread.__init__(self)
      self.threadID = threadID
      self.name = name
   def run(self):
      print("Conveyor belt started ... " , self.name)
      feedProduct(self.name, 0.3)
      print("Conveyor belt stopped ... " , self.name)

def feedProduct(threadName, delay):
   while True:
      if exitFlag:
         return
      time.sleep(delay)
      if  not belt.isFull():
         # print("9832749824729847298479832479249837498274")
         belt.addProduct(Product(arrivalTime = globalTime.time+random.randint(deltaTStart,deltaTEnd), weight = random.randint(weightStart,weightEnd)))
      # print(threadName," time is:" ,time.ctime(time.time()))

factory = Conveyor(1, "Belt-Thread")

factory.start()


while globalTime.time <= numberOfFrames:
   time.sleep(0.1)
   if episode >= episodes:
      break
   ### agent do action here (if needed!)
   if actionTime % actionAmount == 0: # should take an action (also wait is action)
      actionTime = 0 # reseting action time
      agent.learn(episode)
      episode += 1
   ## tille here

   ### update graphic canvas here
   print("Start**************")
   print("Belt:")
   print(len(belt.products),belt.products)
   print("-------------")
   print("Buffer:")
   print(buffer.slots)
   print("-------------")
   print("Pallet:")
   print(pallet.products)
   print("End***************")
   ## till here 

   ### update time
   globalTime.increaseTime()
   actionTime += 1

exitFlag = True


