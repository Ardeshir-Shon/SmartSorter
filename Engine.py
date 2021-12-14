from Agent import Agent
from Product import Product
from Belt import Belt
from Buffer import Buffer
from Pallet import Pallet

import random
from matplotlib import pyplot as plt
import numpy as np

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

numberOfEpisodes = 1000

belt = Belt(1)
buffer = Buffer(3,5)
pallet = Pallet(8)
agent = Agent(belt=belt,buffer=buffer,pallet=pallet, globalTime = globalTime)


deltaTStart = 0
deltaTEnd = 5

weightStart = 1
weightEnd = 10

exitFlag = 0

episode = 1

with open("log.txt", "w") as log:
   log.write(" ------------------ = New Trial =  -------------------\n")

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


while episode <= numberOfEpisodes:
   print("-------------")
   time.sleep(0.1)
   
   if episode >= numberOfEpisodes:
      break
   
   ### agent do action here (if needed!)
   agent.learn(episode)
   episode += 1
   print("Episode: ",episode)
   print("Total reward: ",np.array(agent.episodeRewards[-1])/np.array(agent.episodeSteps[-1]))
   print("----------------------")
   belt.empty()
   buffer.empty()
   pallet.empty()
   ## tille here

   ### update graphic canvas here
   ## till here 

exitFlag = True


average_rewards  = np.array(agent.episodeRewards)/np.array(agent.episodeSteps)
plt.plot(average_rewards, "-o", label="Average Reward")
plt.ylabel("Average Reward")
plt.xlabel("Episodes")
plt.legend('Average Reward')
plt.savefig("rewards.png", dpi=300)
# print("Average Reward: ", len(average_rewards))


