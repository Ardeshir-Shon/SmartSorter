import torch
from AgentStable import Agent
from Product import Product
from Belt import Belt
from Buffer import Buffer
from Pallet import Pallet

import random
from matplotlib import pyplot as plt
import numpy as np
import math

import threading
import time
import csv


class Time():
   def __init__(self,time:int) -> None:
      self.time = time
   
   def increaseTime(self):
      self.time += 1

globalTime = Time(0)

numberOfEpisodes = 100000

belt = Belt(1)
buffer = Buffer(2,3)
pallet = Pallet(3)
agent = Agent(belt=belt,buffer=buffer,pallet=pallet, globalTime = globalTime)


deltaTStart = 0
deltaTEnd = 2

weightStart = 1
weightEnd = 8

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
      feedProduct(self.name, 0.05)
      print("Conveyor belt stopped ... " , self.name)

def feedProduct(threadName, delay):
   while True:
      if exitFlag:
         return
      time.sleep(delay)
      if  not belt.isFull():
         belt.addProduct(Product(arrivalTime = globalTime.time+random.randint(deltaTStart,deltaTEnd), weight = random.randint(weightStart,weightEnd)))

factory = Conveyor(1, "Belt-Thread")

factory.start()

with open("log.csv", "a") as log:
   log.write("Episode,Episode Reward,Episode Steps,Normalized Episode Reward,Epsilon,Final Pallet Reward,Quartile Reward Per Step,Shipped Pallet\n")

while episode <= numberOfEpisodes:
   print("-------------")
   
   time.sleep(0.05)
   
   if episode >= numberOfEpisodes:
      break
   
   ### agent do action here (if needed!)
   agent.learn(episode)
   
   torch.save(agent.act_net.state_dict(), "./act_net_done.pth")
   
   normalReward = np.nanquantile(agent.actionRewards,0.15) if not math.isnan(np.nanquantile(agent.actionRewards,0.15)) else agent.defaultActionReward
   
   print("Episode: ",episode)
   print("Total reward: ", agent.episodeRewards[-1]/agent.episodeSteps[-1])
   print("Quartile Reward: ",normalReward)
   print("Final Pallet Reward: ", agent.lastRWeight)
   print("----------------------")
   
   with open("log.csv", "a") as log:
      writer = csv.writer(log, delimiter=',' , lineterminator='\n')
      writer.writerow([episode, agent.episodeRewards[-1], agent.episodeSteps[-1], agent.episodeRewards[-1]/agent.episodeSteps[-1], agent.epsilon,agent.lastRWeight,normalReward,agent.lastPallet])
   
   episode += 1
   
   # belt.empty()
   # buffer.empty()
   # pallet.empty()
   
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


