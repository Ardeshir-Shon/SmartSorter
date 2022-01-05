import torch
from AgentSimple import Agent
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

numberOfEpisodes = 10000

belt = Belt(1)
buffer = Buffer(1,2)
pallet = Pallet(2)
agent = Agent(belt=belt,buffer=buffer,pallet=pallet,globalTime=globalTime)


deltaTStart = 0
deltaTEnd = 2

weightStart = 1
weightEnd = 2

exitFlag = 0

episode = 1 # Episode Counter 

with open("log.txt", "w") as log:
   log.write(" ------------------ = New Trial =  -------------------\n")

class Conveyor(threading.Thread):
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
      if not belt.isFull():
         belt.addProduct(Product(arrivalTime = globalTime.time+random.randint(deltaTStart,deltaTEnd), weight = random.randint(weightStart,weightEnd)))

factory = Conveyor(1, "Belt-Thread")

factory.start()

with open("log.csv", "a") as log:
   log.write("Episode,Episode Reward,Episode Steps,Normalized Episode Reward,Epsilon,Final Pallet Reward,Shipped Pallet\n")

while episode <= numberOfEpisodes:
   print("-------------")
   
   time.sleep(0.05)
   
   if episode >= numberOfEpisodes:
      break
   
   ### agent do action here (if needed!)
   agent.learn(episode)
   
   torch.save(agent.act_net.state_dict(), "./net.pth")
  
   print("Episode: ",episode)
   print("Episode Reward: ", agent.episodeRewards[-1])
   print("----------------------")
   
   with open("log.csv", "a") as log:
      writer = csv.writer(log, delimiter=',' , lineterminator='\n')
      writer.writerow([episode, agent.episodeRewards[-1], agent.episodeSteps[-1], agent.episodeRewards[-1]/agent.episodeSteps[-1], agent.epsilon,agent.tempWReward,agent.lastPallet])
   
   episode += 1
   
   ## tille here

   ### update graphic canvas here
   ## till here 

exitFlag = True


