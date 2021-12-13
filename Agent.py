from matplotlib.pyplot import get_current_fig_manager, step
import random

from numpy.core.fromnumeric import product
from numpy.lib.function_base import copy

from Product import Product
from Belt import Belt
from Buffer import Buffer
from Pallet import Pallet
from Net2 import Net
from copy import copy as clone

from collections import defaultdict

import argparse
import pickle
from collections import namedtuple
from itertools import count

import os, time
import numpy as np
import matplotlib.pyplot as plt
import math

import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal, Categorical
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

class Agent():

    def __init__(self,belt:Belt,buffer:Buffer,pallet:Pallet,globalTime,capacity = 1024,
            learning_rate = 1e-3,learn_counter=0,memory_counter = 0,batch_size = 256,gamma = 0.95,
            update_count = 0, epsilon = 0.95,Q_network_evaluation=100, time_penalty_coefficient = 1, weight_penalty_coefficient = 1, actionAmount = 2):
        
        self.belt = belt
        self.buffer = buffer
        self.pallet = pallet
        self.globalTime = globalTime
        self.stateValues = defaultdict(lambda:0)
        self.actions = ((self.belt.capacity+1)*(self.buffer.length*self.buffer.width)+1)*[0]
        self.actionAmount = actionAmount
        
        self.num_state_features = 1 + 2 * ( 1 + self.buffer.length*self.buffer.width + self.pallet.capacity)
        self.num_action = (1+1)*((self.buffer.length*self.buffer.width)+1)

        self.time_penalty_coefficient = time_penalty_coefficient
        self.weight_penalty_coefficient = weight_penalty_coefficient
        self.historical_time_rewards = [np.nan]*1000
        self.historical_weight_rewards = [np.nan]*1000

        self.capacity = capacity
        self.learning_rate = learning_rate
        self.memory_counter = memory_counter
        self.learn_counter = learn_counter
        self.batch_size = batch_size
        self.gamma = gamma
        self.update_count = update_count
        self.done_episodes = 0
        self.epsilon = epsilon
        self.Q_network_evaluation = Q_network_evaluation
        self.episodeRewards = []
        self.episodeSteps = []
        self.memory = np.zeros((self.capacity, self.num_state_features *2 +2))

        self.target_net, self.act_net = Net(self.num_state_features,self.num_action), Net(self.num_state_features,self.num_action)
        # self.memory = [None]*self.capacity
        self.optimizer = optim.Adam(self.act_net.parameters(), self.learning_rate)
        self.loss = nn.MSELoss()
    
    def getStateFeatures(self):
        features = []
        
        features.append(self.belt.capacity)
        
        features.append(self.belt.getTopArrivalTime()-self.globalTime.time)
        features.append(self.belt.getTopWeight())
        
        for i in range(self.buffer.length):
            # print("i ",i)
            for j in range(self.buffer.width):
                # print("j ",j)
                p = self.buffer.getSlotProduct(i,j)
                
                if isinstance(p,Product):
                    features.append(p.getArrivalTime()-self.globalTime.time)
                    features.append(p.getWeight())
                else:
                    features.append(0)
                    features.append(0)
        
        products = self.pallet.getProducts()
        for p in products:
            if isinstance(p,Product):
                    features.append(p.getArrivalTime()-self.globalTime.time)
                    features.append(p.getWeight())
            else:
                features.append(0)
                features.append(0)
        
        return features
    
    def getPossibleActions(self,state):
        possibleActions = []
        
        # check feasible actions from belt to buffer or pallet
        if state[2] != 0: # belt is not empty
            for i in range(self.buffer.length):
                for j in range(self.buffer.width):
                    be = (i*self.buffer.length+j)+1
                    if state[3+ 2*(be-1) + 1] != 0: # buffer at i and j is not empty
                        possibleActions.append(0)
                    else:
                        possibleActions.append(1)
            possibleActions.append(1) # from belt to pallet directly
        
        else: # there is no possible move from belt to the buffer or pallet
            for i in range(self.buffer.length*self.buffer.width+1):
                possibleActions.append(0) # belt is empty
        
        # check feasible actions from buffer to pallet
        for i in range(self.buffer.length):
            for j in range(self.buffer.width):
                be = (i*self.buffer.length+j)+1
                if state[3+ 2*(be-1) + 1] != 0: # buffer at i and j is not empty
                    possibleActions.append(1) # can move to the pallet
                else:
                    possibleActions.append(0)

        # add wait action
        possibleActions.append(1)

        return possibleActions
    
    def store_trans(self, state, action, reward, next_state):
        if self.memory_counter % 500 ==0:
            print("The experience pool collects {} time experience".format(self.memory_counter))
        index = self.memory_counter % self.capacity
        trans = np.hstack((np.array(state), np.array(action), np.array(reward), np.array(next_state)))
        self.memory[index,] = trans
        self.memory_counter += 1

    def choose_action(self, state):
        possiblesActions = self.getPossibleActions(state)
        state = torch.unsqueeze(torch.FloatTensor(state) ,0)
        if np.random.randn() <= self.epsilon:
            action_value = self.act_net.forward(state)
            # print("action value:",action_value)
            for i in range(len(possiblesActions)):
                if possiblesActions[i]==0:
                    action_value[0,i] = -np.inf
            action = torch.argmax(action_value).data.numpy()
        else:
            action = random.choice(np.argwhere(np.array(possiblesActions)==1))[0]
            # action = np.random.randint(0,self.num_action)
        return action

    def moveBufferToPallet(self,sourceX:int,sourceY:int):
        product = self.buffer.getSlotProduct(sourceX,sourceY)
        self.pallet.addProduct(product=product)
    
    def moveBeltToPallet(self):
        product = self.belt.grabProduct()
        self.pallet.addProduct(product=product)
    
    def moveBeltToBuffer(self,destX:int,destY:int):
        product = self.belt.grabProduct()
        self.buffer.moveToSlot(product=product,x=destX,y=destY)

    def palletReward(self):
        r_time = 0
        r_weight = 0
        
        p = clone(self.pallet)
        products = p.getProducts()
        
        self.pallet.shipThePallet(globalTime=self.globalTime.time)
        
        for i in range(p.capacity):
            r_time += self.pallet.getShipTime() - products[i].getArrivalTime()
        for i in range(p.capacity-1):
            top_weights = 0
            for j in range(i+1,p.capacity):
                top_weights += products[j].getWeight()
            r_weight += products[i].getWeight() - top_weights
        
        self.historical_time_rewards[self.done_episodes%1000] = r_time
        self.historical_weight_rewards[self.done_episodes%1000] = r_weight

        r_time_median = np.nanmedian(self.historical_time_rewards)
        r_weight_median =np.nanmedian(self.historical_weight_rewards)
        reward = self.time_penalty_coefficient*(r_time/r_time_median)+self.weight_penalty_coefficient*(r_weight/r_weight_median)
        
        self.done_episodes += 1

        return reward
        
    def nextStateReward(self,action):
        bufferCapacity = self.buffer.length*self.buffer.width
        widthBuffer = self.buffer.width
        if action < bufferCapacity: # belt to buffer
            self.moveBeltToBuffer(int(action/widthBuffer),action%widthBuffer)
            nextState = self.getStateFeatures()
            reward = 0
        elif action == bufferCapacity: # belt to pallet
            self.moveBeltToPallet()
            nextState = self.getStateFeatures()
            if self.pallet.isReadyToShip():
                reward = self.palletReward()
            else:
                reward = 0
        elif action > bufferCapacity and action <= 2*bufferCapacity: # buffer to pallet
            self.moveBufferToPallet(int((action-bufferCapacity-1)/widthBuffer),(action-bufferCapacity-1)%widthBuffer)
            nextState = self.getStateFeatures()
            if self.pallet.isReadyToShip():
                reward = self.palletReward()
            else:
                reward = 0
        elif action == 2*bufferCapacity+1: # wait
            # nextState = self.doWait()
            nextState = self.getStateFeatures()
            reward = 0
        return nextState,reward
            
    def update(self):
        # learn 100 times then the target network update
        if self.learn_counter % self.Q_network_evaluation == 0:
            self.target_net.load_state_dict(self.act_net.state_dict())
        self.learn_counter += 1

        sample_index = np.random.choice(self.capacity, self.batch_size)
        batch_memory = self.memory[sample_index, :]
        batch_state = torch.FloatTensor(batch_memory[:, :self.num_state_features])
        #note that the action must be a int
        batch_action = torch.LongTensor(batch_memory[:, self.num_state_features:self.num_state_features+1].astype(int))
        batch_reward = torch.FloatTensor(batch_memory[:, self.num_state_features+1: self.num_state_features+2])
        batch_next_state = torch.FloatTensor(batch_memory[:, -self.num_state_features:])

        q_eval = self.act_net(batch_state).gather(1, batch_action)
        q_next = self.target_net(batch_next_state).detach()
        q_target = batch_reward + self.gamma*q_next.max(1)[0].view(self.batch_size, 1)

        loss = self.loss(q_eval, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


    def learn(self,episode):
        # for episode in range(self.num_episodes):
        state = self.getStateFeatures()
        epsiodeDuration = random.randint(32,128)
        print("number of steps are:",epsiodeDuration)
        episodeReward = 0
        steps = 0
        myfile = open("log.txt", "a")
        myfile.write("------------ episode: "+str(episode)+"\n")
        for t in range(epsiodeDuration):
            time.sleep(0.05)
            myfile.write("Current time: "+str(self.globalTime.time)+"\n")
            if self.globalTime.time % self.actionAmount == 0:
                steps += 1
                action = self.choose_action(state)
                next_state, reward = self.nextStateReward(action)
                self.store_trans(state, action, reward, next_state)
                episodeReward += reward
                string = "***NEW ACTION:::\n"+str(steps)+"/"+str(int(epsiodeDuration/self.actionAmount))
                string += "\nTime: " + str(self.globalTime.time) 
                string += "\nBelt: " + str(self.belt.products) 
                string += "\nBuffer: " + str(self.buffer.slots) 
                string += "\nPallet: " + str(self.pallet.products) + "\n"
                myfile.write(string)
                myfile.flush()
                print(string)

                if self.memory_counter >= self.capacity:
                    self.update()
                    if t == epsiodeDuration-1:
                        print("episode {}, the reward is {}".format(episode, round(reward, 3)))
                if t == epsiodeDuration-1:
                    break
                state = next_state
            self.globalTime.increaseTime()
        myfile.write("@@@@@@@@@@ episode reward ########: "+str(episodeReward/steps)+"\n")
        self.episodeRewards.append(episodeReward)
        self.episodeSteps.append(steps)
        myfile.close()