from matplotlib.pyplot import get_current_fig_manager
import random

from numpy.core.fromnumeric import product

from Product import Product
from Belt import Belt
from Buffer import Buffer
from Pallet import Pallet
from Net import Net

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
from tensorboardX import SummaryWriter

class Agent():

    def __init__(self,belt:Belt,buffer:Buffer,pallet:Pallet,capacity = 1024,
            learning_rate = 1e-3,learn_counter=0,memory_counter = 0,batch_size = 256,gamma = 0.995,
            update_count = 0,num_episodes = 2000, epsilon = 0.95,Q_network_evaluation=100):
        
        self.belt = belt
        self.buffer = buffer
        self.pallet = pallet
        self.stateValues = defaultdict(lambda:0)
        self.actions = ((self.belt.capacity+1)*(self.buffer.length*self.buffer.width)+1)*[0]
        
        self.num_state_features = 1 + 2 * ( 1 + self.buffer.length*self.buffer.width + self.pallet.capacity)
        self.num_action = (1+1)*((self.buffer.length*self.buffer.width)+1)

        self.capacity = capacity
        self.learning_rate = learning_rate
        self.memory_counter = memory_counter
        self.learn_counter = learn_counter
        self.batch_size = batch_size
        self.gamma = gamma
        self.update_count = update_count
        self.num_episodes = num_episodes
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
        
        features.append(self.belt.getTopArrivalTime())
        features.append(self.belt.getTopWeight())
        
        for i in range(self.buffer.length):
            for j in range(self.buffer.width):
                p = self.buffer.getSlotProduct(i,j)
                
                if isinstance(p,Product):
                    features.append(p.getArrivalTime())
                    features.append(p.getWeight())
                else:
                    features.append(0)
                    features.append(0)
        
        products = self.pallet.getProducts()
        for p in products:
            if isinstance(p,Product):
                    features.append(p.getArrivalTime())
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
        trans = np.hstack((state, [action], [reward], next_state))
        self.memory[index,] = trans
        self.memory_counter += 1

    def choose_action(self, state):
        state = torch.unsqueeze(torch.FloatTensor(state) ,0)
        if np.random.randn() <= self.epsilon:
            action_value = self.act_net.forward(state)
            action = torch.argmax(action_value).data.numpy()
        else:
            action = np.random.randint(0,self.num_action)
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
        print("reward for the shipped pallet")

    
    def doWait(self):
        print("we just waited!")
    
    def nextStateReward(self,action):
        bufferCapacity = self.buffer.length*self.buffer.width
        widthBuffer = self.buffer.width
        if action < bufferCapacity: # belt to buffer
            nextState = self.moveBeltToBuffer(int(action/widthBuffer),action%widthBuffer)
            reward = 0
        elif action == bufferCapacity: # belt to pallet
            self.moveBeltToPallet()
            nextState = self.getStateFeatures()
            if self.pallet.isReadyToShip():
                reward = self.palletReward()
            else:
                reward = 0
        elif action > bufferCapacity and action <= 2*bufferCapacity: # buffer to pallet
            nextState = self.moveBufferToPallet(int((action-bufferCapacity-1)/widthBuffer),(action-bufferCapacity-1)%widthBuffer)
            if self.pallet.isReadyToShip():
                reward = self.palletReward()
            else:
                reward = 0
        elif action == 2*bufferCapacity+1: # wait
            nextState = self.doWait()
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


    def learn(self):
        for episode in range(self.num_episodes):
            state = self.getStateFeatures()
            steps = random.randint(16,320)
            print("number of steps are:",steps)
            episodeReward = 0
            for t in range(steps):
                action = self.choose_action(state)
                next_state, reward = self.nextStateReward(action)
                self.store_trans(state, action, reward, next_state)
                episodeReward += reward
                if self.memory_counter >= self.capacity:
                    self.update()
                    if t == steps-1:
                        print("episode {}, the reward is {}".format(episode, round(reward, 3)))
                if t == steps-1:
                    break
                state = next_state
            
            self.episodeRewards.append(episodeReward/steps)
            self.episodeSteps.append(steps)
    
    def doAction(self):
        print("choose action and get reward!")
