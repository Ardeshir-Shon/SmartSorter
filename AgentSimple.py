from matplotlib.pyplot import get_current_fig_manager, step
import random
import math

from numpy.core.fromnumeric import product
from numpy.lib.function_base import copy

from Product import Product
from Belt import Belt
from Buffer import Buffer
from Pallet import Pallet
from Net import Net
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

    def __init__(self, belt:Belt, buffer:Buffer, pallet:Pallet, globalTime, capacity = 64,
            learning_rate = 1e-3,learn_counter=0,memory_counter = 0,batch_size = 512,gamma = 0.95,
            epsilon = 0.7, Q_network_evaluation=32, actionAmount = 2):
        
        self.belt = belt
        self.buffer = buffer
        self.pallet = pallet
        self.globalTime = globalTime
        #self.stateValues = defaultdict(lambda:0)
        #self.actions = ((self.belt.capacity+1)*(self.buffer.length*self.buffer.width)+1)*[0]
        self.actionAmount = actionAmount # time for actions
        
        self.num_state_features = 2 * ( 1 + self.buffer.length*self.buffer.width + self.pallet.capacity)
        self.num_action = (1+1)*((self.buffer.length*self.buffer.width)+1)

        #self.finalPallet = []

        self.capacity = capacity # Memory capactiy
        self.learning_rate = learning_rate
        self.memory_counter = memory_counter
        self.learn_counter = learn_counter
        self.batch_size = batch_size
        self.gamma = gamma
        #self.update_count = update_count
        #self.done_episodes = 0
        self.epsilon = epsilon
        self.Q_network_evaluation = Q_network_evaluation # when target net is updated from acting net
        self.episodeRewards = []
        self.episodeSteps = []
        self.memory = np.zeros((self.capacity, self.num_state_features * 2 + 2))
        self.isGreedy = True # strictly for logging

        if os.path.isfile("./net.pth"):
            print("loaded from existing models ...")
            self.target_net, self.act_net = Net(self.num_state_features,self.num_action), Net(self.num_state_features,self.num_action)
            self.act_net.load_state_dict(torch.load("./net.pth"))
            self.target_net.load_state_dict(torch.load("./net.pth"))
        else:
            print("start from random weights ...")
            self.target_net, self.act_net = Net(self.num_state_features,self.num_action), Net(self.num_state_features,self.num_action)
        # self.memory = [None]*self.capacity
        self.optimizer = optim.Adam(self.act_net.parameters(), self.learning_rate)
        self.loss = nn.MSELoss()
    
    def getStateFeatures(self):
        features = []
        
        products = self.belt.getProducts()
        
        if len(products) != 0:
            for p in products:
                if isinstance(p,Product):
                        # features.append(p.getArrivalTime()-self.globalTime.time)
                        features.append(0)
                        features.append(p.getWeight())
                else:
                    features.append(0)
                    features.append(0)
        else:
            features.append(0)
            features.append(0)

        for i in range(self.buffer.width):
            for j in range(self.buffer.length):
                p = self.buffer.getSlotProduct(i,j)
                
                if isinstance(p,Product):
                    # features.append(p.getArrivalTime()-self.globalTime.time)
                    features.append(0)
                    features.append(p.getWeight())
                else:
                    features.append(0)
                    features.append(0)
        
        products = self.pallet.getProducts()
        for p in products:
            if isinstance(p,Product):
                    # features.append(p.getArrivalTime()-self.globalTime.time)
                    features.append(0)
                    features.append(p.getWeight())
            else:
                features.append(0)
                features.append(0)
        
        return features
    
    def getPossibleActions(self,state):
        possibleActions = []
        if state[1] != 0: # if belt is not empty
            i = 3
            while i < 2+ 2*self.buffer.length*self.buffer.width:
                if state[i] == 0: # buffer slot is not empty
                    possibleActions.append(1)
                else:
                    possibleActions.append(0)
                i += 2
            possibleActions.append(1)
        else:
            i = 3
            while i < 2+ 2*self.buffer.length*self.buffer.width:
                possibleActions.append(0)
                i += 2
            possibleActions.append(0)
        
        i=3
        while i < 2+ 2*self.buffer.length*self.buffer.width:
            if state[i] != 0: # If not empty
                possibleActions.append(1) # Put in pallet
            else:
                possibleActions.append(0) #cannot put in the pallet
            i += 2
        possibleActions.append(1) # for wait
        
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
            self.isGreedy = True
        else:
            self.isGreedy = False
            action = random.choice(np.argwhere(np.array(possiblesActions)==1))[0]
        return action

    def moveBufferToPallet(self,sourceX:int,sourceY:int):
        product = clone(self.buffer.getSlotProduct(sourceX,sourceY))
        self.buffer.moveFromSlot(sourceX,sourceY)
        self.pallet.addProduct(product=product)
    
    def moveBeltToPallet(self):
        product = self.belt.grabProduct()
        self.pallet.addProduct(product=product)
    
    def moveBeltToBuffer(self,destX:int,destY:int):
        product = self.belt.grabProduct()
        self.buffer.moveToSlot(product=product,x=destX,y=destY)
        
    def calculateReward(self):
        r_weight = 0

        topProductIndex = self.pallet.getTopProductIndex()
        p = clone(self.pallet)
        products = p.getProducts()

        if topProductIndex == 0: # Only one product in pallet
            pass

        elif topProductIndex == self.pallet.capacity-1: # Pallet Full
            self.lastPallet = clone(products)
            #self.done_episodes += 1
            self.pallet.shipThePallet(globalTime=self.globalTime.time)

        for i in range(topProductIndex):
            top_weights = 0
            for j in range(i+1,topProductIndex+1):
                top_weights += products[j].getWeight()
            r_weight += products[i].getWeight() - top_weights

        reward = r_weight
        return reward

    def calculateSingleReward(self):
        topProductIndex = self.pallet.getTopProductIndex()

        p = clone(self.pallet)
        products = p.getProducts()

        topProductIndex = self.pallet.getTopProductIndex()

        r_weight = products[topProductIndex-1].getWeight() - products[topProductIndex].getWeight()
        
        reward = r_weight
        return reward
        
    def nextStateReward(self,action):
        reward = -0.5 # default reward
        done =False
        bufferCapacity = self.buffer.length*self.buffer.width
        widthBuffer = self.buffer.length
        actionRewardCoefficient = 0.5
        if action < bufferCapacity: # belt to buffer
            self.moveBeltToBuffer(int(action/widthBuffer),action%widthBuffer)
        elif action == bufferCapacity: # belt to pallet
            self.moveBeltToPallet()
            if self.pallet.isReadyToShip():
                done = True
                reward = self.calculateReward()
            else:
                #reward = actionRewardCoefficient*self.calculateSingleReward()
                pass
        elif action > bufferCapacity and action <= 2*bufferCapacity: # buffer to pallet
            self.moveBufferToPallet(int((action-bufferCapacity-1)/widthBuffer),(action-bufferCapacity-1)%widthBuffer)
            if self.pallet.isReadyToShip():
                done = True
                reward = self.calculateReward()
            else:
                #reward = actionRewardCoefficient*self.calculateSingleReward()
                pass
        elif action == 2*bufferCapacity+1: # wait
            #reward = -0.5
            pass
        
        nextState = self.getStateFeatures()
        return nextState, reward, done
            
    
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
        
        ## Compute L1 and L2 loss component
        # parameters = []
        # for parameter in self.act_net.parameters():
        #     parameters.append(parameter.view(-1))
        # l1 = 0.03 * self.act_net.compute_l1_loss(torch.cat(parameters))
        ## l2 = l2_weight * mlp.compute_l2_loss(torch.cat(parameters))

        loss = self.loss(q_eval, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


    def learn(self,episode):
        # for episode in range(self.num_episodes):
        
        self.epsilon = self.epsilon * 1.00005 if self.epsilon < 0.95 else 0.95
    
        bufferCapacity = self.buffer.length*self.buffer.width # temp for logging
        epsiodeDuration = random.randint((self.buffer.width*self.buffer.length+self.pallet.capacity+self.belt.capacity)*self.actionAmount+64,256)
        # print("number of steps are:",epsiodeDuration)
        episodeReward = 0
        steps = 0
        
        myfile = open("log.txt", "a")
        
        myfile.write("\n------------ episode: "+str(episode)+ " Epsilon: " + str(self.epsilon) +" ------------ \n")
        
        state = self.getStateFeatures()

        for t in range(epsiodeDuration):
            time.sleep(0.02)
            
            #state = self.getStateFeatures()

            if self.globalTime.time % self.actionAmount == 0:
                myfile.write("\nState: "+str(state)+"\n")
                myfile.write("Possible Actions: "+str(self.getPossibleActions(state))+"\n")
                myfile.write("New Action "+ str(steps)+"/"+str(int(epsiodeDuration/self.actionAmount)) +" at time: "+str(self.globalTime.time)+"\n")
                
                steps += 1
                
                myfile.write("Belt Before: "+str(self.belt.products)+"\n")
                myfile.write("Buffer Before: "+str(self.buffer.slots)+"\n")
                myfile.write("Pallet Before: "+str(self.pallet.products)+"\n")
                
                action = self.choose_action(state)
                
                if action < bufferCapacity:
                    selectedAction = 'Belt to Buffer'
                elif action == bufferCapacity:
                    selectedAction = 'Belt to Pallet'
                elif action > bufferCapacity and action <= 2*bufferCapacity:
                    selectedAction = 'Buffer to Pallet'
                elif action == 2*bufferCapacity+1:
                    selectedAction = 'Wait'

                myfile.write("Selected Action: "+str(selectedAction)+" : "+str(action)+"\n")
                myfile.write("isGreedy: "+ str(self.isGreedy)+"\n")
                
                next_state, reward , done = self.nextStateReward(action)
                self.store_trans(state, action, reward, next_state)
                episodeReward += reward

                # myfile.write("r_time_median: "+str(r_time_median)+" r_weight_median: "+str(r_weight_median)+"\n")
                myfile.write("Action Reward: "+str(reward)+"\n")
                myfile.write("Belt After: "+str(self.belt.products)+"\n")
                myfile.write("Buffer After: "+str(self.buffer.slots)+"\n")
                myfile.write("Pallet After: "+str(self.pallet.products)+"\n")
                myfile.flush()

                if self.memory_counter >= self.capacity:
                    self.update()
                    # if t == epsiodeDuration-1:
                    #     print("episode {}, the reward is {}".format(episode, round(reward, 3)))
                if done or t == epsiodeDuration-1:
                    myfile.write("Shipped Pallet: "+str(self.lastPallet)+"\n")
                    break
                state = next_state
            self.globalTime.increaseTime()
        myfile.write("\n@@@@@@@@@@ Episode Reward : "+str(episodeReward)+"\n")
        myfile.write("@@@@@@@@@@ Episode Steps : "+str(steps)+"\n")
        myfile.write("@@@@@@@@@@ Normalized Episode Reward : "+str(episodeReward/steps)+"\n")
        self.episodeRewards.append(episodeReward)
        self.episodeSteps.append(steps)
        myfile.close()
        