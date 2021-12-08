from Product import Product
from Belt import Belt
from Buffer import Buffer
from Pallet import Pallet

from collections import defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F

## action features function
## normalization of the features
## feeding the neural network
## calculating the reward
## choosing the best action

class Agent():
    
    def __init__(self,belt:Belt,buffer:Buffer,pallet:Pallet,method="Q-Learning"):
        self.belt = belt
        self.buffer = buffer
        self.pallet = pallet
        self.method = method
        # self.qvalues = {}
        self.stateValues = defaultdict(0)
        actions = (self.belt.capacity+1)*(self.buffer.length*self.buffer.width+1)*[0]
    
    def doAction(self):
        print("choose action and get reward!")
    
    def calculateReward(self,action):
        print("calculate reward!")
    
    
    def getStateFeatures(self):
        print('get all state features!')
    
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
                    features.append(0) # it is not correct I think!
                    features.append(0) # same as top one; not correct!
        
        products = self.pallet.getProducts()
        for p in products:
            if isinstance(p,Product):
                    features.append(p.getArrivalTime())
                    features.append(p.getWeight())
            else:
                features.append(0) # it is not correct I think!
                features.append(0) # same as top one; not correct!
        
        return features
                
    def getBestAction(self,stateFeatures,model):
        print("should calculate best value based on given model and stateFeatures by examining different actions")
    
    def getPossibleActions(self):
        # Use logic arrays ... ie: [self.belt.slots>0] [self.buffer.slots==0]
        print("get possible actions!")

    def moveBufferToPallet(self,sourceX:int,sourceY:int):
        print("move!")
    
    def moveBeltToPallet(self):
        print("move again!")
    
    def moveBeltToBuffer(self,destX:int,destY:int):
        print("move again again!!")
    
    def generateState(self,features:list): # tuple can be a key in dictionary
        return tuple(features)
    
    def interpretState(self,stateCode:tuple):
        return list(stateCode)


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # 5*5 from image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.linear(self.fc1(x))
        x = F.linear(self.fc2(x))
        x = self.fc3(x)
        return x
