from Product import Product
from Belt import Belt
from Buffer import Buffer
from Pallet import Pallet

from collections import defaultdict

class Agent():
    
    def __init__(self,belt:Belt,buffer:Buffer,pallet:Pallet,method="Q-Learning"):
        self.belt = belt
        self.buffer = buffer
        self.pallet = pallet
        self.method = method
        self.qvalues = {}
        self.stateValues = defaultdict(0)
    
    def doAction(self):
        print("choose action and get reward")
    
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
                    features.append(-1) # it is not correct I think!
                    features.append(-1) # same as top one; not correct!
        
        products = self.pallet.getProducts()
        for p in products:
            if isinstance(p,Product):
                    features.append(p.getArrivalTime())
                    features.append(p.getWeight())
            else:
                features.append(-1) # it is not correct I think!
                features.append(-1) # same as top one; not correct!
        
        return features
                

    
    def moveBufferToPallet(self,sourceX:int,sourceY:int):
        print("move!")
    
    def moveBeltToPallet(self):
        print("move again!")
    
    def moveBeltToBuffer(self,destX:int,destY:int):
        print("move again again!!")
    
    def generateState(self):
        print ("Should generate statecode with a two-way function!")
    
    def interpretState(self,stateCode:int):
        print ("Should generate statecode with a two-way function!")

    def doAction(self,statecode:int):
        print ("Should do action with a statecode!")
    
    ## to be continued ...