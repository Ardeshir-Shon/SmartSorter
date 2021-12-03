from Product import Product
from Belt import Belt
from Buffer import Buffer
from Pallet import Pallet

class Agent():
    
    def __init__(self,belt:Belt,buffer:Buffer,pallet:Pallet,method="Q-Learning"):
        self.belt = belt
        self.buffer = buffer
        self.pallet = pallet
        self.method = method
        self.qvalues = {}
    
    def moveBufferToPallet(self,sourceX:int,sourceY:int):
        print("move!")
    
    def moveBeltToPallet(self):
        print("move again!")
    
    def moveBeltToBuffer(self,destX:int,destY:int):
        print("move again again!!")
    
    def generateState(self):
        print ("Should generate statecode with a two-way function!")
    
    ## to be continued ...

    