from Agent import Agent
from Product import Product
from Belt import Belt
from Buffer import Buffer
from Pallet import Pallet

import random

agent = Agent()
belt = Belt(1)
buffer = Buffer(3,5)
pallet = Pallet(8)

actionAmount = 5 # times to take any action
actionTime = 0 # time took for the action

globalTime = 0

numberOfFrames = 1000

deltaTStart = 0
deltaTEnd = 5

weightStart = 1
weightEnd = 10

while globalTime <= numberOfFrames:
    
    ### agent do action here (if needed!)
    if actionTime % actionAmount == 0: # should take an action (also wait is action)
        actionTime = 0 # reseting action time
        agent.doAction()
    ## tille here

    if  not belt.isFull():
        belt.addProduct(Product(arrivalTime = globalTime+random.randint(deltaTStart,deltaTEnd), weight = random.randint(weightStart,weightEnd)))

    ### update graphic canvas here

    ## till here 

    ### update time
    globalTime += 1
    actionTime += 1


