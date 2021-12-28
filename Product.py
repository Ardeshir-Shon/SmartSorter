class Product():
    def __init__(self,arrivalTime,weight,rigidness=1):
        self.arrivalTime = arrivalTime 
        self.weight = weight
        self.rigidness = rigidness
        self.shipped = False
    
    def __repr__(self) -> str:
        return "|Weight:"+str(self.getWeight())+"|"

    def setShipped(self):
        self.shipped = True
    def getWeight(self):
        return self.weight
    def getArrivalTime(self):
        return self.arrivalTime
    def getRigidness(self):
        return self.rigidness
    def isShipped(self):
        return self.shipped