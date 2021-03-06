from Product import Product
from queue import Queue

class Belt():
    
    def __init__(self,capacity:int):
        self.capacity = capacity
        self.products = [] # will use as a queue (I didn't like the Queue library however performance is better!)
    
    def addProduct(self,product:Product):
        try:
            self.products.append(product) # enqueue
            return True
        except:
            raise Exception('Could not add!')

    
    def grabProduct(self):
        return self.products.pop(0) # dequeue
    
    def isFull(self):
        if len(self.products) == self.capacity:
            return True
        if len(self.products) > self.capacity:
            raise Exception('You are wrongly do simulation!')
        return False
 
    def getTopArrivalTime(self):
        try:
            return self.products[0].getArrivalTime()
        except:
            return 0

    
    def getTopWeight(self):
        try:
            return self.products[0].getWeight()
        except:
            return 0
    
    def getProducts(self):
        return self.products[0:self.capacity]

    def empty(self):
        self.products = []
    