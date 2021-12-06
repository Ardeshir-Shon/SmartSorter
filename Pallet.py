from Product import Product

class Pallet():
    
    def __init__(self,capacity:int):
        self.capacity = capacity
        self.products = capacity*[0] # considered zero for emty slots
        self.shipTime = -1 # means not shipped yet

    def isReadyToShip(self):
        return self.products.count(0) == 0

    def addProduct(self,product:Product):
        try:
            self.products[self.products.index(0)] = product # push
            return True
        except:
            raise Exception('Could not add!')
    
    def shipThePallet(self,globalTime):
        for product in self.products:
            product.setShipped()
        self.shipTime = globalTime
    
    def getShipTime(self):
        return self.shipTime
    
    def getProducts(self):
        return self.products
