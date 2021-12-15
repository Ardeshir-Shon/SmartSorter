from Product import Product

class Pallet():
    
    def __init__(self,capacity:int):
        self.capacity = capacity
        self.products = capacity*[0] # considered zero for emty slots
        self.shipTime = -1 # means not shipped yet

    def isReadyToShip(self):
        for product in self.products:
            if not isinstance(product,Product):
                return False
        return True
        # return self.products.count(0) == 0

    def addProduct(self,product:Product):
        # try:
        self.products[self.products.index(0)] = product # push
        return True
        # except:
        #     raise Exception('Could not add!')
    
    def shipThePallet(self,globalTime):
        for product in self.products:
            product.setShipped()
        self.products = self.capacity*[0]
        self.shipTime = globalTime
    
    def getShipTime(self):
        return self.shipTime
    
    def getProducts(self):
        return self.products
    
    def getTopProductIndex(self):
        try:
            self.products.index(0)
        except:
            return self.capacity-1
        return self.products.index(0)-1 if self.products.index(0) != 0 else 0
    
    def empty(self):
        self.products = self.capacity*[0] # considered zero for emty slots
        self.shipTime = -1 # means not shipped yet
