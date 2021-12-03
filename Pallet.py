from Product import Product

class Pallet():
    
    def __init__(self,capacity:int):
        self.capacity = capacity
        self.products = [] # I will use it as a stack
        self.shipTime = -1 # means not shipped yet

    def addProduct(self,product:Product):
        try:
            self.products.append(product) # push
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
