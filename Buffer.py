from Product import Product

class Buffer():
    
    def __init__(self,width:int,length:int):
        self.width = width
        self.length = length
        self.slots = [ [0]*length for i in range(width) ] 
    
    def moveToSlot(self,product:Product,x:int,y:int):
        if x>(self.width-1) or y>(self.length-1) or x<0 or y<0:
            raise Exception("X and Y are not fit to the buffer dimensions! (Or are negative! :/ )")
        self.slots[x][y] = product
    
    def moveFromSlot(self,x:int,y:int):
        if x>(self.width-1) or y>(self.length-1) or x<0 or y<0:
            raise Exception("X and Y are not fit to the buffer dimensions! (Or are negative! :/ )")
        self.slots[x][y] = 0
    
    def isSlotEmpty(self,x:int,y:int):
        if x>(self.width-1) or y>(self.length-1) or x<0 or y<0:
            raise Exception("X and Y are not fit to the buffer dimensions! (Or are negative! :/ )")
        if self.slots[x][y] == 0:
            return True
        return False
    
    def getSlotProduct(self,x:int,y:int):
        if self.isSlotEmpty(x,y):
            return 0
        if x>(self.width-1) or y>(self.length-1) or x<0 or y<0:
            raise Exception("X and Y are not fit to the buffer dimensions! (Or are negative! :/ )")
        return self.slots[x][y]
    
    def empty(self):
        self.slots = [ [0]*self.length for i in range(self.width) ] 