import numpy as np

class LogisticModel():
    def __init__(self):
        self.w=None
        self.b=None
        self.logits=None

    def __call__(self,x):
        if self.w is None:
            self.w=np.zeros((x.shape[-1],1),dtype=np.float32)
            self.b=np.zeros((1,),dtype=np.float32)

        y=x@self.w+self.b
        sigmoid=1/(1+np.exp(-y))
        self.logits=sigmoid
        return sigmoid

    def dackward(self,y_target):
        if self.logits is None:
             raise NotForwardExcaption('Don`t Forward predict')
        return self.logits-y_target
        

class BinariCrossEntropy():
    def __init__(self):
        self.y_p=None

    def __call__(self,logits: np.ndarray,y_target: np.ndarray)->np.ndarray:
        y=y_target*np.log(logits)+(1-y_target)*np.log(1-logits)
        self.y_p=y_target
        return y
    
    def backward(self):
        if self.y_p is None:
            raise NotForwardExcaption('Don`t Forward predict')
        return self.y_p

class NotForwardExcaption(Exception):
    def __init__(self,m):
            super().__init__(self,m)
