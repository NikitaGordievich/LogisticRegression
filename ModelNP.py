import numpy as np

class LogisticModel():
    def __init__(self):
        self.w=None
        self.b=None
        self.logits=None
        self.x=None

    def __call__(self,x):
        if self.w is None:
            self.w=np.zeros((x.shape[-1],1),dtype=np.float32)
            self.b=np.zeros((1,),dtype=np.float32)

        y=x@self.w+self.b
        sigmoid=1/(1+np.exp(-y))
        self.logits=sigmoid
        self.x=x
        return sigmoid

    def backward(self,y_target):
        if self.logits is None or self.x is None:
             raise NotForwardExcaption('Don`t Forward predict')
        b_grad=self.logits-y_target
        w_grad=self.x.T@b_grad
        return w_grad,np.mean(b_grad)
    
    def apply(self,grad):
        self.w=self.w-grad[0]
        self.b=self.b-grad[1]

        

class BinariCrossEntropy():
    def __init__(self):
        self.y_p=None

    def __call__(self,logits: np.ndarray,y_target: np.ndarray)->np.ndarray:
        y=np.mean(y_target*np.log(logits)+(1-y_target)*np.log(1-logits))
        self.y_p=y_target
        return y
    
    def backward(self):
        if self.y_p is None:
            raise NotForwardExcaption('Don`t Forward predict')
        return self.y_p

class NotForwardExcaption(Exception):
    def __init__(self,m):
            super().__init__(self,m)

class OptimizerSGD():
    def __init__(self,lr=0.01,Y=0.9):
        self.lr=lr
        self.Y=Y
        self.grad=[]

    def __call__(self,grad):
        for i,g in enumerate(grad): 
            if len(self.grad)!=len(grad):
                self.grad.append(np.zeros_like(g,dtype=np.float32))
            self.grad[i]=self.Y*self.grad[i]+ self.lr*grad[i]
        return self.grad
    
