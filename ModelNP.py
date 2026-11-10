import numpy as np

class LogisticModel():
    def __init__(self):
        self.w=None
        self.b=None

    def __call__(self,x):
        if self.w is None:
            self.w=np.zeros((x.shape[-1],1),dtype=np.float32)
            self.b=np.zeros((1,),dtype=np.float32)

        y=x@self.w+self.b
        sigmoid=1/(1+np.exp(-y))
        return sigmoid

    def dackward():
        pass
class BinariCrossEntropy():
    def __init__(self):
        pass

    def __call__(self,p: np.ndarray,y_target: np.ndarray)->np.ndarray:

        y=
