import numpy as np

class L2_Reg:
    @staticmethod
    def f(weights):
        l_i = [ np.power(layer_weights,2).sum() for layer_weights in weights]
        return sum(l_i) / 2   

    @staticmethod
    def df(weights):
        dL = np.copy(weights)
        return dL

class Zero_Reg:
    @staticmethod
    def f(weights):
        return 0

    @staticmethod
    def df(weights):
        dL = [ np.zeros_like(layer_weights) for layer_weights in weights]
        return dL