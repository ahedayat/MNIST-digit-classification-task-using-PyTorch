import numpy as np

class Identity:
    @staticmethod
    def f(x):
        out = np.empty_like(x)
        out = x
        return out
    @staticmethod
    def df(x):
        return np.ones(x.shape)
class Relu:
    @staticmethod
    def f(x):
        out = np.zeros_like(x)
        mask = x > 0
        out[mask] = x[mask]
        out[~mask] = 0
        return out
    @staticmethod
    def df(x):
        out = np.zeros_like(x)
        mask = x > 0
        out[mask] = 1
        out[~mask] = 0
        return out

# class Soft_Relu:
#     @staticmethod
#     def f(x):
#         return np.log( np.add(1, np.exp(x) ) )
#     @staticmethod
#     def df(x):
#         numerator = np.exp(x)
#         denominator = np.add(1,numerator)
#         return np.divide(numerator,denominator)

class Softplus:
    @staticmethod
    def f(x):
        return np.log( np.add(1, np.exp(x) ) )
    @staticmethod
    def df(x):
        numerator = np.exp(x)
        denominator = np.add(1,numerator)
        return np.divide(numerator,denominator)

class Leaky_Relu:
    @staticmethod
    def f(x,alpha=0.2):
        out = np.empty_like(x)
        mask = x >= 0
        out[mask] = x[mask]
        out[~mask] = np.multiply(alpha,x[~mask])
        return out
    @staticmethod
    def df(x,alpha=0.2):
        out = np.empty_like(x)
        mask = x >= 0
        out[mask] = 1
        out[~mask] = alpha
        return out

class Sigmoid:
    @staticmethod
    def f(x):
        return 1.0 / (1.0 + np.exp(-x))
    @staticmethod
    def df(x):
        s = Sigmoid.f(x)
        return s * (1 - s)