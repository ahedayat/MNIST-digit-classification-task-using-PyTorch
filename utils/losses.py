import numpy as np

class MSE_loss:
    @staticmethod
    def f(y_expected, y_predicted):
        return np.power(y_expected-y_predicted,2)
    @staticmethod
    def df(y_expected, y_predicted):
        return (y_predicted-y_expected) * 2

class Hinge_loss:
    @staticmethod
    def f(y_expected, y_predicted, delta=1.0, reg_const= 0.0):
        N = y_expected.shape[0]
        margins = np.maximum(0,delta-np.multiply(y_expected,y_predicted))
        return margins.sum() / N
    @staticmethod
    def df(y_expected, y_predicted, delta=1.0):
        N = y_expected.shape[0]
        ye_x_yp= np.multiply(y_expected,y_predicted)

        dL = np.copy(y_expected)
        mask = ye_x_yp >= 1
        
        dL[mask] = 0
        dL = (-1) * dL
        dL = dL / N

        return dL

        # correct_class = np.argmax(y_expected)

        # dL = np.zeros_like(y_predicted)
        # margin = y_predicted - y_predicted[correct_class,0] + delta
        # mask = margin > 0
        # dL[mask] = 1
        # dL[~mask] = 0
        # dL[correct_class,0] = 0
        
        # return dL

class CrossEntropy:
    @staticmethod
    def f(y,a):
        return np.sum(y * np.nan_to_num(np.log(a)) - (1 - y) * np.nan_to_num(np.log(1 - a)))

    @staticmethod
    def df(y,a):
        return a - y