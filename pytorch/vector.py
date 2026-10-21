import numpy as np

def basic_sigmoid(x):
    #计算sigmoid函数
    return 1/(1+np.exp(-x))

def sigmoid_derivative(x):
    #计算sigmoid函数的倒数
    return basic_sigmoid(x)*(1-basic_sigmoid(x))

if __name__=="__main__":
    print(basic_sigmoid(np.array([1,2,3])))
    print(sigmoid_derivative(np.array([1,2,3])))