import numpy as np
import h5py
import scipy
from PIL import Image
from scipy import ndimage
from data import load_dataset

train_x, train_y, test_x, test_y, classes = load_dataset()
print ("训练集的样本数: ", train_x.shape[0])
print ("测试集的样本数: " , test_x.shape[0])
print ("train_x形状: ", train_x.shape)
print ("train_y形状: ", train_y.shape)
print ("test_x形状: ", test_x.shape)
print ("test_x形状: ", test_y.shape)

train_y

train_x = train_x.reshape(train_x.shape[0], -1).T
test_x = test_x.reshape(test_x.shape[0], -1).T
print(train_x.shape, test_x.shape)

train_x = train_x/255.
test_x = test_x/255.


def basic_sigmoid(x):
    """
    计算sigmoid函数
    """

    ### 开始
    s = 1 / (1 + np.exp(-x))
    ### 结束

    return s


def initialize_with_zeros(shape):
    """
    创建一个形状为 (shape, 1) 的w参数和b=0.
    return:w, b
    """

    ### 开始
    w = np.zeros((shape, 1))
    b = 0

    ### 结束

    assert (w.shape == (shape, 1))
    assert (isinstance(b, float) or isinstance(b, int))

    return w, b


def propagate(w, b, X, Y):
    """
    参数：w,b,X,Y：网络参数和数据
    Return:
    损失cost、参数W的梯度dw、参数b的梯度db
    """
    m = X.shape[1]

    # 前向传播
    ### 开始
    # w (n,1), x (n, m)
    A = basic_sigmoid(np.dot(w.T, X) + b)
    # 计算损失
    cost = -1 / m * np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A))
    ### 结束
    dz = A - Y
    dw = 1 / m * np.dot(X, dz.T)
    db = 1 / m * np.sum(dz)
    # 反向传播
    ### 开始

    ### 结束
    assert (dw.shape == w.shape)
    assert (db.dtype == float)
    cost = np.squeeze(cost)
    assert (cost.shape == ())

    grads = {"dw": dw,
             "db": db}

    return grads, cost


def optimize(w, b, X, Y, num_iterations, learning_rate):
    """
    参数：
    w:权重,b:偏置,X特征,Y目标值,num_iterations总迭代次数,learning_rate学习率
    Returns:
    params:更新后的参数字典
    grads:梯度
    costs:损失结果
    """

    costs = []

    for i in range(num_iterations):

        # 梯度更新计算函数
        ### 开始
        grads, cost = propagate(w, b, X, Y)

        # 取出两个部分参数的梯度
        dw = grads['dw']
        db = grads['db']

        # 按照梯度下降公式去计算
        w = w - learning_rate * dw
        b = b - learning_rate * db

        ### 结束

        if i % 100 == 0:
            costs.append(cost)
        if i % 100 == 0:
            print("损失结果 %i: %f" % (i, cost))
            print(b)

    params = {"w": w,
              "b": b}

    grads = {"dw": dw,
             "db": db}

    return params, grads, costs


def predict(w, b, X):
    '''
    利用训练好的参数预测

    return：预测结果
    '''

    m = X.shape[1]
    Y_prediction = np.zeros((1, m))
    w = w.reshape(X.shape[0], 1)

    # 计算结果
    ### 开始
    A = basic_sigmoid(np.dot(w.T, X) + b)
    ### 结束

    for i in range(A.shape[1]):

        ### 开始
        if A[0, i] <= 0.5:
            Y_prediction[0, i] = 0
        else:
            Y_prediction[0, i] = 1
        ### 结束

    assert (Y_prediction.shape == (1, m))

    return Y_prediction


def model(X_train, Y_train, X_test, Y_test, num_iterations=2000, learning_rate=0.5):
    """
    """

    ### 开始

    # 初始化参数
    w, b = initialize_with_zeros(X_train.shape[0])

    # 梯度下降
    # params:更新后的网络参数
    # grads:最后一次梯度
    # costs:每次更新的损失列表
    params, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate)

    # 获取训练的参数
    # 预测结果
    w = params['w']
    b = params['b']

    Y_prediction_train = predict(w, b, X_train)
    Y_prediction_test = predict(w, b, X_test)
    ### 结束

    # 打印准确率
    print("训练集准确率: {} ".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
    print("测试集准确率: {} ".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))

    d = {"costs": costs,
         "Y_prediction_test": Y_prediction_test,
         "Y_prediction_train": Y_prediction_train,
         "w": w,
         "b": b,
         "learning_rate": learning_rate,
         "num_iterations": num_iterations}

    return d