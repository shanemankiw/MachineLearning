#-*- coding:UTF-8 -*-
import numpy as np
import os
import time
import datetime as dt
import pdb
import matplotlib
from sklearn.decomposition import PCA
from sklearn.datasets import fetch_mldata
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.externals import joblib
from skimage.transform import rotate
from skimage.transform import swirl
from sklearn.preprocessing import PolynomialFeatures
from skimage.feature import hog
from mnist_helpers import *



def PrincipalComponents(n, X_train_pca, X_test_pca):
    pca = PCA(n_components=n, whiten=True)  # 保留n个主成分，打开whiten
    X_train1 = pca.fit_transform(X_train_pca)   # 训练PCA，用作训练集
    X_test1 = pca.transform(X_test_pca)

    # 注意这个pca对象需要保留
    # 后面数据增强部分新增的数据也需要进行同样的PCA处理
    return X_train1, X_test1, pca


# 对数据增强部分用PCA降维
def PrincipalComponents_aug(X_aug, pca):
    X_aug = pca.transform(X_aug)
    return X_aug


if __name__ == "__main__":

    r'''

    数据准备阶段：

    '''

    # 随机种子生成，后面可能会用到
    np.random.seed(42)

    # 从url获取数据；此处我已经下好在地址下面，所以不会再从网络上获取，应该是直接得到
    mnist = fetch_mldata('MNIST original', data_home='./')

    # mnist数据的获取
    # 前60k行是训练集；测试集是后1k行
    X, y = mnist["data"], mnist["target"]
    X_train_copy = X[:60000].copy()  # 原始数据的一份拷贝
    y_train_copy = y[:60000].copy()
    X_test_copy = X[60000:].copy()
    y_test_copy = y[60000:].copy()

    # 求取样本方差
    # 这个> 1100 是得到一个true and false 矩阵
    # 用于移除低方差特征
    variance = np.var(X, axis=0) > 1100
    X = X[:, variance]

    X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

    # m是X_train的一维整形
    m = X_train.shape[0]

    # 获取shuffle的index
    # 打乱训练样本顺序
    shuffle_index = np.random.permutation(60000)
    X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]

    r'''

    PCA准备阶段：

    '''

    # 用于准备PCA之后的训练集
    X_train_pca, X_test_pca = X_train.copy(), X_test.copy()

    # 将28*28维特征降低到n维
    n = 35
    print('N for pca is ' + str(n))

    # 使用pca，并且保存pca对象
    #X_train, X_test, pca = PrincipalComponents(n, X_train_pca, X_test_pca)

    '''
    # 调用多处理器并行计算最优的参数取值
    gsc = GridSearchCV(
        estimator=SVC(),
        param_grid={'C': [1, 3, 5, 7, 10]},
        cv=3,   # 使用3个交叉验证集
        verbose=2,
        n_jobs=-1,  # 自适应使用全部CPU核
        return_train_score=True
    )

    # 对gsc进行训练
    #gsc.fit(X_train[:10000], y_train[:10000])

    # gsc.fit(X_train, y_train)
    #C_chosen = gsc.best_params_
    #print(gsc.best_params_)

    # finally I choose the best from
    # my experiments
    '''
    C_chosen = 2.8

    r'''

    训练模块一：
    使用原始的60k训练集进行训练

    '''

    tic = time.time()

    # 初始化svm（使用多分类的SVC）
    svm = SVC(kernel='rbf', C=C_chosen,gamma=.05,random_state=42,verbose=True)
    print('Training 60k...')

    svm.fit(X_train, y_train)  # 训练
    print('Training 60k Complete')

    toc = time.time()
    print("time for training 60k is " + str((toc - tic)) + " seconds")

    #joblib.dump(svm, "n_" + str(n) + "_C_28_SVM_60k.pkl", compress=3)  # 保存模型

    # 进行测试
    # y_pred_train=svm.predict(X_train)
    y_pred_test = svm.predict(X_test)

    # 打印准确率
    # print(accuracy_score(y_train, y_pred_train))
    print(metrics.accuracy_score(y_test, y_pred_test))

    # 打印分类报告和混淆矩阵
    print("Classification report for 60k classifier %s:\n%s\n"
          % (svm, metrics.classification_report(y_test, y_pred_test)))
    cm = metrics.confusion_matrix(y_test, y_pred_test)
    print("Confusion matrix for 60k :\n%s" % cm)

    
    plot_confusion_matrix(cm)
    

    print("Accuracy for 60k ={}".format(metrics.accuracy_score(y_test, y_pred_test)))
