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


# 我们的情况实际上是，为什么我们实际上的
def PrincipalComponents(n):
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


# 平移扩增进行数据增强
def add_shift(X_train_copy, y_train_copy, X_train, y_train, pca):
    X_reshape = np.reshape(X_train_copy, (1680000, 28))  # combine arrays

    fig = plt.figure(figsize=(10, 25))
    plt.subplot(151)
    plt.imshow(X_reshape[:112, :], cmap=matplotlib.cm.binary)
    # shift up
    X_shiftU = np.delete(X_reshape, 0, 0)
    X_shiftU = np.vstack((X_shiftU, np.zeros((1, 28))))
    plt.subplot(152)
    plt.imshow(X_shiftU[:112, :], cmap=matplotlib.cm.binary)
    X_shiftU = np.reshape(X_shiftU, (60000, 784))
    # shift down
    X_shiftD = np.delete(X_reshape, 1679999, 0)
    X_shiftD = np.vstack((np.zeros((1, 28)), X_shiftD))
    plt.subplot(153)
    plt.imshow(X_shiftD[:112, :], cmap=matplotlib.cm.binary)
    X_shiftD = np.reshape(X_shiftD, (60000, 784))
    # shift right
    X_shiftR = np.delete(X_reshape, 27, 1)
    X_shiftR = np.hstack((np.zeros((1680000, 1)), X_shiftR))
    plt.subplot(154)
    plt.imshow(X_shiftR[:112, :], cmap=matplotlib.cm.binary)
    X_shiftR = np.reshape(X_shiftR, (60000, 784))
    # shift left
    X_shiftL = np.delete(X_reshape, 0, 1)
    X_shiftL = np.hstack((X_shiftL, np.zeros((1680000, 1))))
    plt.subplot(155)
    plt.imshow(X_shiftL[:112, :], cmap=matplotlib.cm.binary)
    X_shiftL = np.reshape(X_shiftL, (60000, 784))

    # concatenate
    X_total = np.vstack((X_shiftU, X_shiftD, X_shiftR, X_shiftL))
    X_total = X_total[:, variance]
    X_total = PrincipalComponents_aug(X_total, pca)
    X_total = np.vstack((X_train, X_total))
    y_total = np.hstack(
        (y_train,
         y_train_copy,
         y_train_copy,
         y_train_copy,
         y_train_copy))

    shuffle_index = np.random.permutation(X_total.shape[0])
    X_total = X_total[shuffle_index]
    y_total = y_total[shuffle_index]
    return X_total, y_total


# 旋转扩增进行数据增强
def add_rotated(
        angle,
        X_train_copy,
        y_train_copy,
        X_train,
        y_train,
        X_total,
        y_total,
        pca):
    X_new = X_train_copy
    m_new = 60000

    random1 = np.array([np.random.choice([-1, 1], size=m_new)])
    random1 = random1.T
    X_rot = np.array(np.zeros((m_new, 28, 28)))
    X_rot[0] = rotate(
        X_new[0].reshape(
            28,
            28),
        random1[0] *
        angle,
        preserve_range=True)

    plt.subplot(221)
    plt.imshow(X_train_copy[0].reshape((28, 28)), cmap=matplotlib.cm.binary)
    plt.subplot(222)
    plt.imshow(X_rot[0], cmap=matplotlib.cm.binary)

    for i in range(1, m_new):
        X_rot[i] = rotate(
            X_new[i].reshape(
                28,
                28),
            random1[i] *
            angle,
            preserve_range=True)

    plt.subplot(223)
    plt.imshow(X_train_copy[40000].reshape(
        (28, 28)), cmap=matplotlib.cm.binary)
    plt.subplot(224)
    plt.imshow(X_rot[40000], cmap=matplotlib.cm.binary)

    X_rot = np.reshape(X_rot, (m_new, 784))
    X_rot = X_rot[:, variance]
    X_rot = PrincipalComponents_aug(X_rot, pca)
    X_total = np.vstack((X_total, X_rot))
    y_total = np.hstack((y_total, y_train_copy))

    shuffle_index = np.random.permutation(X_total.shape[0])

    X_total = X_total[shuffle_index]
    y_total = y_total[shuffle_index]

    return X_total, y_total



# 效果可视化使用到的函数
def plot_digits(instances, images_per_row=10, **options):
    size = 28
    images_per_row = min(len(instances), images_per_row)
    images = [instance.reshape(size, size) for instance in instances]
    n_rows = (len(instances) - 1) // images_per_row + 1
    row_images = []
    n_empty = n_rows * images_per_row - len(instances)
    images.append(np.zeros((size, size * n_empty)))
    for row in range(n_rows):
        rimages = images[row * images_per_row: (row + 1) * images_per_row]
        row_images.append(np.concatenate(rimages, axis=1))
    image = np.concatenate(row_images, axis=0)
    plt.imshow(image, cmap=matplotlib.cm.binary, **options)
    plt.axis("off")


if __name__ == "__main__":

    '''

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

    '''

    PCA准备+阶段：

    '''

    # 用于准备PCA之后的训练集
    X_train_pca, X_test_pca = X_train.copy(), X_test.copy()

    # 将28*28维特征降低到n维
    n = 35

    # 使用pca，并且保存pca对象
    X_train, X_test, pca = PrincipalComponents(n=n)

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
    gsc.fit(X_train[:10000], y_train[:10000])

    # gsc.fit(X_train, y_train)
    print(gsc.best_params_)

    '''

    训练模块一：
    使用原始的60k训练集进行训练

    '''

    tic = time.time()

    # 初始化svm（使用多分类的SVC）
    svm = SVC(kernel='rbf', C=5, gamma=.05, random_state=42)
    print('Training 60k...')

    svm.fit(X_train, y_train)  # 训练
    print('Training 60k Complete')

    toc = time.time()
    print("time for training 60k  is " + str((toc - tic)) + " seconds")

    joblib.dump(svm, "n_30_C_28_SVM_60k.pkl", compress=3)  # 保存模型

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


    '''
    
    训练模块二：
    移位数据增强240k，得到数据集共300k作为训练数据集
    
    '''
    # 移位数据增强
    X_total, y_total = add_shift(
        X_train_copy, y_train_copy, X_train, y_train, pca)


    SVM_300k = SVC(kernel='rbf', C=5, gamma=.05, random_state=42)
    start_time = dt.datetime.now()
    print('Start learning 300k at {}'.format(str(start_time)))
    SVM_300k.fit(X_total, y_total)
    end_time = dt.datetime.now()
    print('Stop learning 300k {}'.format(str(end_time)))
    elapsed_time = end_time - start_time
    print('Elapsed learning 300k {}'.format(str(elapsed_time)))

    # compress=3 压缩率，不压缩的话文件很大，压缩率越高保存读取模型越慢
    joblib.dump(SVM_300k, "n_30_C_28_SVM_300k.pkl", compress=3)#保存模型
    y_pred_train = SVM_300k.predict(X_total)
    y_pred_test = SVM_300k.predict(X_test)
    print(
        metrics.accuracy_score(
            y_total, y_pred_train), metrics.accuracy_score(
            y_test, y_pred_test))


    # y_pred_test = SVM_300k.predict(X_test)
    C_28_SVM_300k = joblib.load('n_30_C_28_SVM_300k.pkl')
    y_pred_test = C_28_SVM_300k.predict(X_test)
    print(metrics.accuracy_score(y_test, y_pred_test))
    # y_pred_total = SVM_300k.predict(X_total)
    # print(metrics.accuracy_score(y_total,y_pred_total))

    '''

    训练模块三：
    旋转数据增强60k，得到数据集共360k作为训练数据集
    具体过程细节与训练模块一类似

    '''

    # 旋转（旋转的角度可以控制，当前函数设置为15度）
    X_total, y_total = add_rotated(
        15, X_train_copy, y_train_copy, X_train, y_train, X_total, y_total, pca)


    SVM_360k = SVC(kernel='rbf', C=2.8, gamma=.05, random_state=42)
    start_time = dt.datetime.now()
    print('Start learning 360k at {}'.format(str(start_time)))
    SVM_360k.fit(X_total, y_total)
    end_time = dt.datetime.now()
    print('Stop learning {}'.format(str(end_time)))
    elapsed_time = end_time - start_time
    print('Elapsed learning {}'.format(str(elapsed_time)))

    joblib.dump(SVM_360k, "n_30_C_28_SVM_360k_ro_15.pkl", compress=3)#保存模型

    # y_pred_train = SVM_360k.predict(X_total)

    y_pred_test = SVM_360k.predict(X_test)
    print(metrics.accuracy_score(y_test, y_pred_test))
    # print(metrics.accuracy_score(y_total,y_pred_train), metrics.accuracy_score(y_test, y_pred_test))


    # Loading model instead
    C_28_SVM_360k = joblib.load('n_30_C_28_SVM_360k_ro_15.pkl')
    y_pred_test = C_28_SVM_360k.predict(X_test)
    print(metrics.accuracy_score(y_test, y_pred_test))

    print("Classification report for 360k classifier %s:\n%s\n" %
          (C_28_SVM_360k, metrics.classification_report(y_test, y_pred_test)))

    cm = metrics.confusion_matrix(y_test, y_pred_test)
    print("Confusion matrix:\n%s" % cm)

    plot_confusion_matrix(cm)

    print("Accuracy={}".format(metrics.accuracy_score(y_test, y_pred_test)))

    '''

    最后的数据统计阶段：

    '''
    # 错误预测数字index
    wrong_index = (y_pred_test - y_test != 0).nonzero()
    wrong_index = wrong_index[0]

    row_sums = cm.sum(axis=1, keepdims=True)
    norm_conf_mx = cm / row_sums
    np.fill_diagonal(norm_conf_mx, 0)  # 对角线填充0，分析错误的部分
    plt.matshow(norm_conf_mx, cmap=plt.cm.Blues)
    print(norm_conf_mx)

    plot_digits(X_test_copy[wrong_index])
