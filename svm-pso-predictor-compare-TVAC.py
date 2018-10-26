# coding: utf-8
import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import svm
from sklearn.model_selection import train_test_split
import cmath
from svm_pso_predictor_K import PSO_K
from svm_pso_predictor_TVIW import PSO_W
from svm_pso_predictor_TVAC import PSO_TVAC
from svm_pso_predictor_mTVAC import PSO_MTVAC
from svm_pso_predictor_mhTVAC import PSO_MHTVAC
from svm_pso_predictor_mTVAC_RIW import PSO_MTVACRW
from svm_pso_predictor_hTVAC import PSO_HTVAC
from svm_pso_predictor_original import PSO
from svm_pso_predictor_RANDIW import PSO_RW
import time

drug = 'AEW541'
data = pd.read_csv('data/drug_cell/drug/%s_train_data-rfe-sa-50.csv' % drug)
X = data.iloc[:, :-1]
y = data.iloc[:, -1]
# random_state=1不变的话，每次得到的数据都是一样的，random_state=None，每次的数据不一样
train_size = 0.6
# x_train, x_test, y_train, y_test = train_test_split(X, y, random_state=1, train_size=train_size)

MAX_ITER = 1000
feature_number = X.shape[1]

for run in range(5):
    # ----------------------程序执行-----------------------
    start = time.time()

    # pso_K = PSO_K(max_iter=MAX_ITER, x_train=x_train, y_train=y_train, x_test=x_test,
    #               y_test=y_test)  # 维度代表变量的个数
    # pso_K.init_Population()
    # fitness_K = pso_K.iterator()

    # pso_W = PSO_W(max_iter=MAX_ITER, x_train=x_train, y_train=y_train, x_test=x_test,
    #               y_test=y_test)  # 维度代表变量的个数
    # pso_W.init_Population()
    # fitness_W = pso_W.iterator()

    pso_TVAC = PSO_TVAC(max_iter=MAX_ITER, data_X=X, data_y=y)  # 维度代表变量的个数
    pso_TVAC.init_Population()
    fitness_TVAC = pso_TVAC.iterator()

    pso_mTVAC = PSO_MTVAC(max_iter=MAX_ITER, data_X=X, data_y=y)  # 维度代表变量的个数
    pso_mTVAC.init_Population()
    fitness_mTVAC = pso_mTVAC.iterator()

    # 去掉了线性递减的惯性权重，改为0.5+rand(0,1)/2
    # pso_mTVACRW = PSO_MTVACRW(max_iter=MAX_ITER, x_train=x_train, y_train=y_train, x_test=x_test,
    #                       y_test=y_test)  # 维度代表变量的个数
    # pso_mTVACRW.init_Population()
    # fitness_mTVACRW = pso_mTVACRW.iterator()

    pso_hTVAC = PSO_HTVAC(max_iter=MAX_ITER, data_X=X, data_y=y)  # 维度代表变量的个数
    pso_hTVAC.init_Population()
    fitness_hTVAC = pso_hTVAC.iterator()

    pso_mhTVAC = PSO_MHTVAC(max_iter=MAX_ITER, data_X=X, data_y=y)  # 维度代表变量的个数
    pso_mhTVAC.init_Population()
    fitness_mhTVAC = pso_mhTVAC.iterator()

    pso_original = PSO(max_iter=MAX_ITER, data_X=X, data_y=y)  # 维度代表变量的个数
    pso_original.init_Population()
    fitness_original = pso_original.iterator()

    # pso_RW = PSO_RW(max_iter=MAX_ITER, x_train=x_train, y_train=y_train, x_test=x_test,
    #                    y_test=y_test)  # 维度代表变量的个数
    # pso_RW.init_Population()
    # fitness_RW = pso_RW.iterator()

    end = time.time()

    best = pso_TVAC  # 找到准确率最高的---越小越高
    if pso_mTVAC.fit < best.fit:
        best = pso_mTVAC
    if pso_hTVAC.fit < best.fit:
        best = pso_hTVAC
    if pso_mhTVAC.fit < best.fit:
        best = pso_mhTVAC
    if pso_original.fit < best.fit:
        best = pso_original
    # -------------------画图--------------------
    plt.figure(figsize=(10, 8))
    plt.title('drug:%s\n feature number=%d,train_size:%.1f,cost: %d seconds\n best:%.4f  C=%.4f,gamma=%.5f' % (
        drug, feature_number, train_size, end - start, -best.fit, best.gbest[0], best.gbest[1]), size=16)
    plt.xlabel("number of generations", size=16)
    plt.ylabel("SVM model accuracy rate", size=16)
    t = np.array([t for t in range(0, MAX_ITER)])
    # fitness_K = np.array(fitness_K)
    # fitness_K2 = [-v for v in fitness_K]  # 取反，得到正数，模型准确率

    # fitness_W2 = [-v for v in fitness_W]  # 取反，得到正数，模型准确率

    fitness_TVAC2 = [-v for v in fitness_TVAC]  # 取反，得到正数，模型准确率

    fitness_mTVAC2 = [-v for v in fitness_mTVAC]  # 取反，得到正数，模型准确率

    # fitness_mTVACRW2 = [-v for v in fitness_mTVACRW]  # 取反，得到正数，模型准确率

    fitness_hTVAC2 = [-v for v in fitness_hTVAC]  # 取反，得到正数，模型准确率

    fitness_mhTVAC2 = [-v for v in fitness_mhTVAC]  # 取反，得到正数，模型准确率

    pso_original2 = [-v for v in fitness_original]  # 取反，得到正数，模型准确率

    # fitness_RW2 = [-v for v in fitness_RW]  # 取反，得到正数，模型准确率

    # plt.plot(t, fitness_K2, color='b', linewidth=3, label='PSO-K')
    plt.plot(t, pso_original2, color='b', linewidth=3.2, label='PSO', ls='-.')
    # plt.plot(t, fitness_W2, color='r', linewidth=3, label='PSO-TVIW', ls=':')
    plt.plot(t, fitness_TVAC2, color='g', linewidth=3.1, label='PSO-TVAC', ls='--')
    # plt.plot(t, fitness_RW2, color='lightblue', linewidth=3, label='PSO-RANDIW', ls='-')
    plt.plot(t, fitness_mTVAC2, color='orange', linewidth=3, label='PSO-mTVAC', ls=':')
    plt.plot(t, fitness_mhTVAC2, color='red', linewidth=2.9, label='PSO-mhTVAC')
    # plt.plot(t, fitness_mTVACRW2, color='black', linewidth=3, label='PSO-mTVACRW')
    plt.plot(t, fitness_hTVAC2, color='yellow', linewidth=2.8, label='PSO-hTVAC')
    plt.legend(loc="lower right", fontsize=16)
    plt.yticks(fontsize='14')
    plt.xticks(fontsize='14')
    # plt.rcParams['savefig.dpi'] = 300  # 图片像素
    # plt.rcParams['figure.dpi'] = 300  # 分辨率
    plt.savefig('image/pso_TVAC/%s/sa-%d-pso-compare-%s-%d.png' % (drug, feature_number, drug, 1 + run))
    # plt.show()
    plt.cla()
