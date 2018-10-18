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
from svm_pso_predictor_hTVAC import PSO_HTVAC
from svm_pso_predictor_original import PSO
from svm_pso_predictor_RANDIW import PSO_RW
import time

# import svm-pso-predictor-K

data = pd.read_csv('data/drug_cell/drug/AEW541_train_data-rfe.csv')
X = data.iloc[:, :-1]
y = data.iloc[:, -1]
x_train, x_test, y_train, y_test = train_test_split(X, y, random_state=1, train_size=0.7)

MAX_ITER = 800

for run in range(5):
    # ----------------------程序执行-----------------------
    start = time.time()

    # pso_K = PSO_K(max_iter=MAX_ITER, x_train=x_train, y_train=y_train, x_test=x_test,
    #               y_test=y_test)  # 维度代表变量的个数
    # pso_K.init_Population()
    # fitness_K = pso_K.iterator()

    pso_W = PSO_W(max_iter=MAX_ITER, x_train=x_train, y_train=y_train, x_test=x_test,
                  y_test=y_test)  # 维度代表变量的个数
    pso_W.init_Population()
    fitness_W = pso_W.iterator()

    pso_TVAC = PSO_TVAC(max_iter=MAX_ITER, x_train=x_train, y_train=y_train, x_test=x_test,
                        y_test=y_test)  # 维度代表变量的个数
    pso_TVAC.init_Population()
    fitness_TVAC = pso_TVAC.iterator()

    pso_mTVAC = PSO_MTVAC(max_iter=MAX_ITER, x_train=x_train, y_train=y_train, x_test=x_test,
                        y_test=y_test)  # 维度代表变量的个数
    pso_mTVAC.init_Population()
    fitness_mTVAC = pso_mTVAC.iterator()

    # pso_hTVAC = PSO_HTVAC(max_iter=MAX_ITER, x_train=x_train, y_train=y_train, x_test=x_test,
    #                     y_test=y_test)  # 维度代表变量的个数
    # pso_hTVAC.init_Population()
    # fitness_hTVAC = pso_hTVAC.iterator()

    pso_original = PSO(max_iter=MAX_ITER, x_train=x_train, y_train=y_train, x_test=x_test,
                       y_test=y_test)  # 维度代表变量的个数
    pso_original.init_Population()
    fitness_original = pso_original.iterator()

    pso_RW = PSO_RW(max_iter=MAX_ITER, x_train=x_train, y_train=y_train, x_test=x_test,
                       y_test=y_test)  # 维度代表变量的个数
    pso_RW.init_Population()
    fitness_RW = pso_RW.iterator()

    end = time.time()
    # -------------------画图--------------------
    plt.figure(1)
    plt.title('PSO comparison, cost: %.2f seconds' % (end-start), size=16)
    plt.xlabel("iterators", size=16)
    plt.ylabel("svm model accuracy rate", size=16)
    t = np.array([t for t in range(0, MAX_ITER)])
    # fitness_K = np.array(fitness_K)
    # fitness_K2 = [-v for v in fitness_K]  # 取反，得到正数，模型准确率

    fitness_W2 = [-v for v in fitness_W]  # 取反，得到正数，模型准确率

    fitness_TVAC2 = [-v for v in fitness_TVAC]  # 取反，得到正数，模型准确率

    fitness_mTVAC2 = [-v for v in fitness_mTVAC]  # 取反，得到正数，模型准确率

    # fitness_hTVAC2 = [-v for v in fitness_hTVAC]  # 取反，得到正数，模型准确率

    pso_original2 = [-v for v in fitness_original]  # 取反，得到正数，模型准确率

    fitness_RW2 = [-v for v in fitness_RW]  # 取反，得到正数，模型准确率

    # plt.plot(t, fitness_K2, color='b', linewidth=3, label='PSO-K')
    plt.plot(t, fitness_W2, color='r', linewidth=3, label='PSO-TVIW', ls=':')
    plt.plot(t, fitness_TVAC2, color='g', linewidth=3, label='PSO-TVAC', ls='--')
    plt.plot(t, pso_original2, color='pink', linewidth=3, label='PSO')
    plt.plot(t, fitness_RW2, color='lightblue', linewidth=3, label='PSO-RANDIW')
    plt.plot(t, fitness_mTVAC2, color='orange', linewidth=3, label='PSO-mTVAC')
    # plt.plot(t, fitness_hTVAC2, color='yellow', linewidth=3, label='PSO-hTVAC')
    plt.legend(loc="lower right", fontsize=16)
    plt.savefig('image/pso-compare-lb%d.png' % (24+run))
    plt.show()
    plt.cla()
