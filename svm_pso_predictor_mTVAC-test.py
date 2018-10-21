# coding: utf-8
import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import svm
from sklearn.model_selection import train_test_split
import cmath

# data = pd.read_csv('data/drug_cell/drug/AEW541_train_data-rfe.csv')
# X = data.iloc[:, :-1]
# y = data.iloc[:, -1]
# x_train, x_test, y_train, y_test = train_test_split(X, y, random_state=1, train_size=0.7)

MAX_ITER = 500

data = pd.read_csv('data/drug_cell/drug/Lapatinib_train_data-rfe.csv')
data_test = pd.read_csv('data/CGP/drug_cell/common_drugs/Lapatinib_train_data.csv')
x_train = data.iloc[:, :-1]
y_train = data.iloc[:, -1]

x_test = data_test.iloc[:, :-1]
y_test = data_test.iloc[:, -1]


# ----------------------PSO参数设置---------------------------------
class PSO_MTVAC():
    def __init__(self, max_iter, pN=30, dim=2):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

        self.w = 0.9  # 惯性权重
        self.wS = 0.9
        self.wE = 0.4
        self.c1 = 2
        self.c2 = 2
        self.c1f = 2.5
        self.c1i = 0.5
        self.c2f = 0.5
        self.c2i = 2.5
        self.r1 = random.uniform(0, 1)
        self.r2 = random.uniform(0, 1)
        self.r3 = random.uniform(0, 1)
        self.r4 = random.uniform(0, 1)
        self.mprop = random.uniform(0, 1)  # 突变概率
        self.rp = random.randint(0, 29)  # 随机选择一个微粒(index)
        self.rd = random.randint(0, 1)  # 随机选择一个维度(index)
        self.m = 2  # 常量, 怎么取值？？？

        self.pN = pN  # 粒子数量
        self.dim = dim  # 搜索维度

        self.maxC = 100
        self.minC = 0.00001
        self.maxGamma = 5
        self.minGamma = 0.00001
        self.max_v = np.array([self.maxC, self.maxGamma])  # 最大速度
        self.min_v = np.array([-self.maxC, -self.maxGamma])  # 最小速度
        self.max_x = np.array([self.maxC, self.maxGamma])  # 粒子位置的上界
        self.min_x = np.array([self.minC, self.minGamma])  # 粒子位置的下界

        self.max_iter = max_iter  # 迭代次数
        self.X = np.zeros((self.pN, self.dim))  # 所有粒子的位置和速度
        self.V = np.zeros((self.pN, self.dim))
        self.pbest = np.zeros((self.pN, self.dim))  # 个体经历的最佳位置和全局最佳位置
        self.gbest = np.zeros((1, self.dim))
        self.p_fit = np.zeros(self.pN)  # 每个个体的历史最佳适应值
        self.fit = 1e10  # 全局最佳适应值

    # ---------------------目标函数Sphere函数-----------------------------

    def function(self, c, g):
        if g <= 0 or c <= 0:
            return 1e10
        model = svm.SVC(C=c, gamma=g)  # gamma缺省值为 1.0/x.shape[1]
        model.fit(self.x_train, self.y_train)
        y_score = model.score(self.x_test, self.y_test)
        return -y_score

    # ---------------------初始化种群----------------------------------
    def init_Population(self):
        for i in range(self.pN):
            for j in range(self.dim):
                self.X[i][j] = random.uniform(self.min_x[j], self.max_x[j])  # 位置的初始范围
                self.V[i][j] = random.uniform(self.min_v[j], self.max_v[j])  # 速度的初始范围
            self.pbest[i] = self.X[i]
            tmp = self.function(self.X[i][0], self.X[i][1])
            self.p_fit[i] = tmp
            if tmp < self.fit:
                self.fit = tmp
                self.gbest = self.X[i]

                # ----------------------更新粒子位置----------------------------------

    def iterator(self):
        fitness = []
        for iter in range(self.max_iter):
            for i in range(self.pN):  # 更新gbest\pbest
                temp = self.function(self.X[i][0], self.X[i][1])
                if temp < self.p_fit[i]:  # 更新个体最优
                    self.p_fit[i] = temp
                    self.pbest[i] = self.X[i]
                    if self.p_fit[i] < self.fit:  # 更新全局最优
                        self.gbest = self.X[i]
                        self.fit = self.p_fit[i]

            # 每次迭代都更新随机系数
            self.r1 = random.uniform(0, 1)
            self.r2 = random.uniform(0, 1)
            self.r3 = random.uniform(0, 1)
            self.r4 = random.uniform(0, 1)
            self.rp = random.randint(0, self.pN - 1)  # 随机选择一个微粒(index)
            self.rd = random.randint(0, self.dim - 1)  # 随机选择一个维度(index)

            for i in range(self.pN):
                for d in range(self.dim):  # 对维度遍历
                    self.V[i][d] = self.w * self.V[i][d] + self.c1 * self.r1 * (self.pbest[i][d] - self.X[i][d]) + \
                                   self.c2 * self.r2 * (self.gbest[d] - self.X[i][d])

                    # 限制速度的简化代码
                    self.V[i][d] = np.sign(self.V[i][d]) * min(abs(self.V[i][d]), self.max_v[d])

                    self.X[i][d] = self.X[i][d] + self.V[i][d]  # 更新粒子位置

                    # 限制粒子位置边界
                    if self.X[i][d] > self.max_x[d]:
                        self.X[i][d] = self.max_x[d]
                    elif self.X[i][d] < self.min_x[d]:
                        self.X[i][d] = self.min_x[d]

            fitness.append(self.fit)

            if len(fitness) >= 2 and (fitness[-1] - fitness[-2] <= 0):
                if self.r1 < self.mprop:
                    if self.r2 < 0.5:
                        self.V[self.rp][self.rd] += self.r3 * self.max_v[self.rd] / self.m
                    else:
                        self.V[self.rp][self.rd] -= self.r4 * self.max_v[self.rd] / self.m

            print('V: ', self.V[0], end=" ")
            print('X: ', self.X[0], end=" ")
            print(self.fit, end=" ")  # 输出最优值
            print('PSO-mTVAC 当前迭代次数：', iter)

            # 更新学习因子
            self.c1 = (self.c1f - self.c1i) * iter / self.max_iter + self.c1i
            self.c2 = (self.c2f - self.c2i) * iter / self.max_iter + self.c2i

            # 更新惯性权重
            self.w = self.wS - (self.wS - self.wE) * iter / self.max_iter
        return fitness

        # ----------------------程序执行-----------------------


my_pso = PSO_MTVAC(pN=30, dim=2, max_iter=MAX_ITER)  # 维度代表变量的个数
my_pso.init_Population()
fitness = my_pso.iterator()
# -------------------画图--------------------
plt.figure(1)
plt.title("Figure1")
plt.xlabel("iterators", size=14)
plt.ylabel("fitness", size=14)
t = np.array([t for t in range(0, MAX_ITER)])
fitness = np.array(fitness)
fitness_2 = [-v for v in fitness]  # 取反，得到正数，模型准确率
plt.plot(t, fitness_2, color='b', linewidth=3)
plt.show()
