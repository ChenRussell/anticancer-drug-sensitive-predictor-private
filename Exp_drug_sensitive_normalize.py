import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random

data = pd.read_csv('data/drug_sensitive_normalize.csv')
# data.set_index('drug', inplace=True, drop=True)
print(data)
# bar = data.plot(kind='bar', width=0.7, bottom=0)
color = ['white', 'black']
rands = [0.05, 0.02, 0.07]
height = 2.2
for i in range(100):
    height -= rands[random.randint(0, 2)]
    if 0.8 <= height:
        color[0] = 'white'
        color[1] = 'black'
    if -0.8 < height < 0.8:
        color[0] = 'red'
        color[1] = 'red'
    if height < -0.8:
        color[0] = 'black'
        color[1] = 'black'
    plt.bar(i, height, width=0.7, facecolor=color[0], edgecolor=color[1])
# 设置坐标轴名称
# plt.xlabel('')
plt.ylabel('IC50 value', fontsize='20')
# plt.xticks(rotation=45, fontsize=12)
# plt.savefig('image/feature_select_experiment.png')


frame = plt.gca()
# y 轴不可见
# frame.axes.get_yaxis().set_visible(False)
# x 轴不可见
frame.axes.get_xaxis().set_visible(False)
plt.yticks(np.arange(-3.2, 3.2, 0.8), fontsize='16')
plt.show()
