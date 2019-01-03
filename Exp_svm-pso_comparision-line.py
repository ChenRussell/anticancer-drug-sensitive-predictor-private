import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('data/svm-pso_accuracy-line.csv')
data.set_index('drug', inplace=True, drop=True)
print(data)
# data.plot(kind='line', linewidth=3)
x = [i for i in range(1, 22)]
plt.plot(x, data['SVM'], color='b', linewidth=3, label='SVM-CV', ls='-.', marker='o', ms=10)
plt.plot(x, data['SVM-PSO'], color='r', linewidth=3, label='SVM-PSO', ls=':', marker='^', ms=10)
plt.plot(x, data['SVM-PSO-mTVAC'], color='orange', linewidth=3, label='SVM-PSO-mhTVAC', ls='-', marker='*', ms=10)
# x = ['17-AAG', 'AEW541', 'AZD0530', 'AZD6244', 'Erlotinib', 'Irinotecan', 'L-685458',
#      'Lapatinib', 'LBW242', 'Nilotinib', 'Nutlin-3', 'Paclitaxel', 'Panobinostat', 'PD-0325901',
#      'PD-0332991', 'PF2341066', 'PHA-665752', 'PLX4720', 'RAF265', 'Sorafenib', 'TAE684', 'TKI258',
#      'Topotecan', 'ZD-6474']

plt.xticks(x, list(data.index), fontsize=12)
# plt.ylim((0.8, 0.95))
plt.yticks(np.arange(0.5, 1, 0.05), fontsize=12)
# 设置坐标轴名称
plt.xlabel('')
plt.ylabel('Model Accuracy', fontsize=20)
plt.xticks(rotation=45, fontsize=12)
# plt.title('SVM-CV vs SVM-PSO', fontsize=20)
plt.legend(loc="lower right", fontsize=16)
plt.grid(True)
# plt.savefig('image/svm_experiment.png')
plt.show()
