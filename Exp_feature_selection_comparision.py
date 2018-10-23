import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# df = pd.DataFrame(np.random.rand(6,4),
#                   index=['one', 'two', 'three', 'four', 'five', 'six'],
#                   columns=pd.Index(['A', 'B', 'C', 'D'], name='Genus'))
# df.plot(kind='bar')

indexs = ['17-AAG', 'AEW541', 'AZD0530', 'AZD6244', 'Erlotinib', 'Irinotecan',
          'Lapatinib', 'LBW242', 'Nutlin-3', 'Paclitaxel', 'Panobinostat', 'PD-0325901',
          'PD-0332991', 'PF2341066', 'PHA-665752', 'RAF265', 'Sorafenib', 'TAE684', 'TKI258',
          'Topotecan', 'ZD-6474']
data = pd.read_csv('data/feature_select_accuracy.csv')
data.set_index('drug', inplace=True, drop=True)
print(data)
data.plot(kind='bar', width=0.7)
# 设置坐标轴名称
plt.xlabel('')
plt.ylabel('SVM Model Accuracy', fontsize=20)
plt.xticks(rotation=45, fontsize=12)
plt.yticks(fontsize=12)
plt.legend(fontsize=16)
plt.grid(True)
# plt.savefig('image/feature_select_experiment.png')
plt.show()