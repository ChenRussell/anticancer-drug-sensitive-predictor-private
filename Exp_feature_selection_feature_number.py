import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# df = pd.DataFrame(np.random.rand(6,4),
#                   index=['one', 'two', 'three', 'four', 'five', 'six'],
#                   columns=pd.Index(['A', 'B', 'C', 'D'], name='Genus'))
# df.plot(kind='bar')

indexs = ['17-AAG', 'AEW541', 'AZD0530', 'AZD6244', 'Erlotinib', 'Irinotecan', 'L-685458',
          'Lapatinib', 'LBW242', 'Nilotinib', 'Nutlin-3', 'Paclitaxel', 'Panobinostat', 'PD-0325901',
          'PD-0332991', 'PF2341066', 'PHA-665752', 'PLX4720', 'RAF265', 'Sorafenib', 'TAE684', 'TKI258',
          'Topotecan', 'ZD-6474']
data = pd.read_csv('data/feature_select_numbers.csv')
data.set_index('feature_numbers', inplace=True, drop=True)
print(data)
# data.plot(kind='line')
x = data.index
y1 = data['17-AAG']
y2 = data['AEW541']
y3 = data['AZD0530']
y4 = data['AZD6244']
plt.subplot(411)
plt.plot(x, y1, color='b', linewidth=3, label='17-AAG' , ls='-.', marker='+', ms=10)
plt.legend(fontsize=16)
plt.yticks(fontsize=14)
plt.xticks(fontsize=14)
plt.grid(True)

plt.subplot(412)
plt.plot(x, y2, color='r', linewidth=3, label='AEW541', ls=':', marker='^', ms=10)
plt.legend(fontsize=16)
plt.yticks(fontsize=14)
plt.xticks(fontsize=14)
plt.grid(True)
plt.ylabel('Accuracy    Rate', fontsize=16)

plt.subplot(413)
plt.plot(x, y3, color='g', linewidth=3, label='AZD0530', ls='-', marker='*', ms=10)
plt.legend(fontsize=16)
plt.yticks(fontsize=14)
plt.xticks(fontsize=14)
plt.grid(True)
plt.ylabel('SVM     Model', fontsize=16)

plt.subplot(414)
plt.plot(x, y4, color='orange', linewidth=3, label='AZD6244', ls='--', marker='o', ms=10)

# plt.subplot(221)
plt.legend(fontsize=16)
plt.yticks(fontsize=14)
plt.xticks(fontsize=14)
# 设置坐标轴名称
plt.xlabel('feature numbers', fontsize=18)

plt.grid(True)
# plt.xticks(rotation=45)
# plt.savefig('image/feature_select_experiment.png')
plt.show()
