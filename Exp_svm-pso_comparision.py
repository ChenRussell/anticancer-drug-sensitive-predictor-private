import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('data/svm-pso_accuracy.csv')
data.set_index('drug', inplace=True, drop=True)
print(data)
data.plot(kind='bar')
# data.boxplot()
# 设置坐标轴名称
plt.xlabel('')
plt.ylabel('Model Accuracy', fontsize=20)
plt.xticks(rotation=0, fontsize=20)
plt.title('drugs both in CCLE&CGP dataset', fontsize=20)
# plt.savefig('image/svm_experiment.png')
plt.show()