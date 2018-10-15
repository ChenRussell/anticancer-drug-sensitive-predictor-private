import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


data = pd.read_csv('data/drug_sensitive_normalize.csv')
# data.set_index('drug', inplace=True, drop=True)
print(data)
data.plot(kind='bar', width=0.9, bottom=0)
# 设置坐标轴名称
# plt.xlabel('')
plt.ylabel('model accuracy')
# plt.xticks(rotation=45, fontsize=12)
# plt.savefig('image/feature_select_experiment.png')
plt.show()