from sklearn.svm import SVC
from rfe_v2 import RFE_V2
import pandas as pd
import time

data = pd.read_csv('data/drug_cell/drug/AEW541_train_data.csv')  # fn=7  0.8095

data = data.fillna(0)
x = data.iloc[:, :-1]
y = data.iloc[:, -1]
svc = SVC(kernel="linear", C=1)
# rfe = RFE(estimator=svc, n_features_to_select=8, step=1)
rfe = RFE_V2(estimator=svc, n_features_to_select=50, step=1)
start = time.time()
print('算法运行开始......')
rfe.fit(x, y)  # 训练时间特别长！！！
print('算法运行结束......')
end = time.time()
print("特征选择运行时间：", end - start)
# x_new = x[x.columns[rfe.get_support()]]
x_new = x.loc[:, rfe.get_support()]
x_new['label'] = y
# print(x_new)
x_new.to_csv('data/drug_cell/drug/AEW541_train_data-rfe-sa-50.csv', index=False, float_format='%.2f')
