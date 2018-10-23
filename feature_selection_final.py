from sklearn.svm import SVC
from sklearn.feature_selection import RFE
import pandas as pd
import time

# data = pd.read_csv('data/drug_cell/drug/17-AAG_train_data.csv')   # fn=10 0.8529
# data = pd.read_csv('data/drug_cell/drug/Erlotinib_train_data.csv') # fn=10 0.9245
# data = pd.read_csv('data/drug_cell/drug/Irinotecan_train_data.csv') # fn=9, 0.8404
# data = pd.read_csv('data/drug_cell/drug/AZD6244_train_data.csv') # fn=12, 0.9394
# data = pd.read_csv('data/drug_cell/drug/Lapatinib_train_data.csv') # fn=7 0.8704
# data = pd.read_csv('data/drug_cell/drug/PD-0325901_train_data.csv') # fn=7 0.9412
# data = pd.read_csv('data/drug_cell/drug/Sorafenib_train_data.csv')  # fn=9  0.8679
# data = pd.read_csv('data/drug_cell/drug/AEW541_train_data.csv') # fn=7  0.8095
# data = pd.read_csv('data/drug_cell/drug/PHA-665752_train_data.csv') # fn=10  0.8214,fn=3时，只有0.69
# data = pd.read_csv('data/drug_cell/drug/Paclitaxel_train_data.csv')  # fn=11  0.8857
# data = pd.read_csv('data/drug_cell/drug/PLX4720_train_data.csv')  # fn=7  没有负样本，不能运行
# data = pd.read_csv('data/drug_cell/drug/AZD0530_train_data.csv')  # fn=11 0.8308
# data = pd.read_csv('data/drug_cell/drug/LBW242_train_data.csv')  # fn=6 0.8000
# data = pd.read_csv('data/drug_cell/drug/Nutlin-3_train_data.csv')  # fn=9
# data = pd.read_csv('data/drug_cell/drug/Panobinostat_train_data.csv')  # fn=14 0.9825, 50%测试集 0.9789
# data = pd.read_csv('data/drug_cell/drug/PD-0332991_train_data.csv')  # fn=8
data = pd.read_csv('data/drug_cell/drug/PF2341066_train_data.csv')  # fn=10

data = data.fillna(0)
x = data.iloc[:, :-1]
y = data.iloc[:, -1]
svc = SVC(kernel="linear", C=1)
rfe = RFE(estimator=svc, n_features_to_select=10, step=1)
start = time.time()
print('算法运行开始......')
rfe.fit(x, y)  # 训练时间特别长！！！
print('算法运行结束......')
end = time.time()
print("特征选择运行时间：", end - start)
# x_new = x[x.columns[rfe.get_support()]]
x_new = x.loc[:, rfe.get_support()]
x_new['label'] = y
print(x_new)
x_new.to_csv('data/drug_cell/drug/PF2341066_train_data-rfe.csv', index=False, float_format='%.2f')
