from sklearn.svm import SVC
from sklearn.feature_selection import RFE
import pandas as pd
import time

# data = pd.read_csv('data/drug_cell/drug/17-AAG_train_data.csv')
# data = pd.read_csv('data/drug_cell/drug/Erlotinib_train_data.csv')
# data = pd.read_csv('data/drug_cell/drug/Irinotecan_train_data.csv') # feature number选择9
# data = pd.read_csv('data/drug_cell/drug/AZD6244_train_data.csv') # feature number选择10,    0.893939393939
# data = pd.read_csv('data/drug_cell/drug/Lapatinib_train_data.csv') # feature number选择7
# data = pd.read_csv('data/drug_cell/drug/PD-0325901_train_data.csv') # feature number选择7
# data = pd.read_csv('data/drug_cell/drug/AEW541_train_data.csv') # feature number选择7  最高就0.666666666667
# data = pd.read_csv('data/drug_cell/drug/PHA-665752_train_data.csv') # feature number选择10  最高就0.666666666667
# data = pd.read_csv('data/drug_cell/drug/Paclitaxel_train_data.csv')  # feature number选择11  最高就0.666666666667
# data = pd.read_csv('data/drug_cell/drug/PLX4720_train_data.csv')  # feature number选择7  没有负样本，不能运行
# data = pd.read_csv('data/drug_cell/drug/AZD0530_train_data.csv')  # feature number选择11
# data = pd.read_csv('data/drug_cell/drug/LBW242_train_data.csv')  # feature number选择6
data = pd.read_csv('data/drug_cell/drug/Nutlin-3_train_data.csv')  # feature number选择9
data = data.fillna(0)
x = data.iloc[:, :-1]
y = data.iloc[:, -1]
svc = SVC(kernel="linear", C=1)
rfe = RFE(estimator=svc, n_features_to_select=9, step=1)
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
x_new.to_csv('data/drug_cell/drug/Nutlin-3_train_data-rfe.csv', index=False, float_format='%.2f')
