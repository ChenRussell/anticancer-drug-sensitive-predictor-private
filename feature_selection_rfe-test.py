from sklearn.svm import SVC
from SA_RFE import SA_RFE
from SA_RFE_mRMR import SA_RFE_mRMR
from SA_mRMR import SA_mRMR
import pandas as pd
import time

drug = 'PD-0332991'
feature_number = 50
data = pd.read_csv('data/drug_cell/drug/%s/%s_train_data.csv' % (drug, drug))

data = data.fillna(0)
x = data.iloc[:, :-1]
y = data.iloc[:, -1]
svc = SVC(kernel="linear", C=1)
rfe = SA_RFE(estimator=svc, n_features_to_select=feature_number, step=1)
# rfe = SA_mRMR(estimator=svc, n_features_to_select=4, step=1)
# rfe = SA_RFE_mRMR(estimator=svc, n_features_to_select=feature_number, step=1)
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
x_new.to_csv('data/drug_cell/drug/%s/%s_train_data-rfe-sa-%d.csv' % (drug, drug, feature_number), index=False,
             float_format='%.2f')
# x_new.to_csv('data/drug_cell/drug/%s/%s_train_data-rfe-mrmr-sa-%d.csv' % (drug, drug, feature_number), index=False,
#              float_format='%.2f')
# x_new.to_csv('data/drug_cell/drug/17-AAG_train_data-rfe-mrmr-sa-50.csv', index=False, float_format='%.2f')
