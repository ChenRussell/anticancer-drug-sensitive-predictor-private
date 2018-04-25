from sklearn.svm import SVC
from sklearn.feature_selection import RFE
import pandas as pd
import getopt
import sys

data = pd.read_csv('data/drug_cell/drug/17-AAG_train_data.csv')
data = data.fillna(0)
x = data.iloc[:, :-1]
y = data.iloc[:, -1]
svc = SVC(kernel="linear", C=1)
rfe = RFE(estimator=svc, n_features_to_select=10, step=1)
rfe.fit(x, y)
x_new = x[x.columns[rfe.get_support()]]
x_new['label'] = y
print(x_new)
x_new.to_csv('data/drug_cell/drug/17-AAG_train_data-rfe.csv', index=False, float_format='%.2f')