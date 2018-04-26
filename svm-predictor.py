# -*- coding:utf-8 -*-
import pandas as pd
from sklearn import svm


data = pd.read_csv('data/drug_cell/drug/Erlotinib_train_data-rfe.csv')
X = data.iloc[:, :-1]
y = data.iloc[:, -1]
# X = X.as_matrix()
# y = y.as_matrix()

# from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split
#x为数据集的feature熟悉，y为label.
# x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
x_train, x_test, y_train, y_test = train_test_split(X, y, random_state=1, train_size=0.7)

print(x_train)
print(y_train)
# print(X, len(X))
# print(y, len(y))
#
model = svm.SVC()
model.fit(x_train, y_train)
print(model.score(x_test, y_test))