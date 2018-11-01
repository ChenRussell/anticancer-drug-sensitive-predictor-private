# -*- coding:utf-8 -*-
import pandas as pd
from sklearn import svm
from sklearn.metrics import roc_curve, auc, f1_score, precision_score, recall_score ###计算roc和auc
import matplotlib.pyplot as plt

data = pd.read_csv('data/drug_cell/drug/Panobinostat/Panobinostat_train_data-rfe-sa-50.csv')
X = data.iloc[:, :-1]
y = data.iloc[:, -1]
# X = X.as_matrix()
# y = y.as_matrix()

# from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split

# x为数据集的feature熟悉，y为label.
# x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
x_train, x_test, y_train, y_test = train_test_split(X, y, train_size=0.6)

print(x_train)
print(y_train)
# print(X, len(X))
# print(y, len(y))
#
# model = svm.SVC(C=502.5649, gamma=0.00019)  # gamma缺省值为 1.0/x.shape[1]
model = svm.SVC(C=1000, gamma=0.00001)  # gamma缺省值为 1.0/x.shape[1]
# model = svm.SVC(C=0.1)
model.fit(x_train, y_train)
y_score = model.decision_function(x_test)
y_pre = model.predict(x_test)
print("score: ", model.score(x_test, y_test))
pre_sco_macro = precision_score(y_pre, y_test, average='macro')
pre_sco_wei = precision_score(y_pre, y_test, average='weighted')
pre_sco = precision_score(y_pre, y_test)
f1_sco = f1_score(y_pre, y_test)
rec_sco = recall_score(y_pre, y_test)
print('recall: ', rec_sco)
print('precision: ', pre_sco)
print('precision-macro: ', pre_sco_macro)
print('precision-wei: ', pre_sco_wei)
print('f1 score: ', f1_sco)
fpr, tpr, threshold = roc_curve(y_test, y_score)  ###计算真正率和假正率
roc_auc = auc(fpr, tpr)  ###计算auc的值
print('auc: ', roc_auc)

# plt.figure()
lw = 2
plt.figure(figsize=(10, 10))
plt.plot(fpr, tpr, color='red',
         lw=lw, label='ROC curve (AUC = %0.2f)' % roc_auc, marker='o')  ###假正率为横坐标，真正率为纵坐标做曲线
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=15)
plt.ylabel('True Positive Rate', fontsize=15)
plt.title('AZD6244', fontsize=20)
plt.legend(loc="lower right", fontsize=20)
# plt.savefig('image/svm_roc1.png')
# plt.show()
