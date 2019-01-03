# -*- coding:utf-8 -*-
import pandas as pd
from sklearn import svm
from sklearn.metrics import roc_curve, auc, f1_score, precision_score, recall_score  ###计算roc和auc
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

data = pd.read_csv('data/drug_cell/drug/Sorafenib/Sorafenib_train_data-rfe.csv')
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

data2 = pd.read_csv('data/drug_cell/drug/Lapatinib/Lapatinib_train_data-rfe.csv')
X2 = data2.iloc[:, :-1]
y2 = data2.iloc[:, -1]

data3 = pd.read_csv('data/drug_cell/drug/PHA-665752/PHA-665752_train_data-rfe.csv')
X3 = data3.iloc[:, :-1]
y3 = data3.iloc[:, -1]

data4 = pd.read_csv('data/drug_cell/drug/Paclitaxel/Paclitaxel_train_data-rfe.csv')
X4 = data4.iloc[:, :-1]
y4 = data4.iloc[:, -1]
# X = X.as_matrix()
# y = y.as_matrix()

# from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split

# x为数据集的feature熟悉，y为label.
# x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
x_train, x_test, y_train, y_test = train_test_split(X, y, train_size=0.6, random_state=2)
x_train2, x_test2, y_train2, y_test2 = train_test_split(X2, y2, train_size=0.6, random_state=2)
x_train3, x_test3, y_train3, y_test3 = train_test_split(X3, y3, train_size=0.6, random_state=2)
x_train4, x_test4, y_train4, y_test4 = train_test_split(X4, y4, train_size=0.6, random_state=2)

print(x_train)
print(y_train)
# print(X, len(X))
# print(y, len(y))
#
# model = svm.SVC(C=502.5649, gamma=0.00019)  # gamma缺省值为 1.0/x.shape[1]
model = svm.SVC(C=373.2254, gamma=0.00032)  # gamma缺省值为 1.0/x.shape[1]
model2 = svm.SVC(C=1000, gamma=0.00094)  # gamma缺省值为 1.0/x.shape[1]
model3 = svm.SVC(C=760.3932, gamma=0.00019)  # gamma缺省值为 1.0/x.shape[1]
model4 = svm.SVC(C=0.1, gamma=1)  # gamma缺省值为 1.0/x.shape[1]
# model = svm.SVC(C=0.1)
model.fit(x_train, y_train)
model2.fit(x_train2, y_train2)
model3.fit(x_train3, y_train3)
model4.fit(x_train4, y_train4)
y_score = model.decision_function(x_test)
y_score2 = model2.decision_function(x_test2)
y_score3 = model3.decision_function(x_test3)
y_score4 = model4.decision_function(x_test4)
y_pre = model.predict(x_test)
print("score: ", model.score(x_test, y_test))
pre_sco_macro = precision_score(y_test, y_pre, average='macro')
pre_sco_wei = precision_score(y_test, y_pre, average='weighted')
pre_sco = precision_score(y_test, y_pre)
f1_sco = f1_score(y_test, y_pre)
rec_sco = recall_score(y_test, y_pre)
accuracy_sco = accuracy_score(y_test, y_pre)
print('recall: ', rec_sco)
print('precision: ', pre_sco)
print('precision-macro: ', pre_sco_macro)
print('precision-wei: ', pre_sco_wei)
print('f1 score: ', f1_sco)
print('accuracy_sco: ', accuracy_sco)
fpr, tpr, threshold = roc_curve(y_test, y_score)  ###计算真正率和假正率
fpr2, tpr2, threshold2 = roc_curve(y_test2, y_score2)  ###计算真正率和假正率
fpr3, tpr3, threshold3 = roc_curve(y_test3, y_score3)  ###计算真正率和假正率
fpr4, tpr4, threshold4 = roc_curve(y_test4, y_score4)  ###计算真正率和假正率
roc_auc = auc(fpr, tpr)  ###计算auc的值
roc_auc2 = auc(fpr2, tpr2)  ###计算auc的值
roc_auc3 = auc(fpr3, tpr3)  ###计算auc的值
roc_auc4 = auc(fpr4, tpr4)  ###计算auc的值
print('auc: ', roc_auc)

# plt.figure()
lw = 3
plt.figure(figsize=(10, 10))
# 假正率为横坐标，真正率为纵坐标做曲线
plt.plot(fpr, tpr, color='red', lw=lw, ls=':', label='Sorafenib (AUC = %0.2f)' % roc_auc, marker='o')
plt.plot(fpr2, tpr2, color='g', lw=lw, ls='-.', label='Lapatinib (AUC = %0.2f)' % roc_auc2, marker='^')
plt.plot(fpr3, tpr3, color='b', lw=lw, ls='--', label='PHA-665752 (AUC = %0.2f)' % roc_auc3, marker='<')
plt.plot(fpr4, tpr4, color='orange', lw=lw, label='Paclitaxel (AUC = %0.2f)' % roc_auc4, marker='*')
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=15)
plt.ylabel('True Positive Rate', fontsize=15)
plt.title('ROC curve', fontsize=20)
plt.legend(loc="lower right", fontsize=20)
# plt.savefig('image/svm_roc1.png')
plt.show()
