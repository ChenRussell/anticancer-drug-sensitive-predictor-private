# -*- coding:utf-8 -*-
import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.metrics import roc_curve, auc, f1_score, precision_score, recall_score, accuracy_score  ###计算roc和auc
import matplotlib.pyplot as plt

# 0样本很多，1样本很少，得到的准确率都很高
data = pd.read_csv('data/drug_cell/drug/17-AAG/17-AAG_train_data-rfe-sa-40-copy.csv')  # 0.9550,gBest: 64.59752,0.37184
data_test = pd.read_csv('data/drug_cell/drug/17-AAG/17-AAG_train_data-rfe-sa-40-test.csv')

# data = pd.read_csv('data/drug_cell/drug/Sorafenib/Sorafenib_train_data-rfe.csv')  # 0.9550,gBest: 64.59752,0.37184
# data_test = pd.read_csv('data/CGP/drug_cell/common_drugs/Sorafenib_train_data.csv')

# data = pd.read_csv('data/drug_cell/drug/Lapatinib/Lapatinib_train_data-rfe.csv')  # 0.9550	gBest: 67.03676,3.33841
# data_test = pd.read_csv('data/CGP/drug_cell/common_drugs/Lapatinib_train_data.csv')

# data = pd.read_csv('data/drug_cell/drug/PHA-665752/PHA-665752_train_data-rfe.csv')  # 0.8423	gBest: 1000.00000,0.00260
# data_test = pd.read_csv('data/CGP/drug_cell/common_drugs/PHA-665752_train_data.csv')

# data = pd.read_csv('data/drug_cell/drug/Paclitaxel/Paclitaxel_train_data-rfe.csv')  # 0.6742	gBest: 963.44505,0.00473
# data_test = pd.read_csv('data/CGP/drug_cell/common_drugs/Paclitaxel_train_data.csv')

# data = pd.read_csv('data/drug_cell/drug/Erlotinib/Erlotinib_train_data-rfe.csv')  # 0.5903	gBest: 68.23076,0.82858
# data_test = pd.read_csv('data/CGP/drug_cell/common_drugs/Erlotinib_train_data.csv')

# CGP只有这个样本是均衡的
# CGP训练，ccle测试，-> 0.6575	gBest: 5.89952,0.04154
# data = pd.read_csv(
#     'data/drug_cell/drug/PD-0325901/PD-0325901_train_data-rfe-sa-10.csv')  # 0.5200	gBest: 997.84511,0.00566
# data_test = pd.read_csv('data/CGP/drug_cell/common_drugs/PD-0325901_train_data.csv')
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

x_test = data_test.iloc[:, :-1]
y_test = data_test.iloc[:, -1]
# X = X.as_matrix()
# y = y.as_matrix()

#
model = svm.SVC(C=149.70583, gamma=0.00009)  # gamma缺省值为 1.0/x.shape[1]
model.fit(X, y)
y_score = model.decision_function(x_test)
y_pre = model.predict(x_test)
f1_sco = f1_score(y_test, y_pre)
rec_sco = recall_score(y_test, y_pre)  # 召回率，查全率, tp/(tp+fn)
pre_sco_macro = precision_score(y_test, y_pre, average='macro')
pre_sco_wei = precision_score(y_test, y_pre, average='weighted')
pre_sco_bin = precision_score(y_test, y_pre)  # 精准率，查准率，tp/(tp+fp)
accuracy_sco = accuracy_score(y_test, y_pre)
print('f1', f1_sco)
print('pre_sco_macro', pre_sco_macro)
print('pre_sco_wei', pre_sco_wei)
print('pre_sco_bin', pre_sco_bin)
print('accuracy_sco', accuracy_sco)
print('SCORE', model.score(x_test, y_test))
print(np.array(y_test))
print('********************')
print(y_pre)
# print(type(y_pre))

fpr, tpr, threshold = roc_curve(y_test, y_score)  ###计算真正率和假正率
roc_auc = auc(fpr, tpr)  ###计算auc的值

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
plt.title('Sorafenib', fontsize=20)
plt.legend(loc="lower right", fontsize=20)
# plt.savefig('image/svm_roc1.png')
plt.show()
