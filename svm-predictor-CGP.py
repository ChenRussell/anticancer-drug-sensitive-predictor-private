# -*- coding:utf-8 -*-
import pandas as pd
from sklearn import svm
from sklearn.metrics import roc_curve, auc  ###计算roc和auc
import matplotlib.pyplot as plt

# 0样本很多，1样本很少，得到的准确率都很高
data = pd.read_csv('data/drug_cell/drug/Lapatinib/Lapatinib_train_data-rfe.csv')
data_test = pd.read_csv('data/CGP/drug_cell/common_drugs/Lapatinib_train_data.csv')
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

x_test = data_test.iloc[:, :-1]
y_test = data_test.iloc[:, -1]
# X = X.as_matrix()
# y = y.as_matrix()

#
model = svm.SVC(C=5)  # gamma缺省值为 1.0/x.shape[1]
model.fit(X, y)
y_score = model.decision_function(x_test)
print(model.score(x_test, y_test))

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
