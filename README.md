# anticancer-preditor实现

## 数据预处理
    1. data_standardization.py, 对药物-细胞系数据进行0-1标准化处理，将样本进行分类(1-sensitive,0-resistant)
    2. feature_selection_DataPreprocess.py, 对基因表达数据进行预处理，选择指定药物样本所包含的细胞系，得到训练数据
       ,供特征选择使用。

## 1. Combat.R 批次效应消除

## 2. feature_selection_svm-rfe.py 使用循环特征消除作特征选择
    使用svm-rfe进行特征选择，由于基因表达数据数据维度大(>10000)，因此该算法非常耗时。

## 3. feature_selection_MRMR.py 使用MRMR作特征选择
    使用MRMR算法进行特征选择。

## 4. feature_selection_final.py 综合的特征选择算法
    综合前两种特征选择算法，并加入模拟退火的思想进行改进。

## 5. svm-predictor.py 构建svm分类器
    使用svm分类器进行分类预测。

## 6. svm-pso-predictor.py 构建基于粒子群算法优化的svm分类器
    使用粒子群算法优化svm的两个参数(C,r)
