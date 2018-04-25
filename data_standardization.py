import pandas as pd

data = pd.read_excel('data/drug_cell/CCLE_GNF_data_090613.xls')
print(data.shape)

data2 = pd.read_csv('data/drug_cell/CCLE_NP24.2009_Drug_data_2015.02.24.csv')
# print(data2)

data2 = data2.iloc[:, [0,2,-1]]

traindf=data2.pivot(index='CCLE Cell Line Name', columns='Compound', values='ActArea')

# 用0代替NaN
traindf = traindf.fillna(0)
# print(traindf)
# traindf.to_csv('data/drug_cell/data_standard_res1.csv', float_format='%.2f')

# mean = traindf.mean()
# print(mean)
# print(mean.shape)
# print(type(mean))

# 零-均值规范化
data_normalize = (traindf - traindf.mean())/traindf.std()
# print(data_normalize)
# data_normalize.to_csv('data/drug_cell/data_standard_normalize.csv', sep='\t', float_format='%.2f')

# 将大于平均值0.8的用1表示， 小于平均值0.8的用0来表示
data_col = data_normalize.columns
print(data_col)
print(type(data_col))

for col in data_col:
    data_normalize.loc[data_normalize[col] > 0.8, col] = 1
    # data_normalize.loc[(0.8 > data_normalize[col] > -0.8), col] = -1
    data_normalize.loc[data_normalize[col] < -0.8, col] = 0
    data_normalize.loc[(data_normalize[col] == 1) | (data_normalize[col] == 0), col].to_csv('data/drug_cell/drug/'+col+'.csv')

print(data_normalize)