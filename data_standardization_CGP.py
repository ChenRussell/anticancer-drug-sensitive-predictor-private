import pandas as pd

data = pd.read_excel('data/CGP/v17.3_fitted_dose_response.xlsx')
print(data.shape)

# data2 = pd.read_csv('data/drug_cell/CCLE_NP24.2009_Drug_data_2015.02.24.csv')
# print(data2)

data2 = data.iloc[:, [2, 5, 9]]  # 选择第2列-细胞系ID，第5列-药物名称， 9列-IC50

# 细胞系名称作为索引， 药物名称作为列名， actArea作为values
traindf = data2.pivot_table(index='COSMIC_ID', columns='DRUG_NAME', values='LN_IC50')

# 用0代替NaN
traindf = traindf.fillna(0)
# print(traindf)

# 将转换后的数据存储成文件
# traindf.to_csv('data/CGP/drug_cell/data_standard_res_CGP.csv', float_format='%.2f')

# mean = traindf.mean()
# print(mean)
# print(mean.shape)
# print(type(mean))

# 零-均值规范化
data_normalize = (traindf - traindf.mean()) / traindf.std()
# print(data_normalize)

# 将标准化后的文件存储成文件
# data_normalize.to_csv('data/CGP/drug_cell/data_standard_normalize.csv', sep='\t', float_format='%.2f')

# 将大于平均值0.8的用1表示， 小于平均值0.8的用0来表示
data_col_cgp = data_normalize.columns
print(data_col_cgp)
print(type(data_col_cgp))

for col in data_col_cgp:
    data_normalize.loc[data_normalize[col] > 0.8, col] = 0  # 代表sensitive
    # data_normalize.loc[(0.8 > data_normalize[col] > -0.8), col] = -1
    data_normalize.loc[data_normalize[col] < -0.8, col] = 1  # 代表resistant
    # 选择标准化后结果为1或0的细胞系, ----为什么文件结果没有列名？？ 难道是因为只有一列？
    if col == 'VNLG/124':
        data_normalize.loc[(data_normalize[col] == 1) | (data_normalize[col] == 0), col] \
            .to_csv('data/CGP/drug_cell/drug/VNLG_124.csv')
    else:
        data_normalize.loc[(data_normalize[col] == 1) | (data_normalize[col] == 0), col] \
            .to_csv('data/CGP/drug_cell/drug/' + col + '.csv')

print(data_normalize)
