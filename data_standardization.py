import pandas as pd

data = pd.read_excel('data/drug_cell/CCLE_GNF_data_090613.xls')
print(data.shape)

data2 = pd.read_csv('data/drug_cell/CCLE_NP24.2009_Drug_data_2015.02.24.csv')
# print(data2)

data2 = data2.iloc[:, [0, 2, -1]]  # 选择第0列-细胞系名称，第1列-药物名称， 最后一列-actArea

# 细胞系名称作为索引， 药物名称作为列名， actArea作为values
traindf = data2.pivot(index='CCLE Cell Line Name', columns='Compound', values='ActArea')

# 用0代替NaN
traindf = traindf.fillna(0)
# print(traindf)

# 将转换后的数据存储成文件
# traindf.to_csv('data/drug_cell/data_standard_res1.csv', float_format='%.2f')

# mean = traindf.mean()
# print(mean)
# print(mean.shape)
# print(type(mean))

# 零-均值规范化
data_normalize = (traindf - traindf.mean()) / traindf.std()
# print(data_normalize)

# 将标准化后的文件存储成文件
# data_normalize.to_csv('data/drug_cell/data_standard_normalize.csv', sep='\t', float_format='%.2f')

# 将大于平均值0.8的用1表示， 小于平均值0.8的用0来表示
data_col = data_normalize.columns
print(data_col)
print(type(data_col))

for col in data_col:
    data_normalize.loc[data_normalize[col] > 0.8, col] = 1  # 代表sensitive
    # data_normalize.loc[(0.8 > data_normalize[col] > -0.8), col] = -1
    data_normalize.loc[data_normalize[col] < -0.8, col] = 0  # 代表resistant
    # 选择标准化后结果为1或0的细胞系, ----为什么文件结果没有列名？？ 难道是因为只有一列？
    data_normalize.loc[(data_normalize[col] == 1) | (data_normalize[col] == 0), col] \
        .to_csv('data/drug_cell/drug/' + col + '.csv')

print(data_normalize)

# CCLE数据集中的药物
# ['17-AAG', 'AEW541', 'AZD0530', 'AZD6244', 'Erlotinib', 'Irinotecan',
#        'L-685458', 'LBW242', 'Lapatinib', 'Nilotinib', 'Nutlin-3',
#        'PD-0325901', 'PD-0332991', 'PF2341066', 'PHA-665752', 'PLX4720',
#        'Paclitaxel', 'Panobinostat', 'RAF265', 'Sorafenib', 'TAE684', 'TKI258',
#        'Topotecan', 'ZD-6474']

# CCLE和CGP数据集中公共的药物 data_col.interaction(data_col_cgp)
# ['Erlotinib', 'Lapatinib', 'Nilotinib--discard', 'PHA-665752', 'Paclitaxel',
#        'Sorafenib']
# PD0325901
# PLX-4720
