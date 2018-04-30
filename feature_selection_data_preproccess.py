import pandas as pd

# 读取mRNA expression
data = pd.read_csv('data/CCLE_Expression_Entrez_2012-09-29 _copy.gct', delimiter='\t', error_bad_lines=False)

print(data.shape)
print(data.columns)


data = data.drop('Description', axis=1)  # 删除Description列, 改为删除Name列-----不能改，因为Description有NaN值
cols = data.columns
print(cols)

# data = data.pivot(index='Name')
data.index = data['Name'].tolist()  # 将Name列转换为索引列， 改为Description
data = data.drop('Name', axis=1)    # 删除Name列, 改为Description

# print(data.head(5))

data_stack = data.stack()
# print(data_stack.head(5))
data_unstack = data_stack.unstack(0)    # 行列转换
print(data_unstack.head(5))
# 转换成文件, index=False, 去掉索引
# data_unstack.to_csv('data/gene_expression/ccle_expression_trans_index_col.csv',index=False, float_format='%.2f')


# 将每一种药物的标签与基因表达数据结合
# drug_info = pd.read_csv('data/drug_cell/drug/17-AAG.csv', header=None)
# drug_info = pd.read_csv('data/drug_cell/drug/Erlotinib.csv', header=None)
# drug_info = pd.read_csv('data/drug_cell/drug/Irinotecan.csv', header=None)
# drug_info = pd.read_csv('data/drug_cell/drug/Lapatinib.csv', header=None)
# drug_info = pd.read_csv('data/drug_cell/drug/PD-0325901.csv', header=None)
drug_info = pd.read_csv('data/drug_cell/drug/AEW541.csv', header=None)
# print(drug_info)
drug_info_cell_Col = drug_info[0]   # 选择cell列
# print(drug_info_cell_Col)
# print(type(drug_info_cell_Col))
drug_info_label_Col = drug_info[1]  # 选择label列
print(drug_info_label_Col)
print(len(drug_info_label_Col))

# 选择指定的cell行
data_unstack_select = data_unstack.loc[drug_info_cell_Col]
# print(data_unstack_select)
data_unstack_select['label'] = drug_info_label_Col.values
print(data_unstack_select)
data_unstack_select.to_csv('data/drug_cell/drug/AEW541_train_data.csv',index=False, float_format='%.2f')