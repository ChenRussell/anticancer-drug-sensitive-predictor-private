import pandas as pd

# 读取mRNA expression
data = pd.read_csv('data/CGP/sanger1018_brainarray_ensemblgene_rma.txt', delimiter='\t', error_bad_lines=False)

print(data.shape)
print(data.columns)
print(data.head(5))

data.index = data['ensembl_gene'].tolist()  # 将ensembl_gene列转换为索引列
data = data.drop('ensembl_gene', axis=1)  # 删除ensembl_gene列

data_stack = data.stack()
# print(data_stack.head(5))
data_unstack = data_stack.unstack(0)  # 行列转换
print(data_unstack.head(5))
# 转换成文件, index=False, 去掉索引
# data_unstack.to_csv('data/CGP/gene_expression/ccle_expression_trans_index_col.csv',index=False, float_format='%.2f')

# Lapatinib
# cols = ['ENSG00000154639','ENSG00000151012','ENSG00000175591','ENSG00000102595','ENSG00000142765',
#         'ENSG00000171004','ENSG00000128512']

# Sorafenib
cols = ['ENSG00000131149', 'ENSG00000104093', 'ENSG00000135069', 'ENSG00000110002', 'ENSG00000196387',
        'ENSG00000102445', 'ENSG00000169583', 'ENSG00000102362', 'ENSG00000198846']
data_unstack = data_unstack[cols]

print(data_unstack.head(5))
print(data_unstack.shape)

drug_info = pd.read_csv('data/CGP/drug_cell/drug/Lapatinib.csv', header=None)

drug_info_cell_Col = drug_info[0]  # 选择cell列
drug_info_label_Col = drug_info[1]  # 选择label列
print(drug_info_cell_Col)
print(type(drug_info_cell_Col))
drug_info_cell_Col_list = list(drug_info_cell_Col)
drug_info_cell_Col_str = [str(i) for i in drug_info_cell_Col_list]  # 将数字转换成字符串

# 选择指定的cell行
data_unstack_select = data_unstack.loc[drug_info_cell_Col_str]
data_unstack_select['label'] = drug_info_label_Col.values
print(data_unstack_select)
# data_unstack_select.fillna(0)
data_unstack_select.to_csv('data/CGP/drug_cell/common_drugs/Sorafenib_train_data.csv', index=False, float_format='%.2f')
