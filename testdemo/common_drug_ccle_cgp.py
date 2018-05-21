import pandas as pd

data_cgp = pd.read_csv('../data/CGP/drug_cell/data_standard_res_CGP.csv')

data_ccle = pd.read_csv('../data/drug_cell/data_standard_res1.csv')

col_cgp = data_cgp.columns
col_ccle = data_ccle.columns

print(type(col_cgp))
# print(col_cgp)
list_cgp = list(col_cgp)
print(len(list_cgp))
# print(type(list_cgp))
list_ccle = list(col_ccle)
print(len(list_ccle))

print([l for l in list_cgp if l in list_ccle])
# set_cgp = set(list_cgp)
# # print(set_cgp)
# set_ccle = set(list_ccle)
# print(set_cgp & set_ccle)