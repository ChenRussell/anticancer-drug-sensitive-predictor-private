# 首先导入基本的绘图包
import matplotlib.pyplot as plt
import pandas as pd

# 添加成绩表
plt.style.use("ggplot")
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.sans-serif'] = ['SimHei']

# 新建一个空的DataFrame
df = pd.DataFrame()
# 添加成绩单，最后显示成绩单表格
df["769P_KIDNEY"] = [3.9, 4.2, 4.8, 5.3, 5.6, 6.1, 6.7, 7.5, 8.1, 8.5, 9.1]
df["8305C_THYROID"] = [3.7, 4.1, 4.8, 5.3, 5.6, 6.1, 6.7, 7.5, 8.1, 8.5, 10.5]
df["A2058_SKIN"] = [4.4, 4.2, 4.8, 5.3, 5.6, 6.1, 6.7, 7.5, 9.1, 10.5, 12.1]
df["AU565_BREAST"] = [3.6, 4.4, 4.8, 5.1, 5.6, 6.3, 6.7, 7.8, 8.2, 8.5, 9.8]
# df["MC-CAR_Blood"] = [3, 4000, 4500, 5300, 6000, 6600, 7200, 8100, 8800, 9100, 10000]
# df["ES5_BONE"] = [4, 4100, 4600, 5200, 6200, 6700, 7100, 8200, 8900, 9330, 9000]
# df["5637_Bladder"] = [5, 4200, 4500, 5600, 6100, 6200, 7400, 8300, 8800, 9100, 12000]
# df["C-4-I_Cervix"] = [4, 4100, 4300, 5400, 6200, 6600, 7300, 8200, 8400, 8900, 9200]
df["MC-CAR_Blood"] = [3, 4, 4, 5.300, 6.000, 6.600, 7.200, 8.100, 8.800, 9.100, 10.000]
df["ES5_BONE"] = [4, 4.100, 4.600, 5.200, 6.200, 6.700, 7.100, 8.200, 8.900, 9.330, 9.000]
df["5637_Bladder"] = [5, 4.200, 4.500, 5.600, 6.100, 6.200, 7.400, 8.300, 8.800, 9.100, 12.000]
df["C-4-I_Cervix"] = [4, 4.100, 4.300, 5.400, 6.200, 6.600, 7.300, 8.200, 8.400, 8.900, 9.200]

# 用matplotlib来画出箱型图
bplot = plt.boxplot(x=df.values, patch_artist=True, labels=df.columns, whis=1.5)

colors = ['RED', 'RED', 'RED', "RED", "LIGHTBLUE", "LIGHTBLUE", "LIGHTBLUE", "LIGHTBLUE"]
for patch, color in zip(bplot['boxes'], colors):
    patch.set_facecolor(color)  # 为不同的箱型图填充不同的颜色

plt.title('使用Combat之后', fontsize=22)
plt.xticks(fontsize=18)
# plt.yscale('log')
plt.show()
