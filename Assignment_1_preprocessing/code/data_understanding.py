import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# 读取iris.data文件
D1 = pd.read_csv('data/D1.csv', header=None)
# 添加列名
D1.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']


# 计算每一列数值型数据的均值和方差
means = D1.mean()
stds = D1.std()

# 输出每一列数值型数据的均值和方差
# 打开txt文件
f = open('result/列均值方差.txt', 'w')
# 将要保存的输出写入txt文件中
print('均值：', file=f)
print(means,file=f)
print('方差：', file=f)
print(stds,file=f)
# 关闭txt文件
f.close()

# 绘制盒图
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))
plt.subplots_adjust(wspace=0.4, hspace=0.4)

for ax, col in zip(axes.flatten(), D1.columns):
    ax.boxplot(D1[col], vert=False, widths=0.6)
    ax.set_title(col)
    ax.set_xlabel('cm')
    ax.spines['right'].set_linewidth(1.5)
    ax.spines['top'].set_linewidth(1.5)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.tick_params(axis='both', which='both', length=0)
# 保存图片
plt.savefig('result/iris_boxplot.eps', format='eps',dpi=1000)





#%%
