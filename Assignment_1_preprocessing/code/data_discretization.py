import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.preprocessing import KBinsDiscretizer  # 等频离散化
from caimcaim import CAIMD                          # CAIM离散化
import own_information_gain_discretization          # 自己的信息增益离散化
import own_chimerge                                 # 自己的卡方离散化

#######################################################################################
# 读取数据集D1
D1 = pd.read_csv('data/D1.csv', header=0,
                 names=['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class'])

#######################################################################################
# 等频离散化
eqf_discretizer = KBinsDiscretizer(
    n_bins=5, encode='ordinal', strategy='quantile')
sepal_length = eqf_discretizer.fit_transform(D1.iloc[:, :1])

# 绘制散点图
fig, ax = plt.subplots()
scatter = ax.scatter(D1.iloc[:, :1], D1.index, c=sepal_length)
ax.set_xlabel('sepal_length')
ax.set_ylabel('Index')
ax.set_title('Discretized Sepal Length')
fig.colorbar(scatter, ticks=[0, 1, 2, 3, 4])
fig.savefig('result/discretized_sepalLength.eps', format='eps', dpi=1000)

#######################################################################################
# 信息增益离散化
discretized_subsets = own_information_gain_discretization.discretize(D1, 'sepal_width', 0, 10, 3)
discretized_data = pd.DataFrame()
i = 0
for subset in discretized_subsets:
    # print(subset)
    # print('----------------------------------------')
    # ax.scatter(subset['sepal_width'], subset.index)
    subset['sepal_width'] = i
    discretized_data = pd.concat([discretized_data, subset])
    i += 1
discretized_data = discretized_data.sort_index()
sepal_width = discretized_data['sepal_width']

# 绘制散点图
fig, ax = plt.subplots()
scatter = ax.scatter(D1['sepal_width'], D1.index, c=sepal_width)
ax.set_xlabel('Sepal Width')
ax.set_ylabel('Index')
ax.set_title('Discretized on Sepal Width')
fig.colorbar(scatter, ticks=[0, 1, 2, 3, 4, 5, 6])
fig.savefig('result/discretized_sepalWidth.eps', format='eps', dpi=1000)

#######################################################################################
# 卡方离散化
bins = own_chimerge.my_chimerge(feature='petal_length', data=D1, max_interval=5)
bins.append(D1['petal_length'].max())
petal_length = pd.cut(D1['petal_length'], bins, labels=[0, 1, 2, 3, 4], include_lowest=True)

# 绘制散点图
fig, ax = plt.subplots()
scatter = ax.scatter(D1['petal_length'], D1.index, c=petal_length)
ax.set_xlabel('petal_length')
ax.set_ylabel('Index')
ax.set_title('Discretized Petal Length')
fig.colorbar(scatter, ticks=[0, 1, 2, 3, 4])
fig.savefig('result/discretized_petalLength.eps', format='eps', dpi=1000)

#######################################################################################
# CAIM离散化
X = D1.iloc[:, :4].values
y = D1['class']
caim = CAIMD()
x_disc = caim.fit_transform(X, y)
petal_width = x_disc[:, 3]

# 绘制散点图
fig, ax = plt.subplots()
scatter = ax.scatter(D1['petal_width'], D1.index, c=petal_width)
ax.set_xlabel('Petal Width')
ax.set_ylabel('Index')
ax.set_title('Discretized on Petal Width')
fig.colorbar(scatter, ticks=[0, 1, 2])
fig.savefig('result/discretized_petalWidth.eps', format='eps', dpi=1000)

#######################################################################################
#保存离散化后的数据
D1 = pd.read_csv('data/D1.csv', header=0,
                 names=['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class'])
D1['sepal_length'] = sepal_length
D1['sepal_width'] = sepal_width
D1['petal_length'] = petal_length
D1['petal_width'] = petal_width
D1.to_csv('data/D1-discrete.csv', index=False, header=False)
