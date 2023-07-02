import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("data/raw/housing.data", header=None,
                   sep='\s+')  # sep='\s+'表示使用一个或者多个空格作为分隔符
data.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM',
                'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
data.head()

# z-score标准化
mean = np.mean(data, axis=0)
std = np.std(data, axis=0)
data = (data - mean) / std

# RM - 每间住宅的平均房间数【每栋住宅房间数】   DIS -波士顿的五个就业中心加权距离【与波士顿的五个就业中心加权距离】
# MEDV - 自有住房的中位数报价, 单位1000美元 【自住房屋房价中位数】
D2 = pd.concat([data['RM'], data['DIS'], data['MEDV']], axis=1)
D2.columns = ['RM', 'DIS', 'MEDV']
D2.to_csv('data/D2.csv')

# 绘制散点图
fig, ax = plt.subplots()
scatter = ax.scatter(D2.RM, D2.DIS, c=D2.MEDV)
ax.set_xlabel('RM')
ax.set_ylabel('DIS')
ax.set_title('Normalized D2 dataset')
fig.colorbar(scatter)
fig.savefig('img/Normalized_D2_dataset.eps', format='eps', dpi=1000)
