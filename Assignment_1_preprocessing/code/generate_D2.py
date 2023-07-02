import pandas as pd
import numpy as np

# 读取iris.data文件
D1 = pd.read_csv('data/D1.csv', header=None)
# 添加列名
D1.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']
# 指定需要添加噪声的两列数据
col1 = 0  # 第1列
col2 = 2  # 第3列
# 在指定列上添加噪声
noisecol1 = np.random.normal(0, 1.0, D1.iloc[:, col1].shape)  # 创建大小为n的噪声数组
noisecol2 = np.random.normal(0, 0.2, D1.iloc[:, col2].shape)  # 创建大小为n的噪声数组

noisecol1 += D1.iloc[:, col1]   # 在第1列上添加噪声
noisecol2 += D1.iloc[:, col2]   # 在第3列上添加噪声

# 将这两列数据保存到文件 D2.csv
selected_cols = pd.DataFrame({'sepal_length': noisecol1, 'petal_length': noisecol2})
selected_cols.to_csv('data/D2.csv', index=False, header=False)

