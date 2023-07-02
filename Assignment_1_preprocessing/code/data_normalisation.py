import pandas as pd
import numpy as np

# 读取iris.data文件
D1 = pd.read_csv('data/D1.csv', header=None)

# z-score标准化
mean = np.mean(D1.iloc[:, 0:4], axis=0)
std = np.std(D1.iloc[:, 0:4], axis=0)
D1.iloc[:, 0:4] = (D1.iloc[:, 0:4] - mean) / std
D1.to_csv('data/D1_zscore.csv', index=False, header=False)

# 读取iris.data文件
D1 = pd.read_csv('data/D1.csv', header=None)
# min-max标准化
min_val = np.min(D1.iloc[:, 0:4], axis=0)
max_val = np.max(D1.iloc[:, 0:4], axis=0)
D1.iloc[:, 0:4] = (D1.iloc[:, 0:4] - min_val) / (max_val - min_val)
D1.to_csv('data/D1_minmax.csv', index=False, header=False)

# 读取iris.data文件
D1 = pd.read_csv('data/D1.csv', header=None)
# 十进制小数定标标准化
factor = 10 ** np.ceil(np.log10(np.max(D1.iloc[:, 0:4])))
D1.iloc[:, 0:4] = D1.iloc[:, 0:4] / factor
D1.to_csv('data/D1_float.csv', index=False, header=False)

# 读取iris.data文件
D1 = pd.read_csv('data/D1.csv', header=None)
# logistic标准化
D1.iloc[:, 0:4]= 1 / (1 + np.exp(-D1.iloc[:, 0:4]))
D1.to_csv('data/D1_log.csv', index=False, header=False)



#%%
