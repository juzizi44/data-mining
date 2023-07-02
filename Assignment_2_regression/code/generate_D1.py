import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# 生成包含两个正弦周期的数据集
x = np.linspace(0, 4 * np.pi, 100)
y = np.sin(x) + np.sin(2 * x)

# 均匀采样 20 个数据样本
x_sampled = np.linspace(0, 4 * np.pi, 20)
y_sampled = np.sin(x_sampled) + np.sin(2 * x_sampled)

# 对每个样本的目标变量 yi 添加一个随机的扰动值
y_sampled += np.random.normal(0, 0.1, 20)

# 形成数据集 D1
D1 = np.column_stack((x_sampled, y_sampled))
pd.DataFrame(D1).to_csv('data/D1.csv', index=None)

# 绘图
fig, ax = plt.subplots()
scatter = ax.scatter(D1[:, 0], D1[:, 1], color='r', label='Sampled points')
plot = ax.plot(x, y, label='True function')
ax.set_xlabel('x_sampled')
ax.set_ylabel('y_sampled')
ax.set_title('True function vs Sampled points')

ax.legend()
# 保存图像
fig.savefig('img/scatter_plot.eps', format='eps', dpi=1000)
