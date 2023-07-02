import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

D1 = pd.read_csv("data/D1.csv").values
# 定义m的取值范围
m_range = range(1, 6)

# 初始化MAE和RMSE的列表
MAE_list = []
RMSE_list = []

# 对于每个m值进行回归和测试
for m in m_range:
    # 将D1按照80%:20%的比例划分为Dtrain和Dtest
    np.random.shuffle(D1)
    split_index = int(len(D1) * 0.8)
    Dtrain = D1[:split_index]
    Dtest = D1[split_index:]

    # 将x转换为多项式特征
    poly = PolynomialFeatures(degree=m)
    Xtrain = poly.fit_transform(Dtrain[:, 0].reshape(-1, 1))
    Xtest = poly.transform(Dtest[:, 0].reshape(-1, 1))

    # 使用线性回归模型拟合数据
    model = LinearRegression()
    model.fit(Xtrain, Dtrain[:, 1])

    # 在测试集上进行预测
    y_pred = model.predict(Xtest)

    # 计算MAE和RMSE
    MAE = mean_absolute_error(Dtest[:, 1], y_pred)
    RMSE = np.sqrt(mean_squared_error(Dtest[:, 1], y_pred))

    # 将MAE和RMSE添加到列表中
    MAE_list.append(MAE)
    RMSE_list.append(RMSE)

    # 绘制拟合曲线
    x_plot = np.linspace(0, 4 * np.pi, 100)
    X_plot = poly.transform(x_plot.reshape(-1, 1))
    y_plot = model.predict(X_plot)
    plt.plot(x_plot, y_plot, label=f"m={m}")


# 绘制数据集和拟合曲线的图例和标签
x = np.linspace(0, 4 * np.pi, 100)
y = np.sin(x) + np.sin(2 * x)
plt.plot(x, y, label='Original')
plt.scatter(D1[:, 0], D1[:, 1], alpha=0.5, label="data")
plt.legend()
# 保存数据集和拟合曲线
plt.savefig('img/fitting_curve.eps', format='eps', dpi=1000)

# 绘制MAE和RMSE的条形图
fig, ax = plt.subplots()
opacity = 0.8
ax.bar(m_range, MAE_list, alpha=opacity, label="MAE")
ax.bar(m_range, RMSE_list, label="RMSE", alpha=opacity, bottom=MAE_list)
ax.legend()
ax.set_xticks(m_range)
ax.set_xlabel("m")
ax.set_ylabel("error")
ax.set_title("MAE and RMSE for different m")

# 保存MAE和RMSE的条形图
plt.savefig('img/error_bars.eps', format='eps', dpi=1000)

plt.show()
