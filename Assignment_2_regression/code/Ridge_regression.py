import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

data = pd.read_csv('data/D2.csv')
# 提取D2列作为训练集
X = data[['RM', 'DIS']].values
y = data['MEDV'].values


def ridge_regression(X, y, alpha):
    """
    实现Ridge回归模型

    参数：
        X - 自变量，形状为(n_samples, n_features)
        y - 因变量，形状为(n_samples, )
        alpha - 正则化系数

    返回值：
        回归系数，形状为(n_features, )
    """
    n_features = X.shape[1]
    A = np.dot(X.T, X) + alpha * np.identity(n_features)
    b = np.dot(X.T, y)
    return np.linalg.solve(A, b)


# 变换正则化系数λ的取值，并获取正则化路径数据
alphas = [0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000, 1000000]
coefs = []
for a in alphas:
    coef = ridge_regression(X, y, a)
    coefs.append(coef)
coefs = np.array(coefs)

# 绘制正则化路径图
plt.figure(figsize=(10, 6))
ax = plt.gca()
ax.plot(alphas, coefs[:, 0], label='RM')
ax.plot(alphas, coefs[:, 1], label='DIS')
ax.plot(alphas, coefs.sum(axis=1), label='Total')
ax.set_xscale('log')
ax.set_xlim(ax.get_xlim())
plt.xlabel('alpha')
plt.ylabel('weights')
plt.title('Ridge coefficients as a function of the regularization')
plt.axis('tight')
plt.legend()
plt.savefig('img/The_path_of_regularization.eps', format='eps', dpi=1000)
plt.show()


# 将D2随机划分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
# 使用训练集训练Ridge回归模型
alpha = 10
coef = ridge_regression(X_train, y_train, alpha)

# 重复以上步骤5次或以上，获取多组MAE和RMSE值，并绘制条形图
n_runs = 10
mae_list = []
rmse_list = []
for i in range(n_runs):
    # 将D2随机划分为训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    # 使用训练集训练Ridge回归模型
    coef = ridge_regression(X_train, y_train, alpha)

    # 在测试集上进行测试，并计算MAE和RMSE
    y_pred = np.dot(X_test, coef)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    mae_list.append(mae)
    rmse_list.append(rmse)

# 将多组MAE和RMSE值绘制成条形图
plt.figure(figsize=(8, 6))
index = np.arange(n_runs+1)
opacity = 0.8

# 绘制MAE条形图
plt.bar(index, mae_list+[np.mean(mae_list)], alpha=opacity, label='MAE')
# 绘制RMSE条形图
plt.bar(index, rmse_list+[np.mean(rmse_list)], alpha=opacity,
        label='RMSE', bottom=mae_list+[np.mean(mae_list)])

# 添加标签和标题
plt.xlabel('Run')
plt.ylabel('Error')
plt.title('MAE and RMSE of Ridge Regression with alpha={}'.format(alpha))
# plt.xticks(index + bar_width / 2, ['Run {}'.format(i+1) for i in range(n_runs)]+['Average'])
plt.xticks(index, ['Run {}'.format(i+1) for i in range(n_runs)]+['Average'])

plt.legend()
plt.savefig('img/MAE_and_RMSE_of_Ridge.eps', format='eps', dpi=1000)

plt.show()
