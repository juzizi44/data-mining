import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import seaborn as sns
import matplotlib.pyplot as plt

# 读取CSV文件
data = pd.read_csv('data/raw/wdbc.data', header=None)
# 更新列名
columns = ['ID', 'Diagnosis'] + ['Feature_' + str(i) for i in range(1, 31)]
data.columns = columns
data['Diagnosis'] = data['Diagnosis'].replace({'M': 0, 'B': 1})
data['Diagnosis'] = data['Diagnosis'].astype(float)

# 分离特征和标签
X = data.iloc[:, 3:-1]
y = data.iloc[:, 2]

# 训练集和测试集划分
# 划分数据集的函数
def split_dataset(X, y, test_size):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=4882)
    return X_train, X_test, y_train, y_test

# 设置不同大小的训练集
train_sizes = [0.2, 0.4, 0.6, 0.8]
results = []

for size in train_sizes:
    X_train, X_test, y_train, y_test = split_dataset(X, y, test_size=1 - size)
    
    # 训练模型并预测
    adbr = AdaBoostRegressor(DecisionTreeRegressor(max_depth=4), n_estimators=100, random_state=126783)
    adbr.fit(X_train, y_train)
    y_pred = adbr.predict(X_test)

    
    # 计算均方误差
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    
    # 计算决定系数（R2 score）
    r2 = r2_score(y_test, y_pred)
    
    results.append({'Train Size': size, 'MSE': mse, 'MAE': mae, 'R2': r2})

# 打印结果
for result in results:
    size = result['Train Size']
    mse = result['MSE']
    mae = result['MAE']
    r2 = result['R2']
    print("Train Size: {:.2f}\tMSE: {:.2f}\tMAE: {:.2f}\tR2 Score: {:.3f}".format(size, mse, mae, r2))

# 创建一个数据框来存储预测结果
result_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})

# 使用seaborn绘制预测结果的散点图
sns.scatterplot(x='Actual', y='Predicted', data=result_df)

# 添加直线表示理想情况下的预测结果
sns.lineplot(x=result_df['Actual'], y=result_df['Actual'], linestyle='dashed', color='red', label='Ideal')

# 设置图形的标题和轴标签
plt.title('Regression Predictions')
plt.xlabel('Actual')
plt.ylabel('Predicted')


# 显示图形
plt.savefig('./img/regression_cancer_adaboost.eps', format='eps', dpi=500)

plt.show()