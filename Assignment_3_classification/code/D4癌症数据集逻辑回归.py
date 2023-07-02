import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


class MyLogisticRegression:
    def __init__(self, learning_rate=0.001, max_iter=10000):
        # 初始化逻辑回归模型的参数
        self._theta = None
        self.intercept_ = None
        self.coef_ = None
        # 设置学习率和最大迭代次数
        self.learning_rate = learning_rate
        self.max_iter = max_iter

    def _sigmoid(self, z):
        # 定义sigmoid函数，用于输出0-1之间的概率值
        return 1. / (1. + np.exp(-z))

    def fit(self, x_train, y_train):
        # 定义代价函数J和其梯度dJ
        def J(theta, X_b, y_train):
            y_hat = self._sigmoid(X_b.dot(theta))
            return - np.sum(y_train*np.log(y_hat) + (1-y_train)*np.log(1-y_hat)) / len(y_train)

        def dJ(theta, X_b, y_train):
            y_hat = self._sigmoid(X_b.dot(theta))
            return X_b.T.dot(y_hat - y_train) / len(y_train)

        # 对训练集添加偏置项，并随机初始化模型参数
        X_b = np.hstack([np.ones((len(x_train), 1)), x_train])
        self._theta = np.random.randn(X_b.shape[1])
        # 使用梯度下降法来训练模型
        iter_num = 0
        while iter_num < self.max_iter:
            iter_num += 1
            last_theta = self._theta
            self._theta = self._theta - self.learning_rate * \
                          dJ(self._theta, X_b, y_train)
            # 当前代价函数值和上一次的代价函数值之差小于1e-7时停止迭代
            if (abs(J(self._theta, X_b, y_train) - J(last_theta, X_b, y_train)) < 1e-7):
                break
        # 将最终得到的参数分别赋值给拟合结果的属性
        self.intercept_ = self._theta[0]
        self.coef_ = self._theta[1:]
        return self

    def predict(self, x_predict):
        # 对待预测的数据添加偏置项，并根据阈值0.5来将概率值转换成二分类标签
        X_b = np.hstack([np.ones((len(x_predict), 1)), x_predict])
        y_predict = self._sigmoid(X_b.dot(self._theta))
        y_predict = np.array(y_predict >= 0.5, dtype='int')
        return y_predict

    def score(self, x_test, y_test):
        # 根据预测结果计算混淆矩阵并返回
        y_predict = self.predict(x_test)
        cm = confusion_matrix(y_test, y_predict)
        return cm

    def __repr__(self):
        return "LogisticRegression()"


# 读取数据
D4 = pd.read_csv('data/raw/breast-cancer-wisconsin.data',
                   names=['id', 'clump_thickness', 'uniformity_cell_size', 'uniformity_cell_shape', 'marginal_adhesion',
                          'single_epithelial_size', 'bare_nuclei', 'bland_chromatin', 'normal_nucleoli', 'mitoses',
                          'class'],
                   na_values='?')
D4.to_csv('data/D4.csv')

# 使用前向填充的方式填充缺失值
D4.fillna(method='ffill', inplace=True)

# 删除id列
D4.drop('id', axis=1, inplace=True)

# 对特征数据进行标准化
D4.iloc[:, :-1] = (D4.iloc[:, :-1] - D4.iloc[:, :-
1].mean()) / D4.iloc[:, :-1].std()

# 将二分类标签转换为0和1
D4.loc[:, -1] = D4.iloc[:, -1].replace({2: 0, 4: 1})

# 获取特征和标签数据
x_D4 = D4.iloc[:, :-1].values
y_D4 = D4.iloc[:, -1].values



# 划分数据集为训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(
    x_D4, y_D4, test_size=0.2, random_state=4444)

# 创建逻辑回归模型
model = MyLogisticRegression()

# 训练模型
model.fit(x_train, y_train)

# 预测测试集结果
model.predict(x_test)

# 计算混淆矩阵和指标
cm = model.score(x_test, y_test)
print(cm)
accuracy = (cm[0, 0]+cm[1, 1])/np.sum(cm)
precision = cm[1, 1]/(cm[0, 1]+cm[1, 1])
recall = cm[1, 1]/(cm[1, 0]+cm[1, 1])
f1_score = 2*precision*recall/(precision+recall)

# 输出模型评估指标
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1_score)



# 保存混淆矩阵图片
thresh = cm.max() / 2
plt.imshow(cm, cmap=plt.cm.Blues)
plt.xticks([0, 1], ['benign', 'malignant'])
plt.yticks([0, 1], ['benign', 'malignant'])
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.colorbar()
for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i, j], ha='center', va='center',color="white" if cm[i, j] > thresh else "black")
plt.savefig('./img/logistic_regression_confusion_matrix.eps', format='eps', dpi=500, bbox_inches='tight')
plt.show()
plt.close()

# 保存评估指标柱状图图片
fig, ax = plt.subplots(figsize=(8,6))
bars = ax.bar(['Accuracy', 'Precision', 'Recall', 'F1-score'], [accuracy, precision, recall, f1_score],
              color=['#3399FF', '#FF6666', '#00CC99', '#FFA500'])
for bar in bars:
    height = bar.get_height()
    ax.annotate('{:.3f}'.format(height), xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=12)
plt.ylim(0.8, 1)
plt.xlabel('Metrics', fontsize=14)
plt.grid(axis='y', alpha=0.5)
plt.legend().set_visible(False)

plt.savefig('./img/logistic_regression_evaluation_metrics.eps', format='eps', dpi=500)
plt.show()
plt.close()

#%%
