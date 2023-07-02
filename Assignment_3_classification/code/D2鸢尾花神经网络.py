import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 读取iris.data文件
D2 = pd.read_csv('data/D2.csv', header=None)
# 添加列名
D2.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']

X = D2.iloc[:,:-1].values
y = D2.iloc[:,-1].values
le = LabelEncoder()
y = le.fit_transform(y)
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=666666)

# 将数据集分成训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


class NeuralNetwork:
    def __init__(self, input_dim, hidden_dim, output_dim, learning_rate=0.1):
        # 初始化权重矩阵
        self.W1 = np.random.randn(input_dim, hidden_dim)
        self.b1 = np.zeros((1, hidden_dim))
        self.W2 = np.random.randn(hidden_dim, output_dim)
        self.b2 = np.zeros((1, output_dim))

        # 定义激活函数和损失函数
        self.activation = self.sigmoid
        self.loss = self.cross_entropy

        # 定义学习率
        self.learning_rate = learning_rate

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))

    def cross_entropy(self, y_pred, y_true):
        m = y_true.shape[0]
        logprobs = np.multiply(np.log(y_pred), y_true) + np.multiply(np.log(1 - y_pred), 1 - y_true)
        cost = - np.sum(logprobs) / m
        return cost

    def forward(self, X):
        # 前向传播
        z1 = np.dot(X, self.W1) + self.b1
        a1 = self.activation(z1)
        z2 = np.dot(a1, self.W2) + self.b2
        y_pred = self.activation(z2)
        return y_pred, a1

    def backward(self, X, y_true, y_pred, a1):
        # 反向传播
        m = y_true.shape[0]
        delta3 = y_pred - y_true
        dW2 = np.dot(a1.T, delta3) / m
        db2 = np.sum(delta3, axis=0, keepdims=True) / m
        delta2 = np.dot(delta3, self.W2.T) * self.sigmoid_derivative(a1)
        dW1 = np.dot(X.T, delta2) / m
        db1 = np.sum(delta2, axis=0) / m

        # 更新权重和偏置项
        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1
        self.W2 -= self.learning_rate * dW2


# 初始化神经网络
nn = NeuralNetwork(input_dim=4, hidden_dim=10, output_dim=3)

# 训练神经网络
for i in range(1000):
    # 前向传播
    y_pred_train, a1_train = nn.forward(X_train)

    # 计算损失函数值
    loss = nn.loss(y_pred_train, np.eye(3)[y_train])

    # 反向传播
    nn.backward(X_train, np.eye(3)[y_train], y_pred_train, a1_train)

    # 每迭代100次打印一次损失函数值
    if i % 100 == 0:
        print(f"Loss after iteration {i}: {loss:.4f}")


def predict(X, nn):
    y_pred, _ = nn.forward(X)
    return np.argmax(y_pred, axis=1)


# 预测测试集
y_pred_test = predict(X_test, nn)
cm = confusion_matrix(y_test, y_pred_test)
print(cm)

# 绘制混淆矩阵
y = D2.iloc[:,-1]
fig, ax = plt.subplots()
im = ax.imshow(cm, cmap='Blues')
# 添加颜色条
cbar = ax.figure.colorbar(im, ax=ax)
# 设置标签和刻度
ax.set_xticks(range(len(y.unique() )))
ax.set_yticks(range(len(y.unique() )))
ax.set_xticklabels(y.unique() )
ax.set_yticklabels(y.unique() )
# 在每个方格内显示数值
thresh = cm.max() / 2
for i in range(len(y.unique() )):
    for j in range(len(y.unique() )):
        text = ax.text(j, i, cm[i, j], ha='center', va='center',color="white" if cm[i, j] > thresh else "black")
# 添加标题
ax.set_title('Confusion Matrix')
# 显示图形

plt.savefig('./img/net_iris_confusion_matrix.eps', format='eps', dpi=100, bbox_inches='tight')
plt.show()
plt.close()


accuracy = accuracy_score(y_test, y_pred_test)
precision = precision_score(y_test, y_pred_test, average='weighted')
recall = recall_score(y_test, y_pred_test, average='weighted')
f1 = f1_score(y_test, y_pred_test, average='weighted')

# 打印评价指标
print(f"Accuracy: {accuracy:.3f}")
print(f"Precision: {precision:.3f}")
print(f"Recall: {recall:.3f}")
print(f"F1-measure: {f1:.3f}")
# 保存评估指标柱状图图片
fig, ax = plt.subplots(figsize=(8,6))
bars = ax.bar(['Accuracy', 'Precision', 'Recall', 'F1-score'], [accuracy, precision, recall, f1],
              color=['#3399FF', '#FF6666', '#00CC99', '#FFA500'])
for bar in bars:
    height = bar.get_height()
    ax.annotate('{:.3f}'.format(height), xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=12)
plt.ylim(0, 1)
plt.xlabel('Metrics', fontsize=14)
plt.grid(axis='y', alpha=0.5)
plt.legend().set_visible(False)

plt.savefig('./img/net_iris_evaluation_metrics.eps', format='eps', dpi=1000, bbox_inches='tight')
plt.show()
plt.close()
#%%
