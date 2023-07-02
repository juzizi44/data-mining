# 导入所需的库
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,roc_auc_score
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt


# 数据预处理
# 读取iris.data文件
D1 = pd.read_csv('data/raw/iris.data', header=None)
# 添加列名
D1.columns = ['sepal_length', 'sepal_width',
              'petal_length', 'petal_width', 'class']
# 将字符类型转化为数值类型
le = preprocessing.LabelEncoder()
for col in D1.columns:
    if D1[col].dtype == 'object':
        D1[col] = le.fit_transform(D1[col])


# 训练集和测试集划分
X = D1.iloc[:, :-1]
y = D1.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

#训练模型
ada_best = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(),
                              random_state=42)

# 训练模型并预测
ada_best.fit(X_train, y_train)
y_pred = ada_best.predict(X_test)

# 计算评价指标
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro')

# 打印评价指标
print(f"Accuracy: {accuracy:.3f}")
print(f"Precision: {precision:.3f}")
print(f"Recall: {recall:.3f}")
print(f"F1-measure: {f1:.3f}")

#将评价指标可视化
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

plt.savefig('./img/classification_iris_adaboost.eps', format='eps', dpi=500)
plt.show()
plt.close()