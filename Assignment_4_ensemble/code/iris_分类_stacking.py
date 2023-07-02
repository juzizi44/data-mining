from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import StackingClassifier
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing


# 数据预处理
# 读取iris.data文件
D1 = pd.read_csv('data/raw/iris.data', header=None)
# 添加列名
D1.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']
# 将字符类型转化为数值类型
le = preprocessing.LabelEncoder()
for col in D1.columns:
    if D1[col].dtype == 'object':
        D1[col] = le.fit_transform(D1[col])
X = D1.iloc[:, :-1]
y = D1.iloc[:, -1] 



# 划分训练集、测试集和验证集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# 定义基础分类器
base_classifiers = [
    ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
    ('gb', GradientBoostingClassifier(n_estimators=100, random_state=42)),
    ('kn', KNeighborsClassifier())
]

# 定义元分类器
meta_classifier = LogisticRegression()

# 定义Stacking模型
stacking_model = StackingClassifier(estimators=base_classifiers, final_estimator=meta_classifier)

# 训练Stacking模型
stacking_model.fit(X_train, y_train)

# 在验证集上评估模型性能
val_pred = stacking_model.predict(X_val)
val_accuracy = accuracy_score(y_val, val_pred)
val_precision = precision_score(y_val, val_pred,average='macro')
val_recall = recall_score(y_val, val_pred,average='macro')
val_f1 = f1_score(y_val, val_pred,average='macro')

print("Validation Accuracy: {:.4f}".format(val_accuracy))
print("Validation Precision: {:.4f}".format(val_precision))
print("Validation Recall: {:.4f}".format(val_recall))
print("Validation F1-score: {:.4f}".format(val_f1))



#将val评价指标可视化
fig, ax = plt.subplots(figsize=(8,6))
bars = ax.bar(['Accuracy', 'Precision', 'Recall', 'F1-score'], [val_accuracy, val_precision, val_recall, val_f1],
              color=['#3399FF', '#FF6666', '#00CC99', '#FFA500'])
for bar in bars:
    height = bar.get_height()
    ax.annotate('{:.3f}'.format(height), xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=12)
plt.ylim(0, 1.05)
plt.xlabel('Metrics', fontsize=14)
plt.grid(axis='y', alpha=0.5)
plt.legend().set_visible(False)

plt.savefig('./img/classification_iris_stacking_val.eps', format='eps', dpi=500)
plt.show()
plt.close()

# 在测试集上评估模型性能
test_pred = stacking_model.predict(X_test)
test_accuracy = accuracy_score(y_test, test_pred)
test_precision = precision_score(y_test, test_pred,average='macro')
test_recall = recall_score(y_test, test_pred,average='macro')
test_f1 = f1_score(y_test, test_pred,average='macro')

print("Test Accuracy: {:.4f}".format(test_accuracy))
print("Test Precision: {:.4f}".format(test_precision))
print("Test Recall: {:.4f}".format(test_recall))
print("Test F1-score: {:.4f}".format(test_f1))

#将测试集上评价指标可视化
fig, ax = plt.subplots(figsize=(8,6))
bars = ax.bar(['Accuracy', 'Precision', 'Recall', 'F1-score'], [test_accuracy, test_precision, test_recall, test_f1],
              color=['#3399FF', '#FF6666', '#00CC99', '#FFA500'])
for bar in bars:
    height = bar.get_height()
    ax.annotate('{:.3f}'.format(height), xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=12)
plt.ylim(0, 1.05)
plt.xlabel('Metrics', fontsize=14)
plt.grid(axis='y', alpha=0.5)
plt.legend().set_visible(False)

plt.savefig('./img/classification_iris_stacking_test.eps', format='eps', dpi=500)
plt.show()
plt.close()