import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt


# 读取CSV文件
data = pd.read_csv('data/raw/wdbc.data', header=None)
# 更新列名
columns = ['ID', 'Diagnosis'] + ['Feature_' + str(i) for i in range(1, 31)]
data.columns = columns
data['Diagnosis'] = data['Diagnosis'].replace({'M': 0, 'B': 1})
data['Diagnosis'] = data['Diagnosis'].astype(float)

# 分离特征和标签
X = data.iloc[:, 2:-1]
y = data.iloc[:, 1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 训练模型
rfc = RandomForestClassifier(n_estimators=100,random_state=0) 
     # n_estimators：森林中决策树的数量。默认100
rfc = rfc.fit(X_train,y_train)
y_pred = rfc.predict(X_test)


# 计算评价指标
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

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
plt.ylim(0, 1.05)
plt.xlabel('Metrics', fontsize=14)
plt.grid(axis='y', alpha=0.5)
plt.legend().set_visible(False)

plt.savefig('./img/classification_cancer_randomforest.eps', format='eps', dpi=500)
plt.show()
plt.close()

