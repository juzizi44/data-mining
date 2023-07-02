import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# 加载鸢尾花数据集
# 读取iris.data文件
D2 = pd.read_csv('data/D2.csv', header=None)
# 添加列名
D2.columns = ['sepal_length', 'sepal_width',
              'petal_length', 'petal_width', 'class']
X = D2.iloc[:, :-1]
y = D2.iloc[:, -1]

# 将数据集分成训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# 创建一个SVM分类器
svm = SVC(kernel='linear', C=1)
# 在训练集上训练分类器
svm.fit(X_train, y_train)
# 在测试集上进行预测
y_pred = svm.predict(X_test)
# 计算混淆矩阵
cm = confusion_matrix(y_test, y_pred)


# 绘制混淆矩阵
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
        text = ax.text(j, i, cm[i, j], ha='center', va='center', color="white" if cm[i, j] > thresh else "black")
# 添加标题
ax.set_title('Confusion Matrix')
# 显示图形

plt.savefig('./img/svm_iris_confusion_matrix.eps', format='eps', dpi=100, bbox_inches='tight')
plt.show()
plt.close()

# 计算评价指标
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')
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
plt.ylim(0, 1.1)
plt.xlabel('Metrics', fontsize=14)
plt.grid(axis='y', alpha=0.5)
plt.legend().set_visible(False)

plt.savefig('./img/svm_iris_evaluation_metrics.eps', format='eps', dpi=1000, bbox_inches='tight')
plt.show()
plt.close()

#%%
