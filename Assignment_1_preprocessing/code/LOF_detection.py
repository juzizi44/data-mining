import matplotlib.pyplot as plt
import pandas as pd
from sklearn.neighbors import LocalOutlierFactor

# 读取数据
df = pd.read_csv("data/D2.csv", header=None)

# 使用LOF方法检测离群点
clf = LocalOutlierFactor(n_neighbors=20)
y_pred = clf.fit_predict(df)

# 绘制数据点
plt.scatter(df[0], df[1], c=y_pred, cmap="cool" , label='Inlier')

# 标记离群点
outliers = df[y_pred == -1]
plt.scatter(outliers[0], outliers[1], edgecolors="b", facecolors="none", s=80 , label='Outlier')
plt.legend()

# 添加横纵坐标和标题
plt.xlabel("sepal length")
plt.ylabel("petal length")
plt.title("Outlier Factor")

plt.savefig('result/lof.eps', format='eps', dpi=1000)
