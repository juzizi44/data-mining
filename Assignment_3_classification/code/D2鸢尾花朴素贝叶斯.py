import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from math import log
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

class NaiveBayes:
    def __init__(self):
        self.model = {}#key 为类别名 val 为字典PClass表示该类的该类，PFeature:{}对应对于各个特征的概率
    def calEntropy(self, y): # 计算熵
        valRate = y.value_counts().apply(lambda x : x / y.size) # 频次汇总 得到各个特征对应的概率
        valEntropy = np.inner(valRate, np.log2(valRate)) * -1
        return valEntropy

    def fit(self, xTrain, yTrain = pd.Series()):
        if not yTrain.empty:#如果不传，自动选择最后一列作为分类标签
            xTrain = pd.concat([xTrain, yTrain], axis=1)
        self.model = self.buildNaiveBayes(xTrain)
        return self.model
    def buildNaiveBayes(self, xTrain):
        yTrain = xTrain.iloc[:,-1]

        yTrainCounts = yTrain.value_counts()# 频次汇总 得到各个特征对应的概率

        yTrainCounts = yTrainCounts.apply(lambda x : (x + 1) / (yTrain.size + yTrainCounts.size)) #使用了拉普拉斯平滑
        retModel = {}
        for nameClass, val in yTrainCounts.items():
            retModel[nameClass] = {'PClass': val, 'PFeature':{}}

        propNamesAll = xTrain.columns[:-1]
        allPropByFeature = {}
        for nameFeature in propNamesAll:
            allPropByFeature[nameFeature] = list(xTrain[nameFeature].value_counts().index)
        #print(allPropByFeature)
        for nameClass, group in xTrain.groupby(xTrain.columns[-1]):
            for nameFeature in propNamesAll:
                eachClassPFeature = {}
                propDatas = group[nameFeature]
                propClassSummary = propDatas.value_counts()# 频次汇总 得到各个特征对应的概率
                for propName in allPropByFeature[nameFeature]:
                    if not propClassSummary.get(propName):
                        propClassSummary[propName] = 0#如果有属性灭有，那么自动补0
                Ni = len(allPropByFeature[nameFeature])
                propClassSummary = propClassSummary.apply(lambda x : (x + 1) / (propDatas.size + Ni))#使用了拉普拉斯平滑
                for nameFeatureProp, valP in propClassSummary.items():
                    eachClassPFeature[nameFeatureProp] = valP
                retModel[nameClass]['PFeature'][nameFeature] = eachClassPFeature
        return retModel

    def predictBySeries(self, data):
        curMaxRate = None
        curClassSelect = None
        for nameClass, infoModel in self.model.items():
            rate = 0
            rate += np.log(infoModel['PClass'])
            PFeature = infoModel['PFeature']

            for nameFeature, val in data.items():
                propsRate = PFeature.get(nameFeature)
                if not propsRate:
                    continue
                rate += np.log(propsRate.get(val, 0))#使用log加法避免很小的小数连续乘，接近零
                #print(nameFeature, val, propsRate.get(val, 0))
            #print(nameClass, rate)
            if curMaxRate == None or rate > curMaxRate:
                curMaxRate = rate
                curClassSelect = nameClass

        return curClassSelect
    def predict(self, data):
        if isinstance(data, pd.Series):
            return self.predictBySeries(data)
        return data.apply(lambda d: self.predictBySeries(d), axis=1)

# 读取iris.data文件
D2 = pd.read_csv('data/D2.csv', header=None)
# 添加列名
D2.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']

x = D2.iloc[:,:-1]
y = D2.iloc[:,-1]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=6666)

naiveBayes = NaiveBayes()
treeData = naiveBayes.fit(x_train,y_train)
y_predict = naiveBayes.predict(x_test)



y_pred = naiveBayes.predict(x_test)
cm = confusion_matrix(y_test, y_pred)
print(cm)
acc = accuracy_score(y_test, y_pred)
pre_micro = precision_score(y_test, y_pred, average='micro')
pre_macro = precision_score(y_test, y_pred, average='macro')
rec_micro = recall_score(y_test, y_pred, average='micro')
rec_macro = recall_score(y_test, y_pred, average='macro')
f1_micro = f1_score(y_test, y_pred, average='micro')
f1_macro = f1_score(y_test, y_pred, average='macro')

print("Accuracy: {:.2f}%".format(acc*100))
print("Precision (micro): {:.2f}%".format(pre_micro*100))
print("Precision (macro): {:.2f}%".format(pre_macro*100))
print("Recall (micro): {:.2f}%".format(rec_micro*100))
print("Recall (macro): {:.2f}%".format(rec_macro*100))
print("F1 score (micro): {:.2f}%".format(f1_micro*100))
print("F1 score (macro): {:.2f}%".format(f1_macro*100))







# 绘制混淆矩阵
fig, ax = plt.subplots()
im = ax.imshow(cm, cmap='Blues')
# 添加颜色条
cbar = ax.figure.colorbar(im, ax=ax)
# 设置标签和刻度
ax.set_xticks(range(len(y.unique() )))
ax.set_yticks(range(len(y.unique() )))
ax.set_xticklabels(np.unique(y))
ax.set_yticklabels(np.unique(y))
# 在每个方格内显示数值
thresh = cm.max() / 2
for i in range(len(y.unique() )):
    for j in range(len(y.unique() )):
        text = ax.text(j, i, cm[i, j], ha='center', va='center', color="white" if cm[i, j] > thresh else "black")
# 添加标题
ax.set_title('Confusion Matrix')
# 显示图形

plt.savefig('./img/bayes_iris_confusion_matrix.eps', format='eps', dpi=100, bbox_inches='tight')
plt.show()
plt.close()



# 绘制评价指标柱状图
x_labels = ['Accuracy', 'Precision', 'Recall', 'F1-measure']
micro_scores = [acc, pre_micro, rec_micro, f1_micro]
macro_scores = [acc, pre_macro, rec_macro, f1_macro]

x = np.arange(len(x_labels))
width = 0.35

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, micro_scores, width, label='Micro',color='#FF6666')
rects2 = ax.bar(x + width/2, macro_scores, width, label='Macro',color='#00CC99')

# 在柱子上方添加数值标签
for i, rect in enumerate(rects1):
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width() / 2., height,
            f'{micro_scores[i]:.3f}',
            ha='center', va='bottom', color='black')

for i, rect in enumerate(rects2):
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width() / 2., height,
            f'{macro_scores[i]:.3f}',
            ha='center', va='bottom', color='black')

ax.set_ylim(0.5, 0.9)
ax.set_ylabel('Scores')
ax.set_xticks(x)
ax.set_xticklabels(x_labels)
ax.legend()

fig.tight_layout()


plt.savefig('./img/bayes_iris_evaluation_metrics.eps', format='eps', dpi=1000, bbox_inches='tight')

plt.show()
plt.close()




#%%
