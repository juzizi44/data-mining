# 数据集（20 分）
- 从 UCI dataset repository 中下载以下数据集
  - （10 分）自行下载一个数据集，要求既包含连续的数值型属性，也包含离散的符号型属性（D1）；
  - （10 分）IRIS(D2), Wine Quality (red vinho verde wine) (D3), Breast Cancer Wisconsin (Diagnostic) Data Set (D4)。下载以后，仔细阅读数据集的使用说明，理解其用途及每一列数据的含义。

# 分类器的训练和测试（60 分）
- （15 分）逻辑回归：将 D4 按照 |Dtrain| : |Dtest = 80% : 20% 的比例进行划分，用 Dtrain 训练一个逻辑回归分类器，用 Dtest 测试其性能，评价指标可以用 accuracy、precision、recall 和F1-measure；
- （30 分）决策树、朴素贝叶斯：分别将 D1、D2 按照一定的比例划分为训练集 Dtrain 和测试集 Dtest（比例自行设定），用 Dtrain 分别训练一个决策树分类器（自选决策树算法）和一个朴素贝叶斯分类器，用 Dtest 测试其性能，评价指标分别用 accuracy、precision、recall 和F1-measure 的 macro、micro 版本；
- （15 分）神经网络、SVM：分别将 D2、D3 按一定的比例（自行设定）划分为训练集 Dtrain 和测试集 Dtest，用 Dtrain 分别训练神经网络（一个输入层、一个隐藏层、一个输出层，隐藏层神经元数目自行设定）和支持向量机模型（SVM 调用 sklearn 包中的实现即可），用 Dtest 测试训练所得分类器的性能。