# Data Mining Practice

数据集：UCI dataset repository 

## Assignment_1——数据预处理(Data pre-processing)

- 认识数据：均值、方差、盒图
- 数据标准化：z_score、min-max、十进制小数定标和 logistic 方法
- 数据离散化：等频离散化、信息增益离散化、卡方离散化、CAIM 离散化方法
- 离群点检测： LOF 方法

## Assignment_2——回归(Regression)

- 一元多项式回归
- Ridge 回归或 Lasso 回归

## Assignment_3——分类(classification)

- 逻辑回归
- 决策树
- 朴素贝叶斯
- 神经网络
- SVM

## Assignment_4——集成学习(Ensemble Learning)

- 随机森林与 AdaBoost，分类
- 随机森林与 AdaBoost，回归

## Assignment_5——聚类(clustering)

- DPC
- K-Means
- DBSCAN
- SpectralClustering（谱聚类）
- EM算法（高斯混合模型）

# 实验课

## Experiment 1: 数据预处理

缺失值填补：

   1. 随机删除iris中的一些属性值（随机删除总量的5%）；

   2. 采用均值、中位数等缺失值填补：

结果比较实验：

   3. 对数据进行标准化处理，随机选择80%数据训练，20%数据测试，设置种子可重复实验；

   4. 使用KNN分类测试比较有缺失值和无缺失值的数据训练和测试结果；

实验代码可调用sklearn，结果需列表比较accuracy，precision，recall等。

## Experiment 2: 分类应用实践

根据给出数据进行分类预测，可以采用已有的各种分类方法来获得较高的accuracy、precious、recall、AUC、F1-score等。

实验结果中，以测试集的AUC值作为本次实验的分数。

给出测试集的验证代码，读入和train同样格式的数据后，自动进行预处理，然后进行模型测试和AUC等评价指标的显示。

数据集中部分属性需要预处理，最后一列为分类标签1和2。

## Experiment 3: 线性回归

1、设计一元线性回归完整代码（不使用sklearn等可直接调用的包） 

2、设计多元线性回归完整代码（根据公式计算，不使用sklearn等可直接调用的包） 注意：根据公式，逆矩阵存在的情况下，才可对矩阵求逆。 可以通过linalg.det()来计算行列式。

3、选做：岭回归实现  

## Experiment 4: logistic算法

1. 实现Logistic算法
2. 使用k-fold交叉验证进行实验（可调用工具包）
3. 在数据集Autistic Spectrum Disorder Screening Data for Children Data Set（https://archive.ics.uci.edu/ml/datasets/Autistic+Spectrum+Disorder+Screening+Data+for+Children++#）进行实验测试，并列出实验结果表，包括Accuracy，precision，recall等
4.  数据集需要必要的预处理（也可采用之前应用过的方法）

## Experiment 5: KNN

1、实现K近邻算法

2、采用K近邻对1个数据集进行分类分析，进行分类训练和测试，并给出分析结果

数据集可以采用阿尔及利亚森林火灾数据集：UCI Machine Learning Repository: Algerian Forest Fires Dataset Data Set

## Experiment 6: SMO

1、对已有SMO算法进行解读（添加注释，详细说明原理）

2、采用SMO对1个数据集（可采用上一个实验的数据）进行分类分析，进行分类训练和测试，并给出分析结果

## Experiment 7: 集成学习

1、实现Adaboost方法对2个数据集进行分类分析，进行分类训练和测试，并给出分析结果；

2、采用已掌握算法（决策树、朴素贝叶斯、KNN或SVM）对数据集进行分析，并和Adaboost算法给出实验结果比较；

3、说明sklearn中Adaboost的算法原理（选做，读源码，写代码注释和算法流程）

实验要求：实验报告中要包含存储结构说明，算法思路设计，运行时的输入输出截图，算法源码中添加详细注释，写明算法流程等

## Experiment 8: 聚类算法

K-Means算法的应用与实现

1、使用sklearn中的方法进行聚类实验，处理给定数据集（3选1，各国幸福指数，汽车型号，青少年市场细分）（根据数据集具体情况进行预处理和数据集划分）。

2、实现基础的k-means算法，并进行对应数据集测试，和sklearn中的实验结果进行比较，附实验结果数据比较（表）。

3、选做，尝试优化算法，进行聚类性能的改进。

## Experiment 9: 关联规则

调用已有方法进行关联分析实验，处理给定数据集（根据数据集具体情况进行预处理和数据集划分）。



