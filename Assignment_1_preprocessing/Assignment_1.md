# 数据集（10 分）
•（6 分）从 UCI dataset repository 中下载一个数据集，形成数据集 D1，满足以下要求：
– 包含至少 4 列以上连续的数值型数据，包含一列符号型数据，作为样本的类别标签；
– 至少包含 100 个以上的样本；
下载以后，仔细阅读数据集的使用说明，理解其用途及每一列数据的含义。
•（4 分）选定 D1 中的两列数值型数据，对其中的每一项数据添加大小不一的噪声，使其中出现离群值。提取这两列数据，将其存入文件，形成数据集 D2。

# 任务
编写程序，完成以下任务：

1. 认识数据（10 分）：对下载的数据进行分析，计算每一列数值型数据的均值、方差，画出该列数据的盒图；

2. 数据标准化（20 分）：分别用 z_score、min-max、十进制小数定标和 logistic 方法对数据集 D1 进行标准化处理，使所有列的数据处于同一规模，处理后的数据集记为 D1-zscore、D1-minmax、D1-float、D1-log；

3. 数据离散化（20 分）：对数据集 D1 的前 4 列分别使用等频离散化、信息增益离散化、卡方离散化、CAIM 离散化方法进行离散化，处理后的数据集记为 D1-discrete；

4. 离群点检测（20 分）：用 LOF 方法检测数据集 D2 中的离群点。

对于上述每一个任务，编程过程中可以使用 numpy、pandas、scikit-learn、scorecardbundle及 matplotlib 包中的相关功能。