import draw
from sklearn import preprocessing
import pandas as pd

data = pd.read_csv('data/ecoli.data', header=None,sep='\s+')
# 将字符类型转化为数值类型
le = preprocessing.LabelEncoder()
for col in data.columns:
    if data[col].dtype == 'object':
        data[col] = le.fit_transform(data[col])
data=data.iloc[:, 1:]

   

X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values
file_name='ecoli_real'
draw.draw_ecoli(X,y,file_name)


# ==============================================调用自己写的DPC=========================================================
import DPC # 自己写的DPC
import evaluation_metrics  # 自己写的计算指标（NMI\RI\Purity）

file_name = "ecoli_DPC"

# 定义簇标签对应的颜色字典
dic_colors = {0: (.8, 0, 0), 1: (0, .8, 0),
                2: (0, 0, .8), 3: (.8, .8, 0),
                4: (.8, 0, .8), 5: (0, .8, .8),
                6: (0, 0, 0),7:(.5, .5, .5),8:(.2, .5, .7)}
 
datas = data.iloc[:, :-1].values

dists = DPC.getDistanceMatrix(datas)  # 计算距离矩阵
dc = DPC.select_dc(dists)  # 选择局部密度阈值
print("number", dc)
rho = DPC.get_density(dists, dc, method="Gaussian")  # 计算局部密度
deltas, nearest_neiber = DPC.get_deltas(dists, rho)  # 计算相对密度和最近邻数据点
DPC.draw_decision(datas,rho, deltas, name=file_name+"_decision")  # 绘制决策图
centers = DPC.find_centers_K(rho, deltas, 8)  # 找到聚类中心点
print("cluster-centers", centers)
labs = DPC.cluster_PD(rho, centers, nearest_neiber)  # 进行聚类
DPC.draw_cluster(datas, labs, centers, dic_colors, name=file_name+"_cluster")  # 绘制聚类结果
true_labels = data.iloc[:,-1]
predicted_labels = labs
ecoli_DPC_NMI,ecoli_DPC_RI ,ecoli_DPC_Purity  = evaluation_metrics.metr(true_labels,predicted_labels)
print("ecoli_DPC_NMI:", ecoli_DPC_NMI)
print("ecoli_DPC_RI:", ecoli_DPC_RI)
print("ecoli_DPC_Purity:", ecoli_DPC_Purity)



# ==============================================调用K-means=========================================================
from sklearn.cluster import KMeans
import draw  # 自己写的绘图函数
import evaluation_metrics  # 自己写的计算指标（NMI\RI\Purity）


# 训练集和测试集划分
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

k = 8  # 聚类的簇数
kmeans = KMeans(n_clusters=k)
kmeans.fit(X)
labels = kmeans.labels_  # 聚类结果的标签

ecoli_kmeans_NMI,ecoli_kmeans_RI ,ecoli_kmeans_Purity  = evaluation_metrics.metr(y,labels)
print("ecoli_kmeans_NMI:", ecoli_kmeans_NMI)
print("ecoli_kmeans_RI:", ecoli_kmeans_RI)
print("ecoli_kmeans_Purity:", ecoli_kmeans_Purity)

file_name='ecoli_kmeans'
draw.draw_ecoli(X,labels,file_name)

# ==============================================调用DBSCAN=========================================================
from sklearn.cluster import DBSCAN
import draw  # 自己写的绘图函数
import evaluation_metrics  # 自己写的计算指标（NMI\RI\Purity）

# 加载数据
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# DBSCAN聚类
dbscan = DBSCAN(eps=0.5, min_samples=10)
labels = dbscan.fit_predict(X)

# 计算指标
ecoli_dbscan_NMI, ecoli_dbscan_RI, ecoli_dbscan_Purity = evaluation_metrics.metr(y, labels)
print("ecoli_dbscan_NMI:", ecoli_dbscan_NMI)
print("ecoli_dbscan_RI:", ecoli_dbscan_RI)
print("ecoli_dbscan_Purity:", ecoli_dbscan_Purity)

# 绘制聚类结果
file_name = 'ecoli_dbscan'
draw.draw_ecoli(X, labels, file_name)

# ==============================================调用谱聚类=========================================================

from sklearn.cluster import SpectralClustering
import draw  # 自己写的绘图函数
import evaluation_metrics  # 自己写的计算指标（NMI\RI\Purity）

# 加载数据
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# SpectralClustering聚类
spectral_clustering = SpectralClustering(n_clusters=8)
labels = spectral_clustering.fit_predict(X)

# 计算指标
ecoli_spectral_NMI, ecoli_spectral_RI, ecoli_spectral_Purity = evaluation_metrics.metr(y, labels)
print("ecoli_spectral_NMI:", ecoli_spectral_NMI)
print("ecoli_spectral_RI:", ecoli_spectral_RI)
print("ecoli_spectral_Purity:", ecoli_spectral_Purity)

# 绘制聚类结果
file_name = 'ecoli_spectral'
draw.draw_ecoli(X, labels, file_name)


# ==============================================调用EM算法高斯混合模型=========================================================

from sklearn.mixture import GaussianMixture
import draw  # 自己写的绘图函数
import evaluation_metrics  # 自己写的计算指标（NMI\RI\Purity）

# 加载数据
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# EM算法聚类（高斯混合模型）
em = GaussianMixture(n_components=8)
em.fit(X)
labels = em.predict(X)

# 计算指标
ecoli_em_NMI, ecoli_em_RI, ecoli_em_Purity = evaluation_metrics.metr(y, labels)
print("ecoli_em_NMI:", ecoli_em_NMI)
print("ecoli_em_RI:", ecoli_em_RI)
print("ecoli_em_Purity:", ecoli_em_Purity)

# 绘制聚类结果
file_name = 'ecoli_em'
draw.draw_ecoli(X, labels, file_name)

# ==============================================指标总结图=========================================================
import draw
file_name='ecoli'
labels = ['K-means', 'DBSCAN', 'Spectral', 'EM', 'DPC']
NMI_values = [ecoli_kmeans_NMI, ecoli_dbscan_NMI, ecoli_spectral_NMI, ecoli_em_NMI, ecoli_DPC_NMI]
RI_values = [ecoli_kmeans_RI, ecoli_dbscan_RI, ecoli_spectral_RI, ecoli_em_RI, ecoli_DPC_RI]
Purity_values = [ecoli_kmeans_Purity, ecoli_dbscan_Purity, ecoli_spectral_Purity, ecoli_em_Purity, ecoli_DPC_Purity]

draw.plot_clustering_metrics(labels, NMI_values, RI_values, Purity_values,file_name)
