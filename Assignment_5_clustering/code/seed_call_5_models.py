from sklearn import preprocessing
import pandas as pd
import draw


# 数据预处理
data = pd.read_csv('data/seeds_dataset.txt', header=None,sep='\s+')
        
# 训练集和测试集划分
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values
file_name='seed_real'
draw.draw_seed(X,y,file_name)



# ==============================================调用自己写的DPC=========================================================
import DPC # 自己写的DPC
import evaluation_metrics  # 自己写的计算指标（NMI\RI\Purity）

file_name = "seed_DPC"

# 定义簇标签对应的颜色字典
dic_colors = {0: (.8, 0, 0), 1: (0, .8, 0),
                2: (0, 0, .8), 3: (.8, .8, 0),
                4: (.8, 0, .8), 5: (0, .8, .8),
                7: (0, 0, 0)}
 
datas = data.iloc[:, :-1].values

dists = DPC.getDistanceMatrix(datas)  # 计算距离矩阵
dc = DPC.select_dc(dists)  # 选择局部密度阈值
print("number", dc)
rho = DPC.get_density(dists, dc, method="Gaussian")  # 计算局部密度
deltas, nearest_neiber = DPC.get_deltas(dists, rho)  # 计算相对密度和最近邻数据点
DPC.draw_decision(datas,rho, deltas, name=file_name+"_decision")  # 绘制决策图
centers = DPC.find_centers_K(rho, deltas, 3)  # 找到聚类中心点
print("cluster-centers", centers)
labs = DPC.cluster_PD(rho, centers, nearest_neiber)  # 进行聚类
DPC.draw_cluster(datas, labs, centers, dic_colors, name=file_name+"_cluster")  # 绘制聚类结果
true_labels = data.iloc[:,-1]
predicted_labels = labs
seed_DPC_NMI,seed_DPC_RI ,seed_DPC_Purity  = evaluation_metrics.metr(true_labels,predicted_labels)
print("seed_DPC_NMI:", seed_DPC_NMI)
print("seed_DPC_RI:", seed_DPC_RI)
print("seed_DPC_Purity:", seed_DPC_Purity)



# ==============================================调用K-means=========================================================
from sklearn.cluster import KMeans
import draw  # 自己写的绘图函数
import evaluation_metrics  # 自己写的计算指标（NMI\RI\Purity）


# 训练集和测试集划分
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

k = 3  # 聚类的簇数
kmeans = KMeans(n_clusters=k)
kmeans.fit(X)
labels = kmeans.labels_  # 聚类结果的标签

seed_kmeans_NMI,seed_kmeans_RI ,seed_kmeans_Purity  = evaluation_metrics.metr(y,labels)
print("seed_kmeans_NMI:", seed_kmeans_NMI)
print("seed_kmeans_RI:", seed_kmeans_RI)
print("seed_kmeans_Purity:", seed_kmeans_Purity)

file_name='seed_kmeans'
draw.draw_seed(X,labels,file_name)

# ==============================================调用DBSCAN=========================================================
from sklearn.cluster import DBSCAN
import draw  # 自己写的绘图函数
import evaluation_metrics  # 自己写的计算指标（NMI\RI\Purity）

# 加载数据
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# DBSCAN聚类
dbscan = DBSCAN(eps=0.5, min_samples=5)
labels = dbscan.fit_predict(X)

# 计算指标
seed_dbscan_NMI, seed_dbscan_RI, seed_dbscan_Purity = evaluation_metrics.metr(y, labels)
print("seed_dbscan_NMI:", seed_dbscan_NMI)
print("seed_dbscan_RI:", seed_dbscan_RI)
print("seed_dbscan_Purity:", seed_dbscan_Purity)

# 绘制聚类结果
file_name = 'seed_dbscan'
draw.draw_seed(X, labels, file_name)

# ==============================================调用谱聚类=========================================================

from sklearn.cluster import SpectralClustering
import draw  # 自己写的绘图函数
import evaluation_metrics  # 自己写的计算指标（NMI\RI\Purity）

# 加载数据
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# SpectralClustering聚类
spectral_clustering = SpectralClustering(n_clusters=3)
labels = spectral_clustering.fit_predict(X)

# 计算指标
seed_spectral_NMI, seed_spectral_RI, seed_spectral_Purity = evaluation_metrics.metr(y, labels)
print("seed_spectral_NMI:", seed_spectral_NMI)
print("seed_spectral_RI:", seed_spectral_RI)
print("seed_spectral_Purity:", seed_spectral_Purity)

# 绘制聚类结果
file_name = 'seed_spectral'
draw.draw_seed(X, labels, file_name)


# ==============================================调用EM算法高斯混合模型=========================================================

from sklearn.mixture import GaussianMixture
import draw  # 自己写的绘图函数
import evaluation_metrics  # 自己写的计算指标（NMI\RI\Purity）

# 加载数据
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# EM算法聚类（高斯混合模型）
em = GaussianMixture(n_components=3)
em.fit(X)
labels = em.predict(X)

# 计算指标
seed_em_NMI, seed_em_RI, seed_em_Purity = evaluation_metrics.metr(y, labels)
print("seed_em_NMI:", seed_em_NMI)
print("seed_em_RI:", seed_em_RI)
print("seed_em_Purity:", seed_em_Purity)

# 绘制聚类结果
file_name = 'seed_em'
draw.draw_seed(X, labels, file_name)

# ==============================================指标总结图=========================================================
import draw
file_name='seed'
labels = ['K-means', 'DBSCAN', 'Spectral', 'EM', 'DPC']
NMI_values = [seed_kmeans_NMI, seed_dbscan_NMI, seed_spectral_NMI, seed_em_NMI, seed_DPC_NMI]
RI_values = [seed_kmeans_RI, seed_dbscan_RI, seed_spectral_RI, seed_em_RI, seed_DPC_RI]
Purity_values = [seed_kmeans_Purity, seed_dbscan_Purity, seed_spectral_Purity, seed_em_Purity, seed_DPC_Purity]

draw.plot_clustering_metrics(labels, NMI_values, RI_values, Purity_values,file_name)


