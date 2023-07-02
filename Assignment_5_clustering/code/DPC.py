import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict

 
def getDistCut(distList,distPercent):
    return max(distList) * distPercent / 100
 
def getRho(n,distMatrix,distCut):
    rho = np.zeros(n,dtype=float)
    for i in range(n-1):
        for j in range(i+1,n):
            rho[i] = rho[i] + np.exp(-(distMatrix[i, j] / distCut) ** 2)
            rho[j] = rho[j] + np.exp(-(distMatrix[i, j] / distCut) ** 2)
    return rho


def DPCA(n,distMatrix,rho,blockNum):
    rhoOrdIndex = np.flipud(np.argsort(rho))
    delta = np.zeros(n,dtype=float)
    leader = np.ones(n,dtype=int) * int(-1)
    maxdist = 0
    for ele in range(n):
        if distMatrix[rhoOrdIndex[0],ele] > maxdist:
            maxdist = distMatrix[rhoOrdIndex[0],ele]
    delta[rhoOrdIndex[0]] = maxdist
    for i in range(1,n):
        mindist = np.inf
        minindex = -1
        for j in range(i):
            if distMatrix[rhoOrdIndex[i],rhoOrdIndex[j]] < mindist:
                mindist = distMatrix[rhoOrdIndex[i],rhoOrdIndex[j]]
                minindex = rhoOrdIndex[j]
        delta[rhoOrdIndex[i]] = mindist
        leader[rhoOrdIndex[i]] = minindex
    gamma = delta * rho
    gammaOrdIdx = np.flipud(np.argsort(gamma))
    clusterIdx = np.ones(n,dtype=int) * (-1)
    for k in range(blockNum):
        clusterIdx[gammaOrdIdx[k]] = k
    for i in range(n):
        if clusterIdx[rhoOrdIndex[i]] == -1:
            clusterIdx[rhoOrdIndex[i]] = clusterIdx[leader[rhoOrdIndex[i]]]
    clusterSet = OrderedDict()
    for k in range(blockNum):
        clusterSet[k] = []
    for i in range(n):
        clusterSet[clusterIdx[i]].append(i)
    return clusterSet


def getDistanceMatrix(datas):
    N,D = np.shape(datas)
    dists = np.zeros([N,N])
    
    for i in range(N):
        for j in range(N):
            vi = datas[i,:]
            vj = datas[j,:]
            dists[i,j]= np.sqrt(np.dot((vi-vj),(vi-vj)))
    return dists

def select_dc(dists):    
    N = np.shape(dists)[0]
    tt = np.reshape(dists,N*N)
    percent = 2.0
    position = int(N * (N - 1) * percent / 100)
    dc = np.sort(tt)[position  + N]
    
    return dc
    
def get_density(dists,dc,method=None):
    N = np.shape(dists)[0]
    rho = np.zeros(N)
    
    for i in range(N):
        if method == None:
            rho[i]  = np.where(dists[i,:]<dc)[0].shape[0]-1
        else:
            rho[i] = np.sum(np.exp(-(dists[i,:]/dc)**2))-1
    return rho
    

def get_deltas(dists,rho):
    N = np.shape(dists)[0]
    deltas = np.zeros(N)
    nearest_neiber = np.zeros(N)

    index_rho = np.argsort(-rho)
    for i,index in enumerate(index_rho):

        if i==0:
            continue
  
        index_higher_rho = index_rho[:i]
 
        deltas[index] = np.min(dists[index,index_higher_rho])
        
        index_nn = np.argmin(dists[index,index_higher_rho])
        nearest_neiber[index] = index_higher_rho[index_nn].astype(int)
    
    deltas[index_rho[0]] = np.max(deltas)   
    return deltas,nearest_neiber
        

def find_centers_auto(rho,deltas):
    rho_threshold = (np.min(rho) + np.max(rho))/ 2
    delta_threshold  = (np.min(deltas) + np.max(deltas))/ 2
    N = np.shape(rho)[0]
    
    centers = []
    for i in range(N):
        if rho[i]>=rho_threshold and deltas[i]>delta_threshold:
            centers.append(i)
    return np.array(centers)

  
def find_centers_K(rho,deltas,K):
    rho_delta = rho*deltas
    centers = np.argsort(-rho_delta)
    return centers[:K]


def cluster_PD(rho,centers,nearest_neiber):
    K = np.shape(centers)[0]
    if K == 0:
        print("can not find centers")
        return
    
    N = np.shape(rho)[0]
    labs = -1*np.ones(N).astype(int)
    

    for i, center in enumerate(centers):
        labs[center] = i
   

    index_rho = np.argsort(-rho)
    for i, index in enumerate(index_rho):

        if labs[index] == -1:
            labs[index] = labs[int(nearest_neiber[index])]
    return labs
        
def draw_decision(datas,rho,deltas,name="1"):       
    plt.cla()
    # 添加标题和坐标轴标签
    plt.title('Scatter Plot')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    for i in range(np.shape(datas)[0]):
        plt.scatter(rho[i],deltas[i],s=16.,color=(0,0,0))
        plt.annotate(str(i), xy = (rho[i], deltas[i]),xytext = (rho[i], deltas[i]))
        plt.xlabel("rho")
        plt.ylabel("deltas")
   
    plt.savefig('./img/' + name + '.eps', format='eps', dpi=500)
    plt.close()
    # plt.show()

def draw_cluster(datas,labs,centers, dic_colors, name="1"):     
    plt.cla()
    # 添加标题和坐标轴标签
    plt.title('Scatter Plot')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    K = np.shape(centers)[0]
    for k in range(K):
        sub_index = np.where(labs == k)
        sub_datas = datas[sub_index]
        plt.scatter(sub_datas[:,0],sub_datas[:,1],s=16.,color=dic_colors[k])
        plt.scatter(datas[centers[k],0],datas[centers[k],1],color="k",marker="+",s = 200.)
    plt.savefig('./img/' + name + '.eps', format='eps', dpi=500)
    # plt.show()
    plt.close()
