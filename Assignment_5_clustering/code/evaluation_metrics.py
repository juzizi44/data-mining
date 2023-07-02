
from sklearn import metrics

def metr(true_labels,predicted_labels):
 
    # 计算NMI（标准化互信息）
    nmi = metrics.normalized_mutual_info_score(true_labels, predicted_labels)
    # print("NMI:", nmi)

    # 计算RI（兰德指数）
    ri = metrics.adjusted_rand_score(true_labels, predicted_labels)

    # 计算Purity（纯度）
    purity = metrics.cluster.contingency_matrix(true_labels, predicted_labels)
    total_samples = sum(sum(purity))
    max_purity = sum(purity.max(axis=0)) / total_samples
    purity_score = sum(purity.max(axis=1)) / total_samples


    return nmi,ri,purity_score
    


