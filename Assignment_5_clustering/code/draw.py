import matplotlib.pyplot as plt


def draw_iris(x, labels, filename):
    # 定义颜色映射
    cmap = plt.cm.get_cmap('viridis')

    # 添加标题和坐标轴标签
    plt.title('Scatter Plot')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')

    # 绘制散点图
    plt.scatter(x[:, 0], x[:, 1], c=labels, cmap=cmap, s=30)  # 绘制散点图
    plt.savefig('./img/' + filename + '_cluster.eps', format='eps', dpi=500)
    
    
def draw_seed(x, labels, filename):
    # 定义颜色映射
    cmap = plt.cm.get_cmap('viridis')

    # 添加标题和坐标轴标签
    plt.title('Scatter Plot')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')

    # 绘制散点图
    plt.scatter(x[:, 0], x[:, 1], c=labels, cmap=cmap, s=30)  # 绘制散点图
    plt.savefig('./img/' + filename + '_cluster.eps', format='eps', dpi=500)
    
    
def draw_ecoli(x, labels, filename):
    # 定义颜色映射
    cmap = plt.cm.get_cmap('viridis')

    # 添加标题和坐标轴标签
    plt.title('Scatter Plot')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')

    # 绘制散点图
    plt.scatter(x[:, 0], x[:, 1], c=labels, cmap=cmap, s=30)  # 绘制散点图
    plt.savefig('./img/' + filename + '_cluster.eps', format='eps', dpi=500)

def plot_clustering_metrics(labels, NMI_values, RI_values, Purity_values,filename):
    # 绘制柱状图
    x = range(len(labels))
    width = 0.3

    fig, ax = plt.subplots()
    rects1 = ax.bar(x, NMI_values, width, label='NMI')
    rects2 = ax.bar([i + width for i in x], RI_values, width, label='RI')
    rects3 = ax.bar([i + 2 * width for i in x], Purity_values, width, label='Purity')

    # 设置图例和标签
    ax.set_ylabel('Score')
    ax.set_title(filename+'Clustering Evaluation Metrics')
    ax.set_xticks([i + width for i in x])
    ax.set_xticklabels(labels)
    ax.legend(loc='lower right')  # 将图例位置设置为右下角
    ax.set_ylim(0, 1.1)

    # 添加数据标签
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{:.2f}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)

    plt.tight_layout()
    plt.savefig('./img/' + filename + '_clustering_evaluation_metrics.eps', format='eps', dpi=500)

    plt.show()
