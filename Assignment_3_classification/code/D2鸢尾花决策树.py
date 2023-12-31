import numpy as np
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score


class Node:
    '''
    Helper class which implements a single tree node.
    '''
    def __init__(self, feature=None, threshold=None, data_left=None, data_right=None, gain=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.data_left = data_left
        self.data_right = data_right
        self.gain = gain
        self.value = value

class DecisionTree:
    '''
    Class which implements a decision tree classifier algorithm.
    '''
    def __init__(self, min_samples_split=2, max_depth=5):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.root = None

    @staticmethod
    def _entropy(s):
        '''
        Helper function, calculates entropy from an array of integer values.

        :param s: list
        :return: float, entropy value
        '''
        # Convert to integers to avoid runtime errors
        counts = np.bincount(np.array(s, dtype=np.int64))
        # Probabilities of each class label
        percentages = counts / len(s)

        # Caclulate entropy
        entropy = 0
        for pct in percentages:
            if pct > 0:
                entropy += pct * np.log2(pct)
        return -entropy

    def _information_gain(self, parent, left_child, right_child):
        '''
        Helper function, calculates information gain from a parent and two child nodes.

        :param parent: list, the parent node
        :param left_child: list, left child of a parent
        :param right_child: list, right child of a parent
        :return: float, information gain
        '''
        num_left = len(left_child) / len(parent)
        num_right = len(right_child) / len(parent)

        # One-liner which implements the previously discussed formula
        return self._entropy(parent) - (num_left * self._entropy(left_child) + num_right * self._entropy(right_child))

    def _best_split(self, X, y):
        '''
        Helper function, calculates the best split for given features and target

        :param X: np.array, features
        :param y: np.array or list, target
        :return: dict
        '''
        best_split = {}
        best_info_gain = -1
        n_rows, n_cols = X.shape

        # For every dataset feature
        for f_idx in range(n_cols):
            X_curr = X[:, f_idx]
            # For every unique value of that feature
            for threshold in np.unique(X_curr):
                # Construct a dataset and split it to the left and right parts
                # Left part includes records lower or equal to the threshold
                # Right part includes records higher than the threshold
                df = np.concatenate((X, y.reshape(1, -1).T), axis=1)
                df_left = np.array([row for row in df if row[f_idx] <= threshold])
                df_right = np.array([row for row in df if row[f_idx] > threshold])

                # Do the calculation only if there's data in both subsets
                if len(df_left) > 0 and len(df_right) > 0:
                    # Obtain the value of the target variable for subsets
                    y = df[:, -1]
                    y_left = df_left[:, -1]
                    y_right = df_right[:, -1]

                    # Caclulate the information gain and save the split parameters
                    # if the current split if better then the previous best
                    gain = self._information_gain(y, y_left, y_right)
                    if gain > best_info_gain:
                        best_split = {
                            'feature_index': f_idx,
                            'threshold': threshold,
                            'df_left': df_left,
                            'df_right': df_right,
                            'gain': gain
                        }
                        best_info_gain = gain
        return best_split

    def _build(self, X, y, depth=0):
        '''
        Helper recursive function, used to build a decision tree from the input data.

        :param X: np.array, features
        :param y: np.array or list, target
        :param depth: current depth of a tree, used as a stopping criteria
        :return: Node
        '''
        n_rows, n_cols = X.shape

        # Check to see if a node should be leaf node
        if n_rows >= self.min_samples_split and depth <= self.max_depth:
            # Get the best split
            best = self._best_split(X, y)
            # If the split isn't pure
            if best['gain'] > 0:
                # Build a tree on the left
                left = self._build(
                    X=best['df_left'][:, :-1],
                    y=best['df_left'][:, -1],
                    depth=depth + 1
                )
                right = self._build(
                    X=best['df_right'][:, :-1],
                    y=best['df_right'][:, -1],
                    depth=depth + 1
                )
                return Node(
                    feature=best['feature_index'],
                    threshold=best['threshold'],
                    data_left=left,
                    data_right=right,
                    gain=best['gain']
                )
        # Leaf node - value is the most common target value
        return Node(
            value=Counter(y).most_common(1)[0][0]
        )

    def fit(self, X, y):
        '''
        Function used to train a decision tree classifier model.

        :param X: np.array, features
        :param y: np.array or list, target
        :return: None
        '''
        # Call a recursive function to build the tree
        self.root = self._build(X, y)

    def _predict(self, x, tree):
        '''
        Helper recursive function, used to predict a single instance (tree traversal).

        :param x: single observation
        :param tree: built tree
        :return: float, predicted class
        '''
        # Leaf node
        if tree.value != None:
            return tree.value
        feature_value = x[tree.feature]

        # Go to the left
        if feature_value <= tree.threshold:
            return self._predict(x=x, tree=tree.data_left)

        # Go to the right
        if feature_value > tree.threshold:
            return self._predict(x=x, tree=tree.data_right)

    def predict(self, X):
        '''
        Function used to classify new instances.

        :param X: np.array, features
        :return: np.array, predicted classes
        '''
        # Call the _predict() function for every observation
        return [self._predict(x, self.root) for x in X]
# 读取iris.data文件
D2 = pd.read_csv('data/D2.csv', header=None)
# 添加列名
D2.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']
X = D2.iloc[:,:-1].values
y = D2.iloc[:,-1].values
le = LabelEncoder()
y = le.fit_transform(y)


x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=6666)
dt = DecisionTree()
dt.fit(x_train,y_train)
y_pred = dt.predict(x_test)
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
y = D2.iloc[:, -1]
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

plt.savefig('./img/decisiontree_iris_confusion_matrix.eps', format='eps', dpi=100, bbox_inches='tight')
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

ax.set_ylim(0.6, 1.08)
ax.set_ylabel('Scores')
ax.set_xticks(x)
ax.set_xticklabels(x_labels)
ax.legend()

fig.tight_layout()


plt.savefig('./img/decisiontree_iris_evaluation_metrics.eps', format='eps', dpi=1000, bbox_inches='tight')
plt.show()
plt.close()