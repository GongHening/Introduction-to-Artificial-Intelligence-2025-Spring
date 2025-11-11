from numpy.random import rand
import mnist
from answerTree import *
import numpy as np

# 超参数
# TODO: You can change the hyperparameters here
num_tree = 11    # 树的数量
ratio_data =0.47   # 采样的数据比例
ratio_feat = 0.33 # 采样的特征比例
hyperparams = {
    "depth":6, 
    "purity_bound":0.5,
    "gainfunc": negginiDA
    } # 每颗树的超参数


def buildtrees(X, Y):
    """
    构建随机森林
    @param X: n*d, 每行是一个输入样本。 n: 样本数量， d: 样本的维度
    @param Y: n, 样本的label
    @return: List of DecisionTrees, 随机森林
    """
    # TODO: YOUR CODE HERE
    # 提示：整体流程包括样本扰动、属性扰动和预测输出
    trees = []
    n_samples = X.shape[0]
    n_features = X.shape[1]
    n_sub_samples = max(1, int(n_samples * ratio_data))
    n_sub_features = max(1, int(n_features * ratio_feat))

    for _ in range(num_tree):
        sample_indices = np.random.choice(n_samples, n_sub_samples, replace=False)
        X_sub = X[sample_indices]
        Y_sub = Y[sample_indices]
        
        feat_indices = np.random.choice(n_features, n_sub_features, replace=False)
        X_sub_feat = X_sub[:, feat_indices]
        
        tree = buildTree(
            X=X_sub_feat,
            Y=Y_sub,
            unused=list(range(n_sub_features)),
            depth=hyperparams["depth"],
            purity_bound=hyperparams["purity_bound"],
            gainfunc=hyperparams["gainfunc"],
            prefixstr=""
        )
        
        def convert_feat_indices(node):
            if not node.isLeaf():
                node.featidx = feat_indices[node.featidx]
                for child in node.children.values():
                    convert_feat_indices(child)
        
        convert_feat_indices(tree)
        trees.append(tree)
    
    return trees
def infertrees(trees, X):
    """
    随机森林预测
    @param trees: 随机森林
    @param X: n*d, 每行是一个输入样本。 n: 样本数量， d: 样本的维度
    @return: n, 预测的label
    """
    pred = [inferTree(tree, X)  for tree in trees]
    pred = list(filter(lambda x: not np.isnan(x), pred))
    upred, ucnt = np.unique(pred, return_counts=True)
    return upred[np.argmax(ucnt)]
