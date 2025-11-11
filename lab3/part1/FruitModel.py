import math
from SST_2.dataset import traindataset, minitraindataset
from fruit import get_document, tokenize
import pickle
import numpy as np
from importlib.machinery import SourcelessFileLoader
from autograd.BaseGraph import Graph
from autograd.BaseNode import *

class NullModel:
    def __init__(self):
        pass

    def __call__(self, text):
        return 0


class NaiveBayesModel:
    def __init__(self):
        self.dataset = traindataset() # 完整训练集，需较长加载时间
        #self.dataset = minitraindataset() # 用来调试的小训练集，仅用于检查代码语法正确性

        # 以下内容可根据需要自行修改，不修改也可以完成本题
        self.token_num = [{}, {}] # token在正负样本中出现次数
        self.V = 0 #语料库token数量
        self.pos_neg_num = [0, 0] # 正负样本数量
        self.count()

    def count(self):
        # TODO: YOUR CODE HERE
        # 提示：统计token分布不需要返回值
        for text, label in self.dataset:
            self.pos_neg_num[label] += 1
            for token in text:
                if token not in self.token_num[label]:
                    self.token_num[label][token] = 1
                else:
                    self.token_num[label][token] += 1
        self.V = len(set(list(self.token_num[0].keys()) + list(self.token_num[1].keys())))

    def __call__(self, text):
        # TODO: YOUR CODE HERE
        # 返回1或0代表当前句子分类为正/负样本
        log_prob = [math.log(self.pos_neg_num[i] / sum(self.pos_neg_num)) for i in range(2)]
        for i in range(2):
            total = sum(self.token_num[i].values()) + self.V*2
            for token in text:
                count = self.token_num[i].get(token, 0)
                log_prob[i] += math.log((count + 1) / total)  # 拉普拉斯平滑
        if log_prob[1] > log_prob[0]:
            return 1
        else:
            return 0


def buildGraph(dim, num_classes, L): #dim: 输入一维向量长度, num_classes:分类数
    # 以下类均需要在BaseNode.py中实现
    # 也可自行修改模型结构
    nodes = [Attention(dim), 
             relu(), 
             LayerNorm((L, dim)), 
             ResLinear(dim), 
             relu(), 
             LayerNorm((L, dim)), 
             ResLinear(dim), 
             Mean(1), 
             Linear(dim, num_classes), 
             LogSoftmax(), 
             NLLLoss(num_classes)]
    
    graph = Graph(nodes)
    return graph


save_path = "model/attention.npy"

class Embedding():
    def __init__(self):
        self.emb = dict() 
        with open("words.txt", encoding='utf-8') as f: #word.txt存储了每个token对应的feature向量，self.emb是一个存储了token-feature键值对的Dict()，可直接调用使用
            for i in range(50000):
                row = next(f).split()
                word = row[0]
                vector = np.array([float(x) for x in row[1:]])
                self.emb[word] = vector
        
    def __call__(self, text, max_len=50):
        D = 100
        mat = np.zeros((max_len, D), dtype=np.float32)
        valid = 0
        for i, token in enumerate(text):
            if i >= max_len:
                break
            if token in self.emb:
                mat[i] = self.emb[token]
                valid += 1
        if valid == 0:
            # 如果所有词都不在字典中，则用随机或平均向量替代
            mat[:] = np.zeros(D)
        return mat



class AttentionModel():
    def __init__(self):
        self.embedding = Embedding()
        with open(save_path, "rb") as f:
            self.network = pickle.load(f)
        self.network.eval()
        self.network.flush()

    def __call__(self, text, max_len=50):
        X = self.embedding(text, max_len)
        X = np.expand_dims(X, 0)
        pred = self.network.forward(X, removelossnode=1)[-1]
        haty = np.argmax(pred, axis=-1)
        return haty[0]


class QAModel():
    def __init__(self):
        self.document_list = get_document()

    def tf(self, word, document):
        # TODO: YOUR CODE HERE
        # 返回单词在文档中的频度
        # document变量结构请参考fruit.py中get_document()函数
        count = document["document"].count(word)
        if len(document["document"]) > 0:
            return count / len(document["document"])  
        else:
            return 0

    def idf(self, word):
        # TODO: YOUR CODE HERE
        # 返回单词IDF值，提示：你需要利用self.document_list来遍历所有文档
        # 注意python整除与整数除法的区别
        doc_count = sum(1 for doc in self.document_list if word in doc["document"])
        return math.log(len(self.document_list) / (1 + doc_count)) 
    
    def tfidf(self, word, document):
        # TODO: YOUR CODE HERE
        # 返回TF-IDF值
        return self.tf(word, document) * self.idf(word)

    def __call__(self, query):
        query = tokenize(query) # 将问题token化
        # TODO: YOUR CODE HERE
        # 利用上述函数来实现QA
        # 提示：你需要根据TF-IDF值来选择一个最合适的文档，再根据IDF值选择最合适的句子
        # 返回时请返回原本句子，而不是token化后的句子，可以参考README中数据结构部分以及fruit.py中用于数据处理的get_document()函数

        # 找到最佳文档
        best_doc = None
        best_score = -1
        for doc in self.document_list:
            score = sum(self.tfidf(word, doc) for word in query)
            if score > best_score:
                best_score = score
                best_doc = doc

        # 在该文档中找出最相关句子
        best_sent = None
        best_sent_score = -1
        for sent in best_doc["sentences"]:
            score = sum(self.idf(word) for word in query if word in sent[0])
            if score > best_sent_score:
                best_sent_score = score
                best_sent = sent

        if best_sent==None:
            return ""
        else:
            return best_sent[1]

modeldict = {
    "Null": NullModel,
    "Naive": NaiveBayesModel,
    "Attn": AttentionModel,
    "QA": QAModel,
}


if __name__ == '__main__':
    embedding = Embedding()
    lr = 1e-3   # 学习率
    wd1 = 1e-5  # L1正则化
    wd2 = 1e-4  # L2正则化
    batchsize = 64
    max_epoch = 10
    
    max_L = 50
    num_classes = 2
    feature_D = 100
    
    graph = buildGraph(feature_D, num_classes, max_L) # 维度可以自行修改

    # 训练
    # 完整训练集训练有点慢
    best_train_acc = 0
    dataloader = traindataset(shuffle=True) # 完整训练集
    #dataloader = minitraindataset(shuffle=True) # 用来调试的小训练集
    for i in range(1, max_epoch+1):
        hatys = []
        ys = []
        losss = []
        graph.train()
        X = []
        Y = []
        cnt = 0
        for text, label in dataloader:
            x = embedding(text, max_L)
            label = np.zeros((1)).astype(np.int32) + label
            X.append(x)
            Y.append(label)
            cnt += 1
            if cnt == batchsize:
                X = np.stack(X, 0)
                Y = np.concatenate(Y, 0)
                graph[-1].y = Y
                graph.flush()
                pred, loss = graph.forward(X)[-2:]
                hatys.append(np.argmax(pred, axis=-1))
                ys.append(Y)
                graph.backward()
                graph.optimstep(lr, wd1, wd2)
                losss.append(loss)
                cnt = 0
                X = []
                Y = []

        loss = np.average(losss)
        acc = np.average(np.concatenate(hatys)==np.concatenate(ys))
        print(f"epoch {i} loss {loss:.3e} acc {acc:.4f}")
        if acc > best_train_acc:
            best_train_acc = acc
            with open(save_path, "wb") as f:
                pickle.dump(graph, f)