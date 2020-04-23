from lvq import LgmlvqModel
from lvq import GmlvqModel
from lvq import GlvqModel
from lvq import plot2d
import seaborn as sn
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from operator import itemgetter
import matplotlib.pyplot as plt
from sklearn.utils import validation
from pylab import mpl
import numpy as np
import pandas as pd
import random
import dataP
import time

# 预备函数
"""===============================汉字转化函数==========================="""
def set_ch():
    mpl.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
    mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

set_ch()

"""===============================二维图函数==========================="""
def plot2(model, x, y, figure, title=""):
    """
    Projects the input data to two dimensions and plots it. The projection is
    done using the relevances of the given glvq model.

    Parameters
    ----------
    model : GlvqModel that has relevances
        (GrlvqModel,GmlvqModel,LgmlvqModel)
    x : array-like, shape = [n_samples, n_features]
        Input data
    y : array, shape = [n_samples]
        Input data target
    figure : int
        the figure to plot on
    title : str, optional
        the title to use, optional
    """
    x, y = validation.check_X_y(x, y)
    name = ['L1', 'L2', 'L3', 'L4', 'L5', 'L6']
    dim = 2
    f = plt.figure(figure)
    f.suptitle(title)
    nb_prototype = model.w_.shape[0]
    d = sorted([(model._compute_distance(x[y == model.c_w_[i]],
                                         model.w_[i]).sum(), i) for i in
                range(nb_prototype)], key=itemgetter(0))
    idxs = list(map(itemgetter(1), d))
    for i in idxs:
        x_p = model.project(x, i, dim, print_variance_covered=True)
        w_p = model.project(model.w_[i], i, dim)
        la = name[i]+'''_prototype'''
        ax = f.add_subplot(3, nb_prototype/3, idxs.index(i) + 1)
        ax.scatter(x_p[:, 0], x_p[:, 1], c=_to_tango_colors(y, 0),
                   alpha=0.2)
        # ax.scatter(X_p[:, 0], X_p[:, 1], c=pred, marker='.')
        ax.scatter(w_p[0], w_p[1],
                   c=_tango_color('aluminium', 5), marker='D', label=la)
        ax.legend()
        ax.scatter(w_p[0], w_p[1],
                   c=_tango_color(i, 0), marker='.', label=name[i])
        ax.legend()
        ax.axis('equal')
    f.show()


colors = {
    "skyblue": ['#729fcf', '#3465a4', '#204a87'],
    "scarletred": ['#ef2929', '#cc0000', '#a40000'],
    "orange": ['#fcaf3e', '#f57900', '#ce5c00'],
    "plum": ['#ad7fa8', '#75507b', '#5c3566'],
    "chameleon": ['#8ae234', '#73d216', '#4e9a06'],
    "butter": ['#fce94f', 'edd400', '#c4a000'],
    "chocolate": ['#e9b96e', '#c17d11', '#8f5902'],
    "aluminium": ['#eeeeec', '#d3d7cf', '#babdb6', '#888a85', '#555753',
                  '#2e3436']
}

color_names = list(colors.keys())


def _tango_color(name, brightness=0):
    if type(name) is int:
        if name >= len(color_names):
            name = name % len(color_names)
        name = color_names[name]
    if name in colors:
        return colors[name][brightness]
    else:
        raise ValueError('{} is not a valid color'.format(name))


def _to_tango_colors(elems, brightness=0):
    elem_set = list(set(elems))
    return [_tango_color(elem_set.index(e), brightness) for e in elems]

"""===============================混淆矩阵标记函数==========================="""
def lab(x):
    x = pd.DataFrame(x)
    x.columns = ['EXV1', 'EXV2', 'EXVleak', 'FWV1', 'FWV2', 'normal']
    x.index = ['EXV1', 'EXV2', 'EXVleak', 'FWV1', 'FWV2', 'normal']
    return x

"""===============================主程序==========================="""
random.seed(1)  # 设置随机种子值
# 读取并查询数据文件
fd = pd.read_excel('fd.xlsx', header=0)
data = pd.read_excel('cdata.xlsx', header=0)
data1 = pd.read_excel('alldata2.xlsx', header=0)
sdata = data
sdata1 = data1

# 分离特征和标签
col = sdata.shape[1]
# 随机森林获得特征重要程度
clf = RandomForestClassifier(max_features=None)
clf.fit(sdata.iloc[:, :-1], sdata.iloc[:, -1])
Re = clf.feature_importances_
fd = dataP.rankdata(Re, col, sdata, bound=0)  # bound为需要取的特征中要程度下边界
fd.index = ['位置', '重要度', '名称']
ax1 = plt.subplot()
plt.bar(x=fd.iloc[2, :], height=fd.iloc[1, :])
plt.xticks(rotation=90)
plt.ylabel('特征重要度')
plt.show()

# 循环查看特征数量对不同lvq模型影响
scorerank1 = pd.DataFrame(columns=['特征数', '测试准确率', '耗时'])  # GMLVQmodel
scorerank2 = pd.DataFrame(columns=['特征数', '测试准确率', '耗时'])  # GRLVQmodel

# 考察n值的影响
for n in range(2, fd.shape[1] + 1):
    nfdata = dataP.newdata(fd, sdata, n, col)  # 按特征重要度排序的新表
    # 分割得到训练集和测试集
    train, test = dataP.split(nfdata, 6, 0.25)
    train = np.array(train)
    test = np.array(test)

    # gmlvq
    time_start = time.time()  # 计时器开始
    lgmlvq = LgmlvqModel(prototypes_per_class=1, max_iter=130, random_state=223, display=True, gtol=0.0003)
    lgmlvq.fit(train[:, :len(train[0]) - 1], train[:, len(train[0]) - 1])
    pt = lgmlvq.predict(test[:, :len(test[0]) - 1])
    score = lgmlvq.score(test[:, :len(test[0]) - 1], test[:, len(test[0]) - 1])
    time_end = time.time()  # 计时器结束
    timecost = time_end - time_start
    # 创建结论表
    scorerank2 = scorerank2.append(pd.DataFrame({'特征数': [n], '测试准确率': [score], '耗时': [timecost]}), ignore_index=True)
    C6 = confusion_matrix(test[:, len(test[0]) - 1], pt)


C6.to_excel('E:\大四\毕业设计\算法\BPWLVQ\C1.xls', index=True)

plot2(lgmlvq, test[:, :len(test[0]) - 1], test[:, len(test[0]) - 1], 1, title=None)

ax2 = plt.subplot()
ax2.bar(x=list(range(len(scorerank2.iloc[:, 2]))), height=scorerank2.iloc[:, 2], color='#FF0000')
ax2.set_ylabel('运行总耗时')
ax3 = ax2.twinx()
ax3.plot(list(range(len(scorerank2.iloc[:, 2]))), scorerank2.iloc[:, 1], label='模型总正确率', color='black')
ax3.legend()
ax3.set_ylabel('模型总正确率')
plt.ylim()
plt.show()
scorerank1.to_excel('E:\大四\毕业设计\算法\BPWLVQ\dataresult\gmlvq.xls', index=True)

ndata = dataP.newdata(fd, sdata, 20, col)  # 查询特征表之后选择特征数量n
train, test = dataP.split(ndata, 6, 0.25)
train.to_excel('E:\大四\毕业设计\算法\BPWLVQ\\train.xls', index=True)
test.to_excel('E:\大四\毕业设计\算法\BPWLVQ\\test.xls', index=True)

for i in range(6):
    exec("o%s=lgmlvq.omegas_[%d]"%(i, i))
    exec("o%s=o%s.T.dot(o%s)"%(i, i, i))
    exec("o%s=pd.DataFrame(o%s)"%(i, i))
    exec("o%s.index=fd.iloc[2,0:19]"%(i))
    exec("o%s.columns=fd.iloc[2,0:19]" % (i))
    exec("v%s, u%s=np.linalg.eig(o%s)" % (i, i, i))
    exec("v%s=pd.Series(v%s)" % (i, i))
plt.bar(x=list(range(len(v0))), height=v0, width=0.8)
plt.bar(x=list(range(len(v1))), height=v1, width=0.8)
plt.bar(x=list(range(len(v2))), height=v2, width=0.8)
plt.bar(x=list(range(len(v3))), height=v3, width=0.8)
plt.bar(x=list(range(len(v4))), height=v4, width=0.8)
plt.bar(x=list(range(len(v5))), height=v5, width=0.8)
sn.heatmap(o0, annot=True, fmt='.0f', cmap='rainbow')
sn.heatmap(o1, annot=True, fmt='.0f', cmap='rainbow')
sn.heatmap(o2, annot=True, fmt='.0f', cmap='rainbow')
sn.heatmap(o3, annot=True, fmt='.0f', cmap='rainbow')
sn.heatmap(o4, annot=True, fmt='.0f', cmap='rainbow')
sn.heatmap(o5, annot=True, fmt='.0f', cmap='rainbow')