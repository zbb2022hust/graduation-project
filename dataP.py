from sklearn.model_selection import train_test_split
import ts_anomaly_detection as ad
import pandas as pd
import numpy as np
'''
读取数据信息，合成特征排序表，分配训练数据集和测试数据集,标签编码, 滤波算法，时序突变值的滑窗方差选取
'''
# 将数据分割为特征和标签
def separateData(data):
    dt = np.array(data)
    col = len(dt[0])
    cl = data.iloc[:, col-1]
    fe = data.iloc[:, :col-1]
    return col, cl, fe

# 根据特征重要度表并和所需特征重要度下边界将特征重新排序
def rankdata(Re, col, data, bound):
    f = np.array(range(col-1))
    fr = np.vstack((f, Re))
    fd = fr.T[np.lexsort(-fr[1, None])].T
    label = np.zeros(shape=(1, col-1), dtype=object)
    for i in range(col-1):
        label[0, i] = data.columns[fd[0, i]]
    fd = pd.DataFrame(fd)
    label = pd.DataFrame(label)
    fd = pd.concat([fd, label], axis=0)
    for i in fd.columns:
        if fd.loc[1, i] < bound:
            fd = fd.drop(i, axis=1)
    return fd

# 选择标签数n以及更新数据表
def newdata(fd, data, n, col):
    fdata = pd.DataFrame()
    for i in range(n):
        fdata[fd.iloc[2, i]] = data.loc[:, fd.iloc[2, i]]
    fdata[data.columns[col-1]] = data.iloc[:, col-1]
    fdata = cleanbool(fdata)
    return fdata

# 分配训练集和测试集
def split(data, n, m):  # n为随机数；m为分割比例
    lab = np.array(data['故障类别'].value_counts().index)
    p1 = data[data['故障类别'] == lab[0]]
    p2 = data[data['故障类别'] == lab[1]]
    p3 = data[data['故障类别'] == lab[2]]
    p4 = data[data['故障类别'] == lab[3]]
    p5 = data[data['故障类别'] == lab[4]]
    p6 = data[data['故障类别'] == lab[5]]
    train1, test1 = train_test_split(p1, test_size=m, random_state=n)
    train2, test2 = train_test_split(p2, test_size=m, random_state=n)
    train3, test3 = train_test_split(p3, test_size=m, random_state=n)
    train4, test4 = train_test_split(p4, test_size=m, random_state=n)
    train5, test5 = train_test_split(p5, test_size=m, random_state=n)
    train6, test6 = train_test_split(p6, test_size=m, random_state=n)
    train = pd.concat([train1, train2, train3, train4, train5, train6], axis=0, ignore_index=True)
    test = pd.concat([test1, test2, test3, test4, test5, test6], axis=0, ignore_index=True)
    return train, test

# 将逻辑值赋值为0或1结构
def cleanbool(data):
    for u in data.columns:
        if data[u].dtype == bool:
            data[u] = data[u].astype('int')
    return data

# Series归一化
def normalization(data):
    mid = np.linspace(0, 1, len(data))
    for i in range(len(data)):
        mid[i] = float(data[i] - np.min(data))/(np.max(data) - np.min(data))
    mid = pd.Series(mid)
    return mid

# EWMA指数加权移动平均滤波法
def EWMA(S, a=0.05):  # S为Series数据；a为权重系数
    nS = np.linspace(0, 1, len(S))
    S.reset_index(drop=True, inplace=True)
    for i in range(len(S)):
        if i <= 0:
            nS[i] = S[i]
        else:
            nS[i] = (1-a)*nS[i-1] + a*S[i]
    nS = pd.Series(nS)
    return nS

# 计算方差
def CV(s):  #s为数组或series
    mean = np.mean(s)
    std = np.std(s, ddof=0)
    CV = std/mean
    return CV

# 计算预测正确率
def accuracy(predict_values, actual):
    correct = 0
    for i in range(len(predict_values)):
        if actual[i] == predict_values[i]:
            correct += 1
    return correct / float(len(predict_values))

#  时序数据突变值选取
def abnormalclean(S1, S2, threshold, N=None):
    # N为滑窗大小；threshold为稳态阈值；S1为滤波后数据，S2为原数据(Series型)；
    ab = slide(S1, S2, N, threshold)  # 滑窗算法获得异常值位置和位置上的数值
    ab.set_index(ab.iloc[:, 0], inplace=True)  # 将异常值位置放置在行标签上
    #  模块高压时序数据突变值与正常值标识图
    S2.plot(color='#000000')
    start = 0
    for i in range(len(ab)):
        if i == len(ab)-1:
            ab.iloc[start:i, 1].plot(color='#FF0000')
            break
        elif ab.index[i]+1 != ab.index[i+1]:
            ab.iloc[start:i, 1].plot(color='#FF0000')
            start = i+1
    return ab

# 滑窗方差法对突变值选取
def slide(S1, S2, N, threshold):  # N为滑窗大小；threshold为稳态阈值；S1滤波数据,S2为原数据，Series型数据
    left = 0
    right = N
    length = len(S1)
    num = length - N
    unsteady = pd.DataFrame(dtype='float64')
    loc = pd.Series(dtype='float64')
    real = pd.Series(dtype='float64')
    stddata = np.linspace(0, 1, num)
    for i in range(num):
        wd = S1[left:right]
        wd2 = S2[left:right]
        CV1 = CV(wd)
        if CV1 > threshold:
            loc = loc.append(pd.Series(np.array(wd.index)[N-1]))
            real = real.append(pd.Series(np.array(wd2)[N-1]))
        left += 1
        right += 1
    unsteady = pd.concat([unsteady, loc, real], axis=1, ignore_index=True)
    return unsteady

def abnormal(x, freq, alpha, ub, hybrid): # S-H-ESD时序异常值检测
    ab = ad.esd_test(x, freq, alpha, ub, hybrid)
    return ab