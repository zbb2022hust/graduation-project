import multiprocessing as mp
from multiprocessing import Pool as ProcessPool
from multiprocessing.dummy import Pool as ThreadPool
from lvq import LgmlvqModel
from sklearn_lvq.utils import plot2d
import readdata as rd
import geatpy.Problem as ea
import numpy as np
import time

class gawlvq(ea.Problem):  # 继承Problem父类
    def __init__(self, PoolType):
        name = 'ga-wlvq'
        M = 2  # 初始化M（目标维数）
        maxormins = [-1, 1]  # 初始化maxormins（目标最小最大化标记列表，1：最小化该目标；-1：最大化该目标）
        Dim = 5  # 初始化Dim（决策变量维数）
        varTypes = [1, 1, 1, 0, 0]  # 初始化varTypes（决策变量的类型，元素为0表示对应的变量是连续的；1表示是离散的）
        lb = [1, 50, 1, 1e-5, 0]  # 决策变量下界
        ub = [5, 2500, 500, 1e-3, 0.01]  # 决策变量上界
        lbin = [1, 1, 1, 1, 1]  # 决策变量下边界（0表示不包含该变量的下边界，1表示包含）
        ubin = [1, 1, 1, 1, 1]  # 决策变量上边界（0表示不包含该变量的上边界，1表示包含）
        ea.Problem.__init__(self, name, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin)
        # 设置用多线程还是多进程
        self.PoolType = PoolType
        if self.PoolType == 'Thread':
            self.pool = ThreadPool(2)  # 设置池的大小
        elif self.PoolType == 'Process':
            num_cores = int(mp.cpu_count())  # 获得计算机的核心数
            self.pool = ProcessPool(num_cores)  # 设置池的大小

    def aimFunc(self, pop):  # 目标函数
        Vars = pop.Phen  # 得到决策变量矩阵
        args = list(zip(list(range(pop.sizes)), [Vars] * pop.sizes))
        if self.PoolType == 'Thread':
            pop.ObjV = np.array(list(self.pool.map(subAimFunc, args)))
        elif self.PoolType == 'Process':
            result = self.pool.map_async(subAimFunc, args)
            result.wait()
            pop.ObjV = np.array(result.get())  # 计算目标函数值，赋值给pop种群对象的ObjV属性
'''===============================LVQ神经网络设置目标函数==========================='''
def subAimFunc(args):
    i = args[0]
    Vars = args[1]
    V1 = int(Vars[i, 0])
    V2 = int(Vars[i, 1])
    V3 = int(Vars[i, 2])
    V4 = Vars[i, 3]
    V5 = Vars[i, 4]
    time_start = time.time()
    lgmlvq = LgmlvqModel(prototypes_per_class=V1, max_iter=V2, random_state=V3, display=True, gtol=V4, regularization=V5)
    lgmlvq.fit(rd.train[:, :len(rd.train[0])-1], rd.train[:, len(rd.train[0])-1])
    time_end = time.time()
    score = lgmlvq.score(rd.test[:, :len(rd.test[0]) - 1], rd.test[:, len(rd.test[0])-1])
    s = time_end - time_start
    return score, s