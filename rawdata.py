import pandas as pd
import numpy as np
import dataP

njs = pd.read_excel('E:\大四\毕业设计\算法\BPWLVQ\\rawdata\coldnormalnj.xlsx', header=0)
wjs = pd.read_excel('E:\大四\毕业设计\算法\BPWLVQ\\rawdata\coldnormalwj.xlsx', header=0)
njsc = njs.columns
wjsc = wjs.columns
datanj = pd.read_excel('E:\大四\毕业设计\算法\BPWLVQ\\rawdata\cool0exv71nj.xlsx', header=0)  # 数据文件读取
datanj = dataP.cleanbool(datanj)

nj = pd.DataFrame()
# 提取预先值
for i in njsc:
    nj = pd.concat([nj, datanj.loc[:, i]], axis=1, ignore_index=True)
nj.columns = njsc
nj.to_excel('warmnormalnj.xls')

nnj = datanj[['室内环境温度', '入管温度', '出管温度', 'EXV', '回风口温度']]  # 利用专家知识初步选择出重要的特征并合成新表
"""===============================36内机数据处理==========================="""
data36 = pd.DataFrame()
for i in datanj.index:
    if datanj.loc[i, '工程编号'] == 1:  # 挑出1号内机数据，并独立成表
        data36 = data36.append(nnj.iloc[i, :], ignore_index=True)
"""===============================71内机数据处理==========================="""
data71 = pd.DataFrame()
for i in datanj.index:
    if datanj.loc[i, '工程编号'] == 7:
        data71 = data71.append(nnj.loc[i, :], ignore_index=True)
"""===============================112内机数据处理==========================="""
data112 = pd.DataFrame()
for i in datanj.index:
    if datanj.loc[i, '工程编号'] == 8:
        data112 = data112.append(nnj.loc[i, :], ignore_index=True)
"""===============================22内机数据处理==========================="""
data22 = pd.DataFrame()
for i in datanj.index:
    if datanj.loc[i, '工程编号'] == 9:
        data22 = data22.append(nnj.loc[i, :], ignore_index=True)
"""===============================5内机数据处理==========================="""
data5 = pd.DataFrame()
for i in datanj.index:
    if datanj.loc[i, '工程编号'] == 10:
        data5 = data5.append(nnj.loc[i, :], ignore_index=True)
"""===============================给各内机数据标签重命名==========================="""
data36.columns = ['EXV1', '入管温度1', '出管温度1', '回风口温度1', '室内环境温度1']
data71.columns = ['EXV5', '入管温度5', '出管温度5', '回风口温度5', '室内环境温度5']
data112.columns = ['EXV2', '入管温度2', '出管温度2', '回风口温度2', '室内环境温度2']
data22.columns = ['EXV3', '入管温度3', '出管温度3', '回风口温度3', '室内环境温度3']
data5.columns = ['EXV4', '入管温度4', '出管温度4', '回风口温度4', '室内环境温度4']
"""===============================外机数据处理==========================="""
datawj = pd.read_excel('E:\大四\毕业设计\算法\BPWLVQ\\rawdata\warmfwvsxwj.xlsx', header=0)
wj = pd.DataFrame()
# 提取预先值
for i in wjsc:
    wj = pd.concat([wj, datawj.loc[:, i]], axis=1, ignore_index=True)
wj.columns = wjsc
wj.to_excel('warmnormalnj.xls')
nwj = datawj[['室外环境温度', '本机分配能力', '本机当前运行能力', '压缩机1运行频率', '风机1运行频率', '模块高压', '模块低压',
       '压缩机1排气温度', '压缩机1壳顶温度', '化霜温度1', '过冷器液出温度', '过冷器气出温度', '汽分进管温度',
       '汽分出管温度', '过冷器EXV', '压缩机1模块温度', '风机1模块温度']]
nwj = nwj.drop([0, 1, 2], axis=0)  # 根据实际选择删除行
nwj.reset_index(drop=True, inplace=True)  # 给行标签重新排序
nwj['模式'] = [1]*2068  # 制热为1；制冷为0；数字为行数
"""===============================内外机数据整合==========================="""
ndata = pd.concat([datawj, data36, data112, data22, data5, data71], axis=1, ignore_index=False)
ndata['故障类别'] = [1]*2068  # 卡死0为1；卡死100为2；泄漏为3；四通阀失效为4；四通阀掉电为5；正常为0
ndata.to_excel('coolexvjam0.xls')
d1 = pd.read_excel('warmexvjam0.xlsx', header=0)
d2 = pd.read_excel('coldexvjam0.xlsx', header=0)
d3 = pd.read_excel('warmexvjam100.xlsx', header=0)
d4 = pd.read_excel('coldexvjam100.xlsx', header=0)
d5 = pd.read_excel('warmexvleak.xlsx', header=0)
d6 = pd.read_excel('warmfwvdd.xlsx', header=0)
d7 = pd.read_excel('warmfwvsx.xlsx', header=0)
d8 = pd.read_excel('warmnormal.xlsx', header=0)
d9 = pd.read_excel('coldnormal.xlsx', header=0)
all = pd.read_excel('alldata.xlsx', header=0)
data112 = data112.drop(range(2), axis=0)
data22 = data22.drop(range(2), axis=0)
data36 = data36.drop(range(1), axis=0)
data5 = data5.drop(range(5), axis=0)
data71 = data71.drop(range(3), axis=0)
datawj = datawj.drop(range(7), axis=0)
data112.reset_index(drop=True, inplace=True)
data22.reset_index(drop=True, inplace=True)
data36.reset_index(drop=True, inplace=True)
data5.reset_index(drop=True, inplace=True)
data71.reset_index(drop=True, inplace=True)
datawj.reset_index(drop=True, inplace=True)


