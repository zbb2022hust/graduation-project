from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import kstest
import pandas as pd
import dataP

data = pd.read_excel('alldata2.xlsx', header=0)

# 数据归一化
'''scaler = MinMaxScaler()
sdata = pd.DataFrame(scaler.fit_transform(data.iloc[:, 0:len(data.columns) - 1]))
sdata = pd.concat([sdata, data.loc[:, '故障类别']], axis=1, ignore_index=True)
sdata.columns = data.columns'''
# 分割出制热和制冷数据
dw = data[data['模式'].isin(['1'])]   # w为制热
dc = data[data['模式'].isin(['0'])]

"""===============================S-H-EDS方法去除异常值==========================="""
# 分割出各故障类别数据
d1 = dw[dw['故障类别'].isin(['1'])]
d7 = dc[dc['故障类别'].isin(['1'])]
d2 = dw[dw['故障类别'].isin(['2'])]
d8 = dc[dc['故障类别'].isin(['2'])]
d3 = dw[dw['故障类别'].isin(['3'])]
d4 = dw[dw['故障类别'].isin(['4'])]
d5 = dw[dw['故障类别'].isin(['5'])]
d6 = dw[dw['故障类别'].isin(['0'])]
d9 = dc[dc['故障类别'].isin(['0'])]

h1 = d1.loc[:, '模块高压']
h2 = d2.loc[:, '模块高压']
h3 = d3.loc[:, '模块高压']
h4 = d4.loc[:, '模块高压']
h5 = d5.loc[:, '模块高压']
h6 = d6.loc[:, '模块高压']
h7 = d7.loc[:, '模块高压']
h8 = d8.loc[:, '模块高压']
h9 = d9.loc[:, '模块高压']
# 正态性检验
kstest(h1, 'norm')
kstest(h2, 'norm')
kstest(h3, 'norm')
kstest(h4, 'norm')
kstest(h5, 'norm')
kstest(h6, 'norm')
kstest(h7, 'norm')
kstest(h8, 'norm')
kstest(h9, 'norm')

hn = list()
ln = list()
"""===============================d1==========================="""
# d1高压异常值检测
HP = d1.loc[:, '模块高压']
HP = dataP.EWMA(HP)
abh = dataP.abnormal(HP, 1, alpha=0.95, ub=0.45, hybrid=True)
# d1低压异常值检测
LP = d1.loc[:, '模块低压']
LP = dataP.EWMA(LP)
abl = dataP.abnormal(LP, 1, alpha=0.95, ub=0.45, hybrid=True)
# 存储异常值位置
hn = hn+abh
ln = ln+abl
"""===============================d7==========================="""
# d7高压异常值检测
HP = d7.loc[:, '模块高压']
HP = dataP.EWMA(HP)
abh = dataP.abnormal(HP, 1, alpha=0.95, ub=0, hybrid=True)
# d7低压异常值检测
LP = d7.loc[:, '模块低压']
LP = dataP.EWMA(LP)
abl = dataP.abnormal(LP, 1, alpha=0.95, ub=0, hybrid=True)

# 修改实际位置
abh = list(map(lambda x: x+1244, abh))
abl = list(map(lambda x: x+1244, abl))
hn = hn+abh
ln = ln+abl
"""===============================d2==========================="""
# d2高压异常值检测
HP = d2.loc[:, '模块高压']
HP = dataP.EWMA(HP)
abh = dataP.abnormal(HP, 1, alpha=0.95, ub=0.42, hybrid=True)
# d2低压异常值检测
LP = d2.loc[:, '模块低压']
LP = dataP.EWMA(LP)
abl = dataP.abnormal(LP, 1, alpha=0.95, ub=0.42, hybrid=True)

# 修改实际位置
abh = list(map(lambda x: x+2729, abh))
abl = list(map(lambda x: x+2729, abl))
hn = hn+abh
ln = ln+abl
"""===============================d8==========================="""
# d8高压异常值检测
HP = d8.loc[:, '模块高压']
HP = dataP.EWMA(HP)
abh = dataP.abnormal(HP, 1, alpha=0.95, ub=0.08, hybrid=True)
# d8低压异常值检测
LP = d8.loc[:, '模块低压']
LP = dataP.EWMA(LP)
abl = dataP.abnormal(LP, 1, alpha=0.95, ub=0.08, hybrid=True)
# 修改实际位置
abh = list(map(lambda x: x+4004, abh))
abl = list(map(lambda x: x+4004, abl))
hn = hn+abh
ln = ln+abl
"""===============================d3==========================="""
# d3高压异常值检测
HP = d3.loc[:, '模块高压']
HP = dataP.EWMA(HP)
abh = dataP.abnormal(HP, 1, alpha=0.95, ub=0.16, hybrid=True)
# d3低压异常值检测
LP = d3.loc[:, '模块低压']
LP = dataP.EWMA(LP)
abl = dataP.abnormal(LP, 1, alpha=0.95, ub=0.2, hybrid=True)
# 修改实际位置
abh = list(map(lambda x: x+5255, abh))
abl = list(map(lambda x: x+5255, abl))
hn = hn+abh
ln = ln+abl
"""===============================d4==========================="""
# d4高压异常值检测
HP = d4.loc[:, '模块高压']
HP = dataP.EWMA(HP)
abh = dataP.abnormal(HP, 1, alpha=0.95, ub=0.16, hybrid=True)
# d4低压异常值检测
LP = d4.loc[:, '模块低压']
LP = dataP.EWMA(LP)
abl = dataP.abnormal(LP, 1, alpha=0.95, ub=0.2, hybrid=True)
# 修改实际位置
abh = list(map(lambda x: x+7061, abh))
abl = list(map(lambda x: x+7061, abl))
hn = hn+abh
ln = ln+abl
"""===============================d5==========================="""
# d5高压异常值检测
HP = d5.loc[:, '模块高压']
HP = dataP.EWMA(HP)
abh = dataP.abnormal(HP, 1, alpha=0.98, ub=0.1, hybrid=True)
# d5低压异常值检测
LP = d5.loc[:, '模块低压']
LP = dataP.EWMA(LP)
abl = dataP.abnormal(LP, 1, alpha=0.98, ub=0.08, hybrid=True)
# 修改实际位置
abh = list(map(lambda x: x+8596, abh))
abl = list(map(lambda x: x+8596, abl))
hn = hn+abh
ln = ln+abl
"""===============================d6==========================="""
# d6高压异常值检测
HP = d6.loc[:, '模块高压']
HP = dataP.EWMA(HP)
abh = dataP.abnormal(HP, 1, alpha=0.95, ub=0.18, hybrid=True)
# d6低压异常值检测
LP = d6.loc[:, '模块低压']
LP = dataP.EWMA(LP)
abl = dataP.abnormal(LP, 1, alpha=0.95, ub=0.18, hybrid=True)
# 修改实际位置
abh = list(map(lambda x: x+11836, abh))
abl = list(map(lambda x: x+11836, abl))
hn = hn+abh
ln = ln+abl
"""===============================d9==========================="""
# d9高压异常值检测
HP = d9.loc[:, '模块高压']
HP = dataP.EWMA(HP)
abh = dataP.abnormal(HP, 1, alpha=0.95, ub=0.15, hybrid=True)
# d9低压异常值检测
LP = d9.loc[:, '模块低压']
LP = dataP.EWMA(LP)
abl = dataP.abnormal(LP, 1, alpha=0.95, ub=0.12, hybrid=True)
# 修改实际位置
abh = list(map(lambda x: x+14436, abh))
abl = list(map(lambda x: x+14436, abl))
hn = hn+abh
ln = ln+abl
"""===============================高压总检测==========================="""
hn.sort()
hn = pd.Series(hn)
HP = data.loc[:, '模块高压']
ax1 = plt.subplot(221)
plt.plot(HP.index, HP, color='#000000')
start = 0
for i in range(len(hn)):
    if i == len(hn)-1:
        HP[hn[start]:hn[i]].plot(color='#0000FF')
        break
    elif (hn[i] + 1) != hn[i + 1]:
        HP[hn[start]:hn[i]].plot(color='#0000FF')
        start = i+1
plt.xlabel('时间序列')
plt.ylabel('模块高压')
"""===============================低压总检测==========================="""
ln.sort()
ln = pd.Series(ln)
LP = data.loc[:, '模块低压']
ax2 = plt.subplot(222)
plt.plot(LP.index, LP, color='#000000')
start = 0
for i in range(len(ln)):
    if i == len(ln)-1:
        LP[ln[start]:ln[i]].plot(color='#0000FF')
        break
    elif (ln[i] + 1) != ln[i + 1]:
        LP[ln[start]:ln[i]].plot(color='#0000FF')
        start = i+1
plt.xlabel('时间序列')
plt.ylabel('模块低压')


th = list()
tl = list()
# 温度指标
"""===============================d1==========================="""
# d1高压异常值检测
t1 = d1.loc[:, '压缩机排气温度']
t1 = dataP.EWMA(t1)
t1h = dataP.abnormal(t1, 1, alpha=0.95, ub=0.15, hybrid=True)
# d1低压异常值检测
t2 = d1.loc[:, '汽分出管温度']
t2 = dataP.EWMA(t2)
t2l = dataP.abnormal(t2, 1, alpha=0.95, ub=0.15, hybrid=True)
# 存储异常值位置
th = th+t1h
tl = tl+t2l
"""===============================d7==========================="""
# d7高压异常值检测
t1 = d7.loc[:, '压缩机排气温度']
t1 = dataP.EWMA(t1)
t1h = dataP.abnormal(t1, 1, alpha=0.95, ub=0, hybrid=True)
# d7低压异常值检测
t2 = d7.loc[:, '汽分出管温度']
t2 = dataP.EWMA(t2)
t2l = dataP.abnormal(t2, 1, alpha=0.95, ub=0, hybrid=True)

# 修改实际位置
t1h = list(map(lambda x: x+1244, t1h))
t2l = list(map(lambda x: x+1244, t2l))
th = th+t1h
tl = tl+t2l
"""===============================d2==========================="""
# d2高压异常值检测
t1 = d2.loc[:, '压缩机排气温度']
t1 = dataP.EWMA(t1)
t1h = dataP.abnormal(t1, 1, alpha=0.95, ub=0.15, hybrid=True)
# d2低压异常值检测
t2 = d2.loc[:, '汽分出管温度']
t2 = dataP.EWMA(t2)
t2l = dataP.abnormal(t2, 1, alpha=0.95, ub=0.15, hybrid=True)

# 修改实际位置
t1h = list(map(lambda x: x+2729, t1h))
t2l = list(map(lambda x: x+2729, t2l))
th = th+t1h
tl = tl+t2l
"""===============================d8==========================="""
# d8高压异常值检测
t1 = d8.loc[:, '压缩机排气温度']
t1 = dataP.EWMA(t1)
t1h = dataP.abnormal(t1, 1, alpha=0.95, ub=0.15, hybrid=True)
# d8低压异常值检测
t2 = d8.loc[:, '汽分出管温度']
t2 = dataP.EWMA(t2)
t2l = dataP.abnormal(t2, 1, alpha=0.95, ub=0.15, hybrid=True)

# 修改实际位置
t1h = list(map(lambda x: x+4004, t1h))
t2l = list(map(lambda x: x+4004, t2l))
th = th+t1h
tl = tl+t2l
"""===============================d3==========================="""
# d3高压异常值检测
t1 = d3.loc[:, '压缩机排气温度']
t1 = dataP.EWMA(t1)
t1h = dataP.abnormal(t1, 1, alpha=0.95, ub=0.15, hybrid=True)
# d3低压异常值检测
t2 = d3.loc[:, '汽分出管温度']
t2 = dataP.EWMA(t2)
t2l = dataP.abnormal(t2, 1, alpha=0.95, ub=0.15, hybrid=True)

# 修改实际位置
t1h = list(map(lambda x: x+5255, t1h))
t2l = list(map(lambda x: x+5255, t2l))
th = th+t1h
tl = tl+t2l
"""===============================d4==========================="""
# d4高压异常值检测
t1 = d4.loc[:, '压缩机排气温度']
t1 = dataP.EWMA(t1)
t1h = dataP.abnormal(t1, 1, alpha=0.95, ub=0.15, hybrid=True)
# d4低压异常值检测
t2 = d4.loc[:, '汽分出管温度']
t2 = dataP.EWMA(t2)
t2l = dataP.abnormal(t2, 1, alpha=0.95, ub=0.15, hybrid=True)

# 修改实际位置
t1h = list(map(lambda x: x+7061, t1h))
t2l = list(map(lambda x: x+7061, t2l))
th = th+t1h
tl = tl+t2l
"""===============================d5==========================="""
# d5高压异常值检测
t1 = d5.loc[:, '压缩机排气温度']
t1 = dataP.EWMA(t1)
t1h = dataP.abnormal(t1, 1, alpha=0.95, ub=0.15, hybrid=True)
# d5低压异常值检测
t2 = d5.loc[:, '汽分出管温度']
t2 = dataP.EWMA(t2)
t2l = dataP.abnormal(t2, 1, alpha=0.95, ub=0.15, hybrid=True)

# 修改实际位置
t1h = list(map(lambda x: x+8596, t1h))
t2l = list(map(lambda x: x+8596, t2l))
th = th+t1h
tl = tl+t2l
"""===============================d6==========================="""
# d6高压异常值检测
t1 = d6.loc[:, '压缩机排气温度']
t1 = dataP.EWMA(t1)
t1h = dataP.abnormal(t1, 1, alpha=0.95, ub=0.15, hybrid=True)
# d6低压异常值检测
t2 = d6.loc[:, '汽分出管温度']
t2 = dataP.EWMA(t2)
t2l = dataP.abnormal(t2, 1, alpha=0.95, ub=0.15, hybrid=True)

# 修改实际位置
t1h = list(map(lambda x: x+11836, t1h))
t2l = list(map(lambda x: x+11836, t2l))
th = th+t1h
tl = tl+t2l
"""===============================d9==========================="""
# d9高压异常值检测
t1 = d9.loc[:, '压缩机排气温度']
t1 = dataP.EWMA(t1)
t1h = dataP.abnormal(t1, 1, alpha=0.95, ub=0.15, hybrid=True)
# d9低压异常值检测
t2 = d9.loc[:, '汽分出管温度']
t2 = dataP.EWMA(t2)
t2l = dataP.abnormal(t2, 1, alpha=0.95, ub=0.15, hybrid=True)

# 修改实际位置
t1h = list(map(lambda x: x+14436, t1h))
t2l = list(map(lambda x: x+14436, t2l))
th = th+t1h
tl = tl+t2l
"""===============================高压总检测==========================="""
th.sort()
th = pd.Series(th)
t1 = data.loc[:, '压缩机排气温度']
ax3 = plt.subplot(223)
plt.plot(t1.index, t1, color='#000000')
start = 0
for i in range(len(th)):
    if i == len(th)-1:
        t1[th[start]:th[i]].plot(color='#0000FF')
        break
    elif (th[i] + 1) != th[i + 1]:
        t1[th[start]:th[i]].plot(color='#0000FF')
        start = i+1
plt.xlabel('时间序列')
plt.ylabel('压缩机排气温度')
"""===============================低压总检测==========================="""
tl.sort()
tl = pd.Series(tl)
t2 = data.loc[:, '汽分出管温度']
ax4 = plt.subplot(224)
plt.plot(t2.index, t2, color='#000000')
start = 0
for i in range(len(tl)):
    if i == len(tl)-1:
        t2[tl[start]:tl[i]].plot(color='#0000FF')
        break
    elif (tl[i] + 1) != tl[i + 1]:
        t2[tl[start]:tl[i]].plot(color='#0000FF')
        start = i+1
plt.xlabel('时间序列')
plt.ylabel('气分出管温度')
"""===============================综合四种异常值位置并剔除==========================="""
# 将高低压判断的异常值剔除
st = pd.concat([ln, hn, th, tl], ignore_index=True)
st.sort_values(inplace=True)
st = pd.DataFrame(st)
st.reset_index(drop=True, inplace=True)
st = pd.Series(st.iloc[:, 0])
ab = pd.Series()
i = 0
for m in range(len(st)):
    ab = pd.concat([ab, pd.Series(st[i])], ignore_index=True)
    if i == len(st)-1:
        break
    for n in list(range(1, 5)):
        if st[n+i] == st[i+n-1]:
            continue
        elif st[n+i] > st[n+i-1]:
            i = i+n
            break
"""===============================EWMA作图==========================="""
# 模块高压趋势提取
p1 = data.loc[:, '模块高压']
p2 = data.loc[:, '模块低压']
p3 = data.loc[:, '压缩机排气温度']
p4 = data.loc[:, '汽分出管温度']

p1 = dataP.EWMA(p1)
p2 = dataP.EWMA(p2)
p3 = dataP.EWMA(p3)
p4 = dataP.EWMA(p4)

ax1 = plt.subplot(221)
plt.plot(p1.index, p1, color='#000000')
plt.xlabel('时间序列')
plt.ylabel('模块高压')
ax2 = plt.subplot(222)
plt.plot(p2.index, p2, color='#000000')
plt.xlabel('时间序列')
plt.ylabel('模块低压')
ax3 = plt.subplot(223)
plt.plot(p3.index, p3, color='#000000')
plt.xlabel('时间序列')
plt.ylabel('压缩机排气温度')
ax4 = plt.subplot(224)
plt.plot(p4.index, p4, color='#000000')
plt.xlabel('时间序列')
plt.ylabel('气分出管温度')

data.drop(index=ab, inplace=True)
data.to_excel('E:\大四\毕业设计\算法\BPWLVQ\cdata.xls', index=True)

