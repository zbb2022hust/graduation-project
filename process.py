import geatpy as ea
import matplotlib.pyplot as plt
import gawlvq as gl

"""===============================GA-WLVQ框架==========================="""
if __name__ == '__main__':

    """===============================实例化问题对象==========================="""
    PoolType = 'Process'  # 设置采用多进程，若修改为: PoolType = 'Thread'，则表示用多线程
    problem = gl.gawlvq(PoolType)  # 生成问题对象
    """=================================种群设置==============================="""
    Encoding = 'BG'       # 编码方式
    NIND = 100            # 种群规模
    Field = ea.crtfld(Encoding, problem.varTypes, problem.ranges, problem.borders) # 创建区域描述器
    population = ea.Population(Encoding, Field, NIND)  # 实例化种群对象（此时种群还没被初始化，仅仅是完成种群对象的实例化）
    """===============================算法参数设置============================="""
    myAlgorithm = ea.moea_NSGA2_templet(problem, population)  # 实例化一个算法模板对象
    myAlgorithm.MAXGEN = 100

    # 最大进化代数
    myAlgorithm.drawing = 1  # 设置绘图方式（0：不绘图；1：绘制结果图；2：绘制目标空间过程动画；3：绘制决策空间过程动画）
    """==========================调用算法模板进行种群进化======================="""
    NDSet = myAlgorithm.run()   # 执行算法模板，得到帕累托最优解集NDSet
    NDSet.save()                # 把结果保存到文件中
    problem.pool.close()  # 及时关闭问题类中的池，否则在采用多进程运算后内存得不到释放
    # 输出
    print('用时：%s 秒'%(myAlgorithm.passTime))
    print('评价次数：%d 次' % (myAlgorithm.evalsNum))
    print('非支配个体数：%s 个'%(NDSet.sizes))
    print('单位时间找到帕累托前沿点个数：%s 个'%(int(NDSet.sizes // myAlgorithm.passTime)))
    # 输出结果
    plt.show()