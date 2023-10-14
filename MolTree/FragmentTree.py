
import sys

import numpy as np
from scipy import special
import itertools


###############################################
# 求解最大公约数
def gcd(a, b):
    """
    第一种方法：欧几里得算法----辗转相除法
    :param a: 第一个数
    :param b: 第二个数
    :return: 最大公约数
    """
    # 如果最终余数为0 公约数就计算出来了
    while (b != 0):
        temp = a % b
        a = b
        b = temp
    return a


###############################################
# 求解最小公倍数
def lcm(a, b):
    return a * b / gcd(a, b)


###############################################
# Compute ERT
def FIND_ERT(a):
    # 初始化
    ERT = np.zeros(shape=(a[0], len(a)))
    for r in range(a[0]):
        if r == 0:
            ERT[r][0] = 0
        else:
            ERT[r][0] = np.inf

    # 初始化为1，或其他任意值，方便后边判断和置零退出循环
    for i in range(1, len(a)):
        ERT[0][i] = 1

    # w_before = [1,0]
    # w = np.tile(w_before,(a[0],1))
    # 计算数组的值
    # temp=0
    ################################################
    # a0与ai不互质
    for i in range(1, len(a)):
        d = gcd(a[0], a[i])
        # w[0]=[1,0]
        for p in range(d):
            n_q = []
            for q in range(a[0]):
                if q % d == p:
                    n_q.append(ERT[q][i])
            temp = min(n_q)
            # counter=0
            # "We repeat this loop until n equals 0".
            while ERT[0][i] != 0:
                n = temp + a[i]  # n一直保留未取最小值之前的值,ERT[][]是取最小值之后的
                # 下标必须是int，float不行
                remainder = int(n % a[0])
                # 先赋值过去（原来位置的数都是0），再比较大小
                nr = ERT[remainder][i - 1]
                ERT[remainder][i] = min(nr, n)
                temp = ERT[remainder][i]
                # counter+=1
                # if n<nr:
                # 论文中该处下下标是从2开始的，论文中的k指a的下标，从1开始，不是从0开始
                # w[remainder]=[i+1,counter]
                # else:
                # counter=0
    return ERT


# Finding One Decompositions.


def change(M, a):
    c = []
    # rule1:元素数量限制
    if M < 500:
        H_n = 72
        C_n = 29
        N_n = 10
        O_n = 18
        S_n = 7
    # if 500 < M <= 1000:
    #     H_n = 126
    #     C_n = 66
    #     N_n = 25
    #     O_n = 27
    #     S_n = 8
    else:
        H_n = np.inf
        C_n = np.inf
        N_n = np.inf
        O_n = np.inf
        S_n = np.inf
    # num = min(M // a[0] + 1, H_n)
    for i in range(min(M // a[0] + 1, H_n)):
        for j in range(min((M - i * a[0]) // a[1] + 1, C_n)):
            # rule4:H/C
            if j < i / 3.1:
                continue
            if j > i / 0.2:
                break
            for m in range(min((M - i * a[0] - j * a[1]) // a[2] + 1, N_n)):
                # rule5:N/C
                if m > j * 1.3:
                    break
                # # # rule2
                # if ((i + m) % 2) != 0: #屏蔽掉可以实现RFI
                #     continue           #屏蔽掉可以实现RFI
                # DU(degree od unsaturation)非负
                #                if -i/2+j+m/2+1<0:
                #                    continue
                for n in range(min((M - i * a[0] - j * a[1] - m * a[2]) // a[3] + 1, O_n)):
                    # rule5:O/C
                    if n > j * 1.2:
                        break
                    for p in range(min((M - i * a[0] - j * a[1] - m * a[2] - n * a[3]) // a[4] + 1, S_n)):
                        # rule5:S/C
                        if p > j * 0.8:
                            break
                        # rule2
                        if 1 * i + 4 * j + 3 * m + 2 * n + 2 * p < 2 * 4:
                            continue
                        if 1 * i + 4 * j + 3 * m + 2 * n + 2 * p < 2 * (i + j + m + n + p) - 2:  # 有问题 原始为-1 是否应为-2
                            continue
                        # if ((1 * i + 4 * j + 3 * m + 2 * n + 2 * p) % 2) != 0: #屏蔽掉可以实现RFI
                        #     continue                                           #屏蔽掉可以实现RFI
                        if i * a[0] + j * a[1] + m * a[2] + n * a[3] + p * a[4] == M:
                            c.append([i, j, m, n, p])
    return c


# 计算节点权重 输入：预测的原子各个元素个数，质量精度，真实质量 输出：该节点被赋予的权重
def node_weight(atom, quality_deviation, real_mass):
    quality_deviation = quality_deviation / 3
    real = [1.00783, 12.00000, 14.00307, 15.99491, 31.97207]  # H C N O S
    calmass = np.multiply(np.array(atom), np.array(real))
    mass = np.sum(calmass) + 1.00783
    wucha = abs(real_mass - mass) / (np.sqrt(2) * quality_deviation)
    weight = special.erfc(wucha)
    return weight


# 计算边权重 输入：质谱峰值（单个峰），Loss列表 输出：带评分Loss列表
def edge_weight(mz, Loss_all, div):
    mz = sorted(mz, reverse=True)
    Neutral_loss = [[6, 6, 0, 0, 0],
                    [2, 1, 0, 1, 0],
                    [4, 1, 0, 1, 0],
                    [0, 1, 0, 1, 0],
                    [2, 1, 0, 2, 0],
                    [0, 1, 0, 2, 0],
                    [4, 2, 0, 2, 0],
                    [2, 2, 0, 1, 0],
                    [6, 3, 0, 2, 0],
                    [4, 3, 0, 4, 0],
                    [2, 3, 0, 3, 0],
                    [8, 5, 0, 4, 0],
                    [8, 4, 0, 0, 0],
                    [8, 5, 0, 0, 0],
                    [2, 2, 0, 0, 0],
                    [2, 0, 0, 1, 0],
                    [4, 1, 0, 0, 0],
                    [4, 2, 0, 0, 0],
                    [9, 3, 1, 0, 0],
                    [1, 1, 1, 1, 0],
                    [4, 1, 2, 1, 0],
                    [2, 0, 0, 0, 1],
                    [0, 0, 0, 0, 1],
                    [0, 0, 0, 2, 1],
                    [0, 0, 0, 3, 1],
                    [2, 0, 0, 4, 1],
                    [5, 1, 1, 0, 0],
                    [3, 1, 1, 0, 0],
                    [3, 0, 1, 0, 0],
                    [10, 6, 0, 4, 0],
                    [10, 6, 0, 5, 0],
                    [8, 6, 0, 6, 0],
                    # [3, 1, 0, 0, 0],
                    # [1,1,0,1,0],
                    ]
    breakloss = []
    n = []
    numarray = []
    k2 = []
    kk = []
    res_all = []
    for i in range(len(Loss_all)):
        father_num = Loss_all[i]['fathernum']
        child_num = Loss_all[i]['childnum']
        loss = Loss_all[i]['loss']
        if loss in Neutral_loss:
            n.append(1)
        else:
            for j in range(len(Neutral_loss)):
                loss2 = list(map(lambda x: x[0] - x[1], zip(loss, Neutral_loss[j])))
                if min(loss2) >= 0:
                    breakloss.append(Neutral_loss[j])
            loopnum = len(breakloss)
            numnum = max(loss)
            for k in range(loopnum):
                for o in range(numnum + 1):
                    loss3 = [idi * o for idi in breakloss[k]]
                    loss2 = list(map(lambda x: x[0] - x[1], zip(loss, loss3)))
                    if loss3 == loss:
                        n.append(o)
                    if min(loss2) < 0:
                        numarray.append(o)
                        break
                    if o == numnum and min(loss2) >= 0:
                        numarray.append(o)
            if loss[2] % 2 == 1 and loss[0] % 2 == 0:
                scoree = 0
            elif not n and len(numarray) > 1:
                for x in range(len(numarray)):
                    z = numarray[x]
                    for zz in range(z):
                        k2.append(zz)
                    kk.append(k2)
                    k2 = []
                s1 = ','.join(str(n1) for n1 in kk)
                s1 = eval(s1)
                for a1 in itertools.product(*s1):
                    res = list(a1)
                    res_all.append(res)
                    if len(res_all) > 10000:
                        break
                sum_list = [0 for _ in range(len(res_all))]
                for num1 in range(len(res_all)):
                    resu = res_all[num1]
                    for num2 in range(len(resu)):
                        maybe = [idi * resu[num2] for idi in breakloss[num2]]
                        sum_list = [x + y for x, y in zip(maybe, sum_list)]
                    if sum_list == loss:
                        n.append(sum(resu))
                    sum_list = [0 for _ in range(len(mz))]
                kk = []
                res_all = []
        if n:
            scoree = min(n)
            reward_score = np.log10(1000 / scoree)
        else:
            reward_score = 0
        breakloss = []
        numarray = []
        n = []
        if loss == [0, 1, 0, 0, 0] or loss == [0, 0, 1, 0, 0]:  # 纯C 纯O 纯N惩罚
            punish = 4 + np.log10(1 - (mz[father_num] - mz[child_num]) / mz[father_num])
        else:
            punish = np.log10(1 - (mz[father_num] - mz[child_num]) / mz[father_num])
        edge_score = reward_score + punish
        node_score = node_weight(Loss_all[i]['father'], div, mz[father_num]) + node_weight(Loss_all[i]['child'], div,
                                                                                           mz[child_num])
        final_score = edge_score + node_score
        Loss_all[i]['score'] = final_score
        if i == 37:
            print(1)
        if i == 39:
            print(1)
    return Loss_all


# 生成最终的分子碎片树（Vertex insertion heuristic法）输入：质谱峰值（单个峰），带评分的Loss列表 输出：所有可能的根节点所成的最大分值碎片树
def Generate_Tree(Loss, mz):
    fplist = [[] for _ in range(len(mz))]
    for j in range(len(Loss)):
        fnum = Loss[j]['fathernum']
        cnum = Loss[j]['childnum']
        ffp = Loss[j]['father']
        cfp = Loss[j]['child']
        if ffp not in fplist[fnum]:
            fplist[fnum].append(ffp)
        if cfp not in fplist[cnum]:
            fplist[cnum].append(cfp)
    Treeweight = []
    Final_score = []
    for j in range(len(fplist[0])):
        root = fplist[0][j]
        loss = dict(fathernum=0, childnum=0, father=root, child=root, loss=0, oriscore=0, finscore=0)
        edge = []
        edge.append(root)
        edgeweight = []
        edgeweight2 = []
        edgeweight.append(loss)
        Weightt = []
        arrayy = []
        arrayy.append(loss)
        for k in range(1, len(mz)):
            for i in range(len(Loss)):
                fnum = Loss[i]['fathernum']
                cnum = Loss[i]['childnum']
                ffp = Loss[i]['father']
                cfp = Loss[i]['child']
                loss = Loss[i]['loss']
                score = Loss[i]['score']
                if cnum == k and ffp in edge:
                    edgeweight2.append(Loss[i])
                    # wee = arrayy[fnum]['finscore'] + score
                    wee = score # 权重值是否依赖于其他边
                    edgeweight.append(
                        dict(fathernum=fnum, childnum=cnum, father=ffp, child=cfp, loss=loss, oriscore=score,
                             finscore=wee))
            kk = sorted(edgeweight, key=lambda x: x['finscore'], reverse=True)
            if len(edgeweight) < 2:
                break
            else:
                Weightt.append(kk[0])
                edge.append(kk[0]['child'])
                edgeweight = []
            arrayy.append(dict(fathernum=0, childnum=kk[0]['childnum'], father=root, child=kk[0]['child'], loss=0,
                               oriscore=kk[0]['oriscore'], finscore=kk[0]['finscore']))
            edgeweight.append(dict(fathernum=0, childnum=0, father=root, child=root, loss=0, oriscore=0, finscore=0))
        Treeweight.append(Weightt)
    for i in range(len(Treeweight)):
        final_weight = Treeweight[i]
        for d in final_weight:
            if d.get('fathernum') == d.get('childnum'):
                final_weight.remove(d)
        final_score = 0
        if final_weight:
            for k in range(len(final_weight)):
                final_score += final_weight[k]['oriscore']
        Final_score.append(final_score)
    # cc = sorted(range(len(Final_score)), key=lambda k: Final_score[k], reverse=True)
    order = sorted(range(len(Final_score)), key=lambda i: Final_score[i], reverse=True)
    Final_score = [Final_score[i] for i in order]
    Treeweight = [Treeweight[i] for i in order]
    return Final_score, Treeweight


# 判断前体离子和母离子 输入：质谱峰值（单个峰） 输出：前体离子和母离子的Loss列表
def find_child(mz_list, dmz , dmz2):
    a = [1, 12, 14, 16, 32]
    mz_list = sorted(mz_list, reverse=True)
    mz_list = [x - 1.00783 for x in mz_list]  # 正离子模式 减去H
    Node_list = []
    aa = []
    Loss = []
    node_list1 = []
    node_list2 = []
    real = [1.00783, 12.00000, 14.00307, 15.99491, 31.97207]
    for i in range(len(mz_list)):
        node_list = change(int(mz_list[i]+0.5), a)
        # Node_list.append(node_list)
        for num in node_list:
            calmass = np.multiply(np.array(num), np.array(real))
            mass = np.sum(calmass)
            # if num[2] == 0 and num[4] == 0:
            #     aa.append(num)
            if i==0:
                if abs(mass - mz_list[i]) < dmz and num[2] == 0 and num[4] == 0:  # 质量分辨率
                    aa.append(num)
            else:
                if abs(mass - mz_list[i]) < dmz2 and num[2] == 0 and num[4] == 0:  # 质量分辨率
                    aa.append(num)
        Node_list.append(aa)
        aa = []
    for i in range(len(Node_list)):
        for j in range(len(Node_list[i])):
            father = Node_list[i][j]
            for k in range(i + 1, len(Node_list)):
                for l in range(len(Node_list[k])):
                    child = Node_list[k][l]
                    loss = list(map(lambda x: x[0] - x[1], zip(father, child)))
                    if min(loss) >= 0:
                        loss = dict(fathernum=i, childnum=k, father=father, child=child, loss=loss)
                        if i == 0:
                            if father not in node_list1:
                                node_list1.append(father)
                            if child not in node_list2:
                                node_list2.append(child)
                            Loss.append(loss)
                        if i > 0:
                            if father in node_list2:
                                node_list1.append(father)
                                node_list2.append(child)
                                Loss.append(loss)
            # if node_list1:
            #     node_list2.append(node_list1)
            #     node_list1=[]
    # for i in range(len(Loss)):
    return Loss  ##


def MolTree(mz,div,div2):
    mode = find_child(mz,div,div2)
    array = edge_weight(mz, mode,div)
    Score, Tree = Generate_Tree(array, mz)
    return Tree
