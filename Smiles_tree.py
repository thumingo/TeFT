from rdkit import rdBase, Chem
from rdkit.Chem import AllChem, Draw
from rdkit.Chem import Recap
from rdkit.Chem import Descriptors
from rdkit.Chem import BRICS
import itertools
from collections import Counter
import Recap_TeFT
import json
from itertools import groupby
from MolTree import FragmentTree
import numpy as np


def get_leaves(recap_decomp, n=1):
    for child in recap_decomp.children.values():
        if child.children:
            # if n < 1000:
            get_leaves(child, n=n + 1)


# SMILES转化为分子式
def getmoleculaformula(smiles):
    m = ['C', 'N', 'O', 'H', 'F', 'P', 'S', 'Cl', 'Br', 'I']
    mol1 = Chem.MolFromSmiles(smiles)
    mol1 = Chem.AddHs(mol1)
    atom1 = mol1.GetAtoms()
    count_dist1 = dict()
    for i in m:
        count_dist1[i] = 0
    for at in atom1:
        atom = at.GetSymbol()
        if atom in count_dist1:
            count_dist1[atom] += 1

    return count_dist1


# 利用Recap进行碎裂
def get_recap_tree(mol):
    recap = Recap_TeFT.RecapDecompose_for1(mol)
    # cc = recap_test1.get_exact_mass(mol)
    # print(Chem.MolToSmiles(mol), cc)
    # get_leaves(recap)
    return recap


# 将分子式拆分为元素和分子对应的字典
def chaifen(mf):
    MF = list(mf)
    # k = len(MF)
    res = []
    for i in range(len(MF)):
        if i < len(MF) - 1 and MF[i].isalpha() and MF[i + 1].isalpha():
            res.append(MF[i])
            res.append(1)
        if i < len(MF) - 1 and MF[i].isdigit() and MF[i + 1].isdigit():
            numm = int(MF[i]) * 10 + int(MF[i + 1])
            res.append(numm)
        if i < len(MF) - 1 and MF[i].isdigit() and MF[i + 1].isalpha() and MF[i - 1].isalpha():
            numm = int(MF[i])
            res.append(numm)
        if i < len(MF) - 1 and MF[i].isalpha() and MF[i + 1].isdigit():
            res.append(MF[i])
        if i == len(MF) - 1:
            if MF[i].isalpha():
                res.append(MF[i])
                res.append(1)
            if MF[i].isdigit():
                res.append(int(MF[i]))
    dicc = zip(res[::2], res[1::2])
    return dict(dicc)


# 计算两个SMILES之间的损失
def caculate_loss(smile1, smile2):
    m = ['C', 'N', 'O', 'H', 'F', 'P', 'S', 'Cl', 'Br', 'I']
    mol1 = Chem.MolFromSmiles(smile1)
    mol2 = Chem.MolFromSmiles(smile2)
    mol1 = Chem.AddHs(mol1)
    mol2 = Chem.AddHs(mol2)
    atom1 = mol1.GetAtoms()
    atom2 = mol2.GetAtoms()
    count_dist1 = dict()
    count_dist2 = dict()
    for i in m:
        count_dist1[i] = 0
        count_dist2[i] = 0
    for at in atom1:
        atom = at.GetSymbol()
        if atom in count_dist1:
            count_dist1[atom] += 1
    for at in atom2:
        atom = at.GetSymbol()
        if atom in count_dist2:
            count_dist2[atom] += 1
    fragfenzi = dict(Counter(count_dist1) - Counter(count_dist2))
    return fragfenzi


# 将SIRIUS生成的碎片树结果转换为字典
def moleculartreetodic_SIRIUS(treejson):
    m = ['C', 'N', 'O', 'H', 'F', 'P', 'S', 'Cl', 'Br', 'I']
    count_dist1 = dict()
    Fragdic = []
    Lossdic = []
    listk = []
    with open(treejson, 'r') as k:
        data = json.load(k)
    fragment = data['fragments']
    loss = data['losses']
    for i in range(len(fragment)):
        mf = fragment[i]['molecularFormula']
        id = fragment[i]['id']
        c = chaifen(mf)
        fragdic = dict(id=id, molecularFormula=c, molwt=fragment[i]['mz'])
        Fragdic.append(fragdic)
        # c.clear()
    for i in range(len(loss)):
        mf = loss[i]['molecularFormula']
        c = chaifen(mf)
        lossformula = abs(fragment[loss[i]['source']]['mz'] - fragment[loss[i]['target']]['mz'])
        fragdic = dict(source=loss[i]['source'], molecularFormula=c, lossformula=lossformula, target=loss[i]['target'])
        Lossdic.append(fragdic)
    return Fragdic, Lossdic


# 寻找相似
def findloss(preloss, sourceid, targetid, loss):
    list_source = []
    list_target = []
    b = list(set(list_source) & set(list_target))
    list_source2 = []
    list_target2 = []
    num = 0
    for k in range(len(preloss)):
        source = preloss[k]['source']
        target = preloss[k]['target']
        lossm = preloss[k]['loss']
        if sourceid == source and lossm < loss:
            list_source.append(dict(source=preloss[k], num=num))
            list_source2.append(preloss[k]['target'])
        if targetid == target and lossm < loss:
            list_target.append(dict(target=preloss[k], num=num))
            list_target2.append(preloss[k]['source'])
    while num < 5:
        num += 1
        for j in range(len(list_source2)):
            for k1 in range(len(preloss)):
                if list_source2[j] == preloss[k1]['source'] and preloss[k1]['loss'] < loss and preloss[k1][
                    'target'] not in list_source2:
                    list_source.append(dict(source=preloss[k1], num=num))
                    list_source2.append(preloss[k1]['target'])
        for j in range(len(list_target2)):
            for k1 in range(len(preloss)):
                if list_target2[j] == preloss[k1]['target'] and preloss[k1]['loss'] < loss and preloss[k1][
                    'source'] not in list_target2:
                    list_target.append(dict(target=preloss[k1], num=num))
                    list_target2.append(preloss[k1]['source'])
        b = list(set(list_source2) & set(list_target2))
        if b:
            break
    if b:
        realreal = []
        for j in range(len(b)):
            simid1 = simid2 = b[j]
            for k in range(len(list_source)):
                if simid1 == list_source[k]['source']['target']:
                    realreal.append(list_source[k]['source'])
                    simid1 = list_source[k]['source']['source']
            for k in range(len(list_target)):
                if simid2 == list_target[k]['target']['source']:
                    realreal.append(list_target[k]['target'])
                    simid2 = list_target[k]['target']['target']
        return realreal
    else:
        return b


# 比较SMILES碎片树和分子碎片树，并评分 输入：SMILES碎片树及损失，分子碎片树及其损失 输出：相似的节点，损失以及最终评分
def comparetree(pretree, preloss, realtree, realloss):
    list_simnode = []
    list_lossreal = []
    listk = []
    list_scorenode = []
    list_scoreloss = []
    listkk = []
    listkk2 = []
    # list1 = []
    # list2 = []
    list_losspre = []
    for i in range(len(pretree)):
        premol = pretree[i]['molwt']
        presm = pretree[i]['smiles']
        for j in range(len(realtree)):
            realmol = realtree[j]['molwt']
            if abs(premol - realmol) < 1.00783 * 2:
                sim = dict(real=realtree[j], predic=pretree[i])
                list_simnode.append(sim)  # 寻找所有相似节点，考虑重复情况，按对存入字典中，生成列表
                # list_scorenode.append(realtree[j]['id'])
                # listkk.append(pretree[i]['id'])
    # 比较相似碎片，去除重复，保留最可能的结果
    for i in range(len(list_simnode)):
        id1 = list_simnode[i]['real']['id']
        id2 = list_simnode[i]['predic']['id']
        mf1 = list_simnode[i]['real']['molecularFormula']
        mf2 = list_simnode[i]['predic']['molecularformula']
        diff = dict((Counter(mf1) - Counter(mf2)) + (Counter(mf2) - Counter(mf1)))
        diff_atom = list(diff.keys())
        if id1 == 0:
            if 'N' in diff_atom or 'O' in diff_atom:
                score = 0
            elif 'H' in diff_atom:
                score = 3
            elif len(diff_atom) == 0:
                score = 6
        else:
            if 'N' in diff_atom or 'O' in diff_atom:
                score = 0
            elif 'H' in diff_atom:
                score = 2
            elif len(diff_atom) == 0:
                score = 5
        if id1 not in listkk:
            listkk.append(id1)
            listkk2.append(id2)
            list_scorenode.append(dict(real=list_simnode[i]['real'], predic=list_simnode[i]['predic'], score=score))
        else:
            for k in range(len(list_scorenode)):
                chongfuid = list_scorenode[k]['real']['id']
                if chongfuid == id1:
                    scores = list_scorenode[k]['score']
                    if scores < score:
                        list_scorenode.remove(list_scorenode[k])
                        list_scorenode.append(
                            dict(real=list_simnode[i]['real'], predic=list_simnode[i]['predic'], score=score))

    # 赋分
    # 根据相似碎片查询损失
    for k in range(len(list_scorenode)):
        id = list_scorenode[k]['real']['id']
        for j in range(len(realloss)):
            source = realloss[j]['source']
            target = realloss[j]['target']
            if source == id and target in listkk:
                list_lossreal.append(realloss[j])
                listk.append(list_scorenode[k])
        id2 = list_scorenode[k]['predic']['id']
        for j in range(len(preloss)):
            source = preloss[j]['source']
            target = preloss[j]['target']
            if source == id2 and target in listkk2:
                list_losspre.append(preloss[j])
            # if id == target:
            #     listk.append(realloss[j])
    for k in range(len(list_lossreal)):
        source = list_lossreal[k]['source']
        target = list_lossreal[k]['target']
        lossreal = list_lossreal[k]['molecularFormula']
        lossm = list_lossreal[k]['lossformula']
        for j in range(len(list_scorenode)):
            id = list_scorenode[j]['real']['id']
            if id == source:
                id_ps = list_scorenode[j]['predic']['id']
            if id == target:
                id_pt = list_scorenode[j]['predic']['id']
        for i in range(len(list_losspre)):
            source2 = list_losspre[i]['source']
            target2 = list_losspre[i]['target']
            losspre = list_losspre[i]['lossformula']
            lossm2 = list_losspre[i]['loss']
            if source2 == id_ps and target2 == id_pt:
                losss = dict((Counter(lossreal) - Counter(losspre) + (Counter(losspre) - Counter(lossreal))))
                if 'N' in losss or 'O' in losss:
                    score = 0
                elif 'H' in losss:
                    score = 1.5
                elif len(losss) == 0:
                    score = 4
                list_scoreloss.append(dict(real=list_lossreal[k], pre=list_losspre[i], loss=losss, score=score))
                break
            else:
                c = findloss(preloss, id_ps, id_pt, lossm)
                if c:
                    lossdic = {}
                    loss_m = 0
                    for k in range(len(c)):
                        lossdic = dict(Counter(c[k]['lossformula']) + Counter(lossdic))
                        loss_m += c[k]['loss']
                    pre = dict(source=c[0]['source'], target=c[-1]['target'], loss=loss_m, lossformula=lossdic)
                    losss = dict((Counter(lossdic) - Counter(lossreal) + (Counter(lossreal) - Counter(lossdic))))
                    if 'N' in losss or 'O' in losss:
                        score = 0
                    elif 'H' in losss:
                        score = 1.5
                    # elif 'C' in losss:
                    #     score = 1
                    elif len(losss) == 0:
                        score = 4
                    list_scoreloss.append(dict(real=list_lossreal[i], pre=pre, loss=losss, score=score))
                    break
    # 计算总分
    sum_node = 0
    sum_loss = 0
    for j in range(len(list_scorenode)):
        sum_node += list_scorenode[j]['score']
    for j in range(len(list_scoreloss)):
        sum_loss += list_scoreloss[j]['score']
    sum_score = 5 * (len(realtree) + len(realloss))
    if sum_score != 0:
        FinalScore = (sum_node + sum_loss) / sum_score
    else:
        FinalScore = 0
    return list_scorenode, list_scoreloss, FinalScore


# 将模型预测结果转换为SMILES树
def predict_SMILESTree(smiles):
    celecoxib = Chem.MolFromSmiles(smiles)
    recap = get_recap_tree(celecoxib)
    Smiledic = []
    oi = recap.GetAllChildren()
    keys = list(oi.keys())
    values = list(oi.values())
    keys.append(recap.smiles)
    for i in range(len(keys)):
        smiletomf = getmoleculaformula(keys[i])
        smiledic = dict(smiles=keys[i], id=i, molwt=Recap_TeFT.get_exact_mass(Chem.MolFromSmiles(keys[i])), parents=[],
                        child=[], molecularformula=smiletomf)
        Smiledic.append(smiledic)

    for i in range(len(keys) - 1):
        parent = values[i].parents
        child = values[i].children
        parents = list(parent.keys())
        children = list(child.keys())
        for j in range(len(parents)):
            num = keys.index(parents[j])
            Smiledic[i]['parents'].append(num)
            if num == len(keys) - 1:
                Smiledic[-1]['child'].append(i)
        for j in range(len(children)):
            Smiledic[i]['child'].append(keys.index(children[j]))
        # print(1)

    loss = []
    for j in range(len(Smiledic)):
        smiles = Smiledic[j]['smiles']
        child = Smiledic[j]['child']
        if child:
            for i in range(len(child)):
                c = child[i]
                smilesfrag = Smiledic[c]['smiles']
                molloss = abs(Recap_TeFT.get_exact_mass(Chem.MolFromSmiles(smiles)) - Recap_TeFT.get_exact_mass(
                    Chem.MolFromSmiles(smilesfrag)))
                target = child[i]
                fenzi = caculate_loss(smiles, smilesfrag)
                lossdic = dict(source=j, target=target, loss=molloss, lossformula=fenzi, smiles=smiles,
                               fragsmiles=smilesfrag)
                loss.append(lossdic)
    return Smiledic, loss


# 将自行编写的分子碎片树结果转换为字典
def moleculartreetodic_Prim(Tree):
    global fragdic, fragdic2
    m = ['C', 'N', 'O', 'H', 'F', 'P', 'S', 'Cl', 'Br', 'I']
    count_dist1 = dict()
    Fragdic = []
    Lossdic = []
    zz = []
    maxtree = Tree
    for i in range(len(maxtree)):
        mf_f = maxtree[i]['father']
        mf_c = maxtree[i]['child']
        id_f = maxtree[i]['fathernum']
        id_c = maxtree[i]['childnum']
        if mf_f not in zz:
            zz.append(mf_f)
            mf_f, molwt = listtodic(mf_f)
            fragdic2 = dict(id=id_f, molecularFormula=mf_f, molwt=molwt)
        if mf_c not in zz:
            zz.append(mf_c)
            mf_c, molwt = listtodic(mf_c)
            fragdic = dict(id=id_c, molecularFormula=mf_c, molwt=molwt)
        if fragdic2 not in Fragdic:
            Fragdic.append(fragdic2)
        if fragdic not in Fragdic:
            Fragdic.append(fragdic)
        # c.clear()
        loss, molwt = listtodic(maxtree[i]['loss'])
        fragdic = dict(source=id_f, molecularFormula=loss, lossformula=molwt, target=id_c)
        Lossdic.append(fragdic)
    return Fragdic, Lossdic


def listtodic(mf):
    m = ['C', 'N', 'O', 'H', 'S']
    num = [3, 0, 1, 2, 4]
    count_dist1 = dict()
    real = [1.00783, 12.00000, 14.00307, 15.99491, 31.97207]  # H C N O S
    calmass = np.multiply(np.array(mf), np.array(real))
    molwt = np.sum(calmass)
    for i in m:
        count_dist1[i] = 0
        # H C N O S
    for i in range(len(mf)):
        if mf[i]:
            idx = num[i]
            count_dist1[m[idx]] = mf[i]
    return count_dist1, molwt
