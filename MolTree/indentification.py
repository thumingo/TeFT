# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 18:56:05 2020

@author: Administrator
"""

# *********
# 分子式识别
# *********

import matplotlib.pyplot as plt
import numpy as np
import xlrd
from rdkit.Chem.rdRGroupDecomposition import Isotope
from xlrd import xldate_as_tuple
import datetime
# import Isotope
import math
from sympy import *
from scipy.special import *
import scipy.signal as signal
from sklearn import preprocessing
import time

'''
xlrd中单元格的数据类型
数字一律按浮点型输出，日期输出成一串小数，布尔型输出0或1，所以我们必须在程序中做判断处理转换
成我们想要的数据类型
0 empty,1 string, 2 number, 3 date, 4 boolean, 5 error
'''


class ExcelData():
    # 初始化方法
    def __init__(self, data_path, sheetname):
        # 定义一个属性接收文件路径
        self.data_path = data_path
        # 定义一个属性接收工作表名称
        self.sheetname = sheetname
        # 使用xlrd模块打开excel表读取数据
        self.data = xlrd.open_workbook(self.data_path)
        # 根据工作表的名称获取工作表中的内容（方式①）
        self.table = self.data.sheet_by_name(self.sheetname)
        # 根据工作表的索引获取工作表的内容（方式②）
        # self.table = self.data.sheet_by_name(0)
        # 获取第一行所有内容,如果括号中1就是第二行，这点跟列表索引类似
        self.keys = self.table.row_values(0)
        # 获取工作表的有效行数
        self.rowNum = self.table.nrows
        # 获取工作表的有效列数
        self.colNum = self.table.ncols

    # 定义一个读取excel表的方法
    def readExcel(self):
        # 定义一个空列表
        datas = []
        for i in range(1, self.rowNum):
            # 定义一个空字典
            sheet_data = {}
            for j in range(self.colNum):
                # 获取单元格数据类型
                c_type = self.table.cell(i, j).ctype
                # 获取单元格数据
                c_cell = self.table.cell_value(i, j)
                if c_type == 2 and c_cell % 1 == 0:  # 如果是整形
                    c_cell = int(c_cell)
                elif c_type == 3:
                    # 转成datetime对象
                    date = datetime.datetime(*xldate_as_tuple(c_cell, 0))
                    c_cell = date.strftime('%Y/%d/%m %H:%M:%S')
                elif c_type == 4:
                    c_cell = True if c_cell == 1 else False
                sheet_data[self.keys[j]] = c_cell
                # 循环每一个有效的单元格，将字段与值对应存储到字典中
                # 字典的key就是excel表中每列第一行的字段
                # sheet_data[self.keys[j]] = self.table.row_values(i)[j]
            # 再将字典追加到列表中
            datas.append(sheet_data)
        # 返回从excel中获取到的数据：以列表存字典的形式返回
        return datas


def change(M, a):
    c = []
    # rule1:元素数量限制
    if M < 500:
        H_n = 72
        C_n = 29
        N_n = 10
        O_n = 18
        # F_n=15
        S_n = 7
    if 500 < M <= 1000:
        H_n = 126
        C_n = 66
        N_n = 25
        O_n = 27
        # F_n=16
        S_n = 8
    else:
        H_n = np.inf
        C_n = np.inf
        N_n = np.inf
        O_n = np.inf
        # F_n=np.inf
        S_n = np.inf

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
                # rule2
                if ((i + m) % 2) != 0:
                    continue
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
                        if 1 * i + 4 * j + 3 * m + 2 * n + 2 * p < 2 * (i + j + m + n + p) - 1:
                            continue
                        if ((1 * i + 4 * j + 3 * m + 2 * n + 2 * p) % 2) != 0:
                            continue
                        if i * a[0] + j * a[1] + m * a[2] + n * a[3] + p * a[4] == M:
                            c.append([i, j, m, n, p])
    return c


start = time.time()
# *********
# 读取数据
# *********
data_path = "C13H18O2.xlsx"
sheetname = "Sheet1"
get_data = ExcelData(data_path, sheetname)
datas = get_data.readExcel()

# ***********
# 数据预处理
# ***********
mass = []
intensity = []
for i in datas:
    mass.append(i['Mass'] - 1.00782500000000)  # MH+形式存在-1，MH-形式存在+1
    intensity.append(i['Intensity'])
# 归一化
intensity = intensity / np.max(intensity)
# 选出指定范围的数据
mass_temp = []
intensity_temp = []
for i in range(len(mass)):
    if 204 < mass[i] < 208:
        if intensity[i] >= 0.01:
            mass_temp.append(mass[i])
            intensity_temp.append(intensity[i])
mass_temp = np.array(mass_temp)
intensity_temp = np.array(intensity_temp)
# 极值点（峰值）
intensity_final = intensity_temp[signal.argrelextrema(intensity_temp, np.greater)]
mass_final = mass_temp[signal.argrelextrema(intensity_temp, np.greater)[0]]

# ***********
# 计算候选分子式
# ***********
M = mass_final[np.argmax(intensity_final)]
M = int(np.around(M, decimals=0))
ele = [1, 12, 14, 16, 32]  # H,C,N,O
comp = change(M, ele)

# **************
# 模拟质谱同位素
# **************
spset_cut = []
for j in range(len(comp)):
    formular = comp[j]
    sp_cut = Isotope.isotope(formular)
    spset_cut.append(sp_cut)
#      
# monoisotopic=[]
# for j in spset_cut:
#    monoisotopic.append(j[0,-1]) 

# **********
# 贝叶斯估计
# **********
# 仪器参数
deta_int = 0.03
sd = 2
###正态分布标准差
P_D = []
P_Mm = []
P_fp = []
for j in range(len(spset_cut)):
    p_fp = 1
    p_Mm = 1
    for i in range(len(intensity_final)):
        f = intensity_final[i]
        M0 = mass_final[0]
        M = mass_final[i]
        p = spset_cut[j][1, i]
        m0 = spset_cut[j][0, 1]
        m = spset_cut[j][0, i]
        # sd_mass=sd*10**(-6)*(1.5-0.5*f)
        sd_mass = 0.5 * (1.5 - 0.5 * f)  # 假定的仪器标准差为1
        sd_int = math.log((f + deta_int) / f)
        p_fp = p_fp * math.erfc((abs(math.log(f / p))) ** 0.5 / (2 ** 0.5 * sd_int))
        if i == 0:
            x = M - m
        else:
            x = (M - M0) - (m - m0)
        p_Mm = p_Mm * math.erfc((abs(x)) ** 0.5 / (2 ** 0.5 * sd_mass))
    P_D.append(p_fp * p_Mm)
    P_Mm.append(p_Mm)
    P_fp.append(p_fp)
t = (time.time() - start)
