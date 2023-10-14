# from MolTree import FragmentTree
import xlrd
import numpy as np
from scipy import interpolate
# import matplotlib.pyplot as plt
import Utility
from MolTree import FragmentTree
import csv
import json
from TeFT_predict import Transformer
from TeFT_predict import greedy_decoder
import torch
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem import MACCSkeys
import Smiles_tree

####################################################
# 1.Obtain the original mass spectrometry csv file #
####################################################

Measure_instrument = 'Kyrin3'
name = 'galangin.csv'
Mass = []
Intensity = []
if Measure_instrument == 'LCQ':
    data = xlrd.open_workbook(name)
    table = data.sheet_by_index(0)
    Mass = table.col_values(0)
    Intensity = table.col_values(1)
    StartNum = 0
    for i in Mass:
        if type(i) == float:
            StartNum = Mass.index(i)
            break
    Mass = Mass[StartNum::]
    Intensity = Intensity[StartNum::]
elif Measure_instrument == 'Kyrin3':
    with open(name, encoding='utf-8-sig') as f:
        for row in csv.reader(f, skipinitialspace=True):
            Mass.append(float(row[0]))
            Intensity.append(float(row[1]))
    f.close()
###########################################################
# 2.Perform mass spectrometry calibration  (if necessary) #
###########################################################
calibration = False
if calibration:
    adjust_node = [100, 150, 200, 250, 300]
    adjust_mz = []
    for i in adjust_node:
        num = Utility.index_number(Mass, i)
        adjust_mz.append(num)
    MZ = np.empty(shape=0)
    INT = np.empty(shape=0)
    for i in range(len(adjust_mz) - 1):
        mz = np.arange(adjust_node[i], adjust_node[i + 1], 0.001)
        adj_mz = np.array(Mass[adjust_mz[i]:adjust_mz[i + 1]])
        adj_amp = np.array(Intensity[adjust_mz[i]:adjust_mz[i + 1]])
        adj_mz, adj_amp = Utility.remove_non_increasing_pairs(adj_mz, adj_amp)
        mz_interp1 = interpolate.splev(mz, interpolate.splrep(adj_mz, adj_amp), der=0)
        # adj_str = str(adjust_node[i]) + "-" + str(adjust_node[i + 1]) + "-" + Measure_instrument + "-230K-4.17.npy"
        adj_str = '200-250-Kyrin3-230K-4.7.npy'
        K = np.load(adj_str)
        After_adj = np.convolve(mz_interp1, K, 'same')
        In = max(After_adj) / max(adj_amp)
        After_adj = [item / In for item in After_adj]
        MZ = np.concatenate((MZ, mz))
        INT = np.concatenate((INT, After_adj))
    print(max(INT))
    INT = np.array(Intensity)
    MZ = np.array(Mass)
    amp = Utility.find_appltitude(INT, 5)
if not calibration:
    INT = np.array(Intensity)
    MZ = np.array(Mass)

# #############
# 3.Find peak #
###############
# Spec_list_amp, Spec_list_mz = (list(t) for t in zip(*sorted(zip(INT, MZ), reverse=True)))
z = Utility.find_appltitude(INT, 0.1)
Spec_list_mz = []
Spec_list_amp = []
for i in z:
    Spec_list_mz.append(MZ[i])
    Spec_list_amp.append(INT[i])
Spec_list_amp, Spec_list_mz = (list(t) for t in zip(*sorted(zip(Spec_list_amp, Spec_list_mz), reverse=True)))
Spec_list_mz = Spec_list_mz[:5]
Spec_list_mz = sorted(Spec_list_mz)
Spec_list_Mz = Utility.add_subtract(Spec_list_mz)
print('mz!')

###################################
# 4.get FT #
###################################
mode = FragmentTree.find_child(Spec_list_mz, 0.04, 0.5)
array = FragmentTree.edge_weight(Spec_list_mz, mode, 0.0005)
Score, Tree = FragmentTree.Generate_Tree(array, Spec_list_mz)
print('moltree!')
###################################
# 5.Use Transformer to predict  #
###################################
config = [Spec_list_Mz, 'C1=CC=C(C=C1)C2=C(C(=O)C3=C(C=C(C=C3O2)O)O)O', 100, 'Galangin2.json']
dict1 = [
    '<PAD>', '<SOS>', '<EOS>',
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    'H', 'C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Br', 'I',
    '[', ']', '(', ')', '/', '\\', ',', '%', '+', '=', '-', '@', '#', '.',
    'h', 'c', 'n', 'o', 'p', 's', 'i',
]
k = np.arange(3, 50000).tolist()
zz = ['<PAD>'] + ['<SOS>'] + ['<EOS>'] + k
kk = np.arange(0, len(dict1))
vocab = dict(zip(dict1, kk))
vocab_y = dict(zip(vocab.values(), vocab.keys()))
zidian_x = dict(zip(zz, np.arange(0, 50000)))
mz_dict = np.arange(0, 50000).tolist()
vocab_smi = dict(zip(dict1, np.arange(0, len(dict1))))
vocab_smi_rev = dict(zip(vocab_smi.values(), vocab_smi.keys()))
vocab_smi_size = len(vocab_smi)

vocab_mz = dict(zip(mz_dict, mz_dict))
vocab_mz_rev = dict(zip(vocab_mz.values(), vocab_mz.keys()))
mz_vocab_size = len(vocab_mz)
smi_vocab_size = len(vocab_smi)

device = 'cuda'
model = Transformer().to(device)
model.load_state_dict(torch.load(
    './final_1.pth'))  #
judge = 0
SS = []
jihe = []
prid_num = config[2]
print(config[3])
# for prid_num in range(prid_num):
enc_input2 = []
dec_input2 = []
real_smiles = []
for i in range(1):
    mz = config[0]
    smiles = config[1]
    mz = [float(x) for x in mz]
    mz = [int(100 * x) for x in mz]
    if max(mz) > 50000:
        continue
    if len(smiles) < 100:
        for j in range(len(smiles)):
            if judge == 1:
                judge = 0
            else:
                if smiles[j] == 'C' and j < len(smiles) - 1:
                    if smiles[j + 1] == 'l':
                        SS.append('Cl')
                        judge = 1
                        continue
                if smiles[j] == 'B' and j < len(smiles) - 1:
                    if smiles[j + 1] == 'r':
                        SS.append('Br')
                        judge = 1
                        continue
                SS.append(smiles[j])
        SS = ['<SOS>'] + SS + ['<EOS>']
        for j in range(0, 100):
            mz.append(0)
            SS.append('<PAD>')
        enc_input = [[vocab_mz[n] for n in mz]]
        dec_input = [[vocab_smi[n] for n in SS]]
        enc_input2.append(enc_input[0][0:100])
        dec_input2.append(dec_input[0][0:101])
        real_smiles.append(smiles)
        SS = []
enc_inputs = torch.LongTensor(enc_input2).to(device)
dec_inputs = torch.LongTensor(dec_input2).to(device)
zero = torch.zeros(100).to(device)
print(len(enc_inputs))
SIM = []
ans = []
sm = ''
# k = 0
for i in range(len(enc_inputs)):
    for j in range(prid_num):
        greedy_dec_predict = greedy_decoder(model, enc_inputs[i].view(1, -1).to(device), sm,
                                            start_symbol=vocab_smi['<SOS>'])
        predictsmiles = [vocab_smi_rev[n.item()] for n in greedy_dec_predict.squeeze()]
        strr = ''
        psm = strr.join(predictsmiles)
        psm = psm.replace('<PAD>', '')
        c = psm.find('<EOS>')
        realsmiles = real_smiles[i]
        predictmol = Chem.MolFromSmiles(psm[0:c])
        realmol = Chem.MolFromSmiles(str(realsmiles))
        if predictmol:
            pre_fp_normal = Chem.RDKFingerprint(predictmol)
            real_fp_normal = Chem.RDKFingerprint(realmol)
            pre_fp_MACCS = MACCSkeys.GenMACCSKeys(predictmol)
            real_fp_MACCS = MACCSkeys.GenMACCSKeys(realmol)
            ecfp4_pre = AllChem.GetMorganFingerprint(predictmol, 2)
            ecfp4_real = AllChem.GetMorganFingerprint(realmol, 2)
            fcfp4_pre = AllChem.GetMorganFingerprint(predictmol, 2, useFeatures=True)
            fcfp4_real = AllChem.GetMorganFingerprint(realmol, 2, useFeatures=True)
            simi_ecfp4 = DataStructs.DiceSimilarity(ecfp4_pre, ecfp4_real)
            simi_fcfp4 = DataStructs.DiceSimilarity(fcfp4_pre, fcfp4_real)
            simi_norm = DataStructs.FingerprintSimilarity(pre_fp_normal, real_fp_normal)
            simi_MACCS_dice = DataStructs.FingerprintSimilarity(pre_fp_MACCS, real_fp_MACCS,
                                                                metric=DataStructs.DiceSimilarity)
            simi_MACCS = DataStructs.FingerprintSimilarity(pre_fp_MACCS, real_fp_MACCS)
            sim_tomitosim = max(simi_norm, simi_MACCS)
            MAX_SIM = max(simi_ecfp4, simi_fcfp4, simi_norm, simi_MACCS_dice, simi_MACCS)
            print('Predict:', i)
            print('predictsmiles:', psm[0:c])
            print('realsmiles:', realsmiles)
            print(' ')
            DITT = dict(num=i, predict=psm[0:c], real=str(realsmiles), similar=MAX_SIM, tomitosim=sim_tomitosim)
            jihe.append(DITT)
            SIM.append(MAX_SIM)
    if SIM:
        max_sim = max(SIM)
        pos = SIM.index(max_sim)
        res = dict(num=i, predict=jihe[pos]['predict'], real=jihe[pos]['real'], similar=max_sim)
        print('sim:', max_sim)
        print('smiles:', jihe[pos]['predict'])
        ans.append(jihe)
    SIM = []
    jihe = []
final = json.dumps(ans)
fileOb = open(config[3], 'w')
fileOb.write(final)
f.close()
print('predict!')
################################
# 6.Get SMILES tree and result #
################################
with open(config[3], 'r') as f:
    ans = json.load(f)
Tree_Simi_Score = []
Tree_Simi_node_result = []
Tree_Simi_loss_result = []
All_data = []
Tree_large_Simi_result = []
ans = ans[0]
Tree = Tree[3:4]
MaxTree = []
for i in range(len(Tree)):
    for j in range(len(ans)):
        smiles = ans[j]['predict']
        Smiledic, loss = Smiles_tree.predict_SMILESTree(smiles)
        realtree, realloss = Smiles_tree.moleculartreetodic_Prim(Tree[i])
        simnode, simloss, score = Smiles_tree.comparetree(Smiledic, loss, realtree, realloss)
        Tree_Simi_node_result.append(simnode)
        Tree_Simi_loss_result.append(simloss)
        Tree_Simi_Score.append(score)
    combined_lists = list(zip(Tree_Simi_Score, Tree_Simi_node_result))
    Result = sorted(combined_lists, key=lambda x: x[0],reverse=True)
    Tree_large_Simi_result.append(Result)
    Tree_Simi_Score = []
    Tree_Simi_node_result = []
final = json.dumps(Tree_large_Simi_result)
fileOb = open('result.json', 'w')
fileOb.write(final)
f.close()
print('end!')

