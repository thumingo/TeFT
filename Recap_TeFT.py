import sys
import weakref
from rdkit import Chem
from rdkit.Chem import rdChemReactions as Reactions
from rdkit.Chem import Descriptors

reactionDefs = (
    # "[#7;+0;D2,D3:1]!@C(!@=O)!@[#7;+0;D2,D3:2]>>*[#7:1].[#7:2]*",  # urea
    # "[C;!$(C([#7])[#7]):1](=!@[O:2])!@[#7;+0;!D1:3]>>*[C:1]=[O:2].*[#7:3]",  # amide
    # "[C:1](=!@[O:2])!@[O;+0:3]>>*[C:1]=[O:2].[O:3]*",  # ester
    # "[C:1](=!@[O:2])!@[O;+0:3]>>[C+:1]=[O:2]",
    # "[N;!D1;+0;!$(N-C=[#7,#8,#15,#16])](-!@[*:1])-!@[*:2]>>*[*:1].[*:2]*",  # amines
    # "[N;!D1](!@[*:1])!@[*:2]>>*[*:1].[*:2]*", # amines
    #
    # # again: what about aromatics?
    # "[#7;R;D3;+0:1]-!@[*:2]>>*[#7:1].[*:2]*",  # cyclic amines
    # "[#6:1]-!@[O;+0]-!@[#6:2]>>[#6:1]*.*[#6:2]",  # ether
    # "[C:1]=!@[C:2]>>[C:1]*.*[C:2]",  # olefin
    # "[n;+0:1]-!@[C:2]>>[n:1]*.[C:2]*",  # aromatic nitrogen - aliphatic carbon
    # "[O:3]=[C:4]-@[N;+0:1]-!@[C:2]>>[O:3]=[C:4]-[N:1]*.[C:2]*",  # lactam nitrogen - aliphatic carbon
    # "[c:1]-!@[c:2]>>[c:1]*.*[c:2]",  # aromatic carbon - aromatic carbon
    # aromatic nitrogen - aromatic carbon *NOTE* this is not part of the standard recap set.
    # "[A:1]=!@[A:2]-!@[A:3]-!@[A:4]-!@[A:5]>>[A:1]-!@[A:2]=!@[A:3].[A:4]=!@[A:5]",
    # 通用裂解规律
    # "[n;+0:1]-!@[c:2]>>[n:1].[c:2]",
    "[#7;+0;D2,D3:1]-!@[S:2](=[O:3])=[O:4]>>[#7:1].[S:2](=[O:3])=[O:4]",
    "[C:1]=!@[C:2]>>[C+:1].[C+:2]",
    "[#6H2:1]-!@[#6H1:2]-[O;-1]>>[#6H1:1]=,:[#6+:2]",
    "[#6H2:1]-!@[#6H2:2]-[O;-1]>>[#6H1:1]=,:[#6H1+:2]",
    "[#6H2:1]-!@[#6H1:2]-[O]>>[#6H1:1]=,:[#6H1:2]",
    "[#6H2:1]-!@[#6H2:2]-[O]>>[#6H1:1]=,:[#6H2:2]",
    "[#6H1:1]-!@[#6H1:2]-[O]>>[#6H0:1]=,:[#6H1:2]",
    "[#6H1:1]-!@[#6H1:2]-[O]>>[#6H0:1]=,:[#6H1:2]",
    # "[C;+0:1]-!@[O;+0,+0:2]>>[C;+1:1].[O;-1:2]",
    "[C;+0:1]-!@[O;+0,+0:2]>>[C;+0:1].[O;+0,+0:2]",
    "[c:1]-[c:2](=O)-[c:3]>>[c:1]-[c:3]",
    # "[c;+0:1]-!@[O;+0,+0:2]>>[c;+1:1].[O;-1:2]",
    "[CH;+0:1]-[OH:2]>>[CH0;+1:1].[O;-1:2]",
    "[#6;+0:1]-!@[N;+0,+0:2]>>[#6+1:1].[N;+1:2]",
    "[C;0:1]-!@[O;-1,+0:2]>>[C;+1:1].[O;-1:2]",
    "[C;0:1]-!@[O;-1,+0:2]>>[C;+1:1].[O;0:2]",
    "[C;+1:1]-!@[N;+0,+0:2]>>[C;+2:1].[N;+1:2]",
    "[C;+1:1]-!@[C;+1,+0:2]>>[C;+1:1].[C;+1:2]",
    "[C;+0:1]-!@[N;+1,+0:2]>>[C;+1:1].[N;+2:2]",
    "[C;+0:1]-!@[C;+0:2]>>[C;+1:1].[C;+1:2]",
    "[C;+0:1]-!@[C;+0:2]>>[C;+0:1].[C;+0:2]",
    "[C;+0:1]-!@[C;+1:2]>>[C;+1:1].[C;+2:2]",
    "[C;+0:1]-!@[S;+0:2]>>[C;+1:1].[S;+1:2]",
    "[c;+0:1]-[O;+0:2]>>[c;+0:1].[O;-1:2]",
    "[c;+0:1]-[C;+0:2]>>[c;+0:1].[C;+0:2]",
    # "[c;+0:1]-[O;+0:2]>>[c;+1:1].[O;-1:2]",
    # # 黄酮醇类化合物裂解规律
    # #"[#6:1]@[#8:2]@[#6:3]>>[#6:1]-[#6:3].[#8H0:2]",
    "[#6:1]@[#6:2](=[O:3])@[#6:4]>>[#6:1]-[#6:4].[#6H0:2](=[O:3])",
    "[c:1]-[C:2](=[C:3]-[O:4])-[c:5]:[c:6]:[c:7]:[c:8](-[O:9])@[c:10]@[c:11]>>[c:1]-[C:2]=[C]1[C]=[C]-[C](=[O])-[C:10]=[C:11]1.[C]-[O]",
    "[c:1]@[Cr4:2]@[Cr4:3]@[c:4]>>[C:2]-[C:3].[c:1]-[c:4]",
    "[Cr4:1]-[c:2]>>[Cr4:1].[c:2]",
    # sulphonamide
    # 芪类化合物裂解规律
    "[c:1]-[C:2]=[C:3]-[c:4]>>[C]=[C]-[C:2]=[C:3]-[c:4]",
    "[cr6:1]@[cr6:2](-[O:7])@[cr6:3]@[cr6:4]@[cr6:5]@[cr6:6]>>[C:1]-[C:3]-[C:4]=[C:5]-[C:6]",
    "[C:1]1=[C:3]-[C:4]-[C:5]=[C:6]1>>[C:1]1-[C:3]=[C:6]1",

)

reactions = tuple([Reactions.ReactionFromSmarts(x) for x in reactionDefs])


def get_exact_mass(mol):
    m = ['H', 'C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Br', 'I']
    mass = [1.00783, 12.00000, 14.00307, 15.99491, 18.99840, 30.97376, 31.97207, 34.96885, 78.91834, 126.90450]
    atoms_list = []
    actuall_mass = 0
    mol = Chem.AddHs(mol)
    atoms = mol.GetAtoms()
    for at in atoms:
        atom = at.GetSymbol()
        idx = m.index(atom)
        actuall_mass = actuall_mass + mass[idx]
        atoms_list.append(atom)
    return actuall_mass


class RecapHierarchyNode1(object):
    """ This class is used to hold the Recap hiearchy
    """
    mol = None
    children = None
    parents = None
    smiles = None

    def __init__(self, mol):
        self.mol = mol
        self.children = {}
        self.parents = {}

    def GetAllChildren(self):
        " returns a dictionary, keyed by SMILES, of children "
        res = {}
        for smi, child in self.children.items():
            res[smi] = child
            child._gacRecurse(res, terminalOnly=False)
        return res

    def GetLeaves(self):
        " returns a dictionary, keyed by SMILES, of leaf (terminal) nodes "
        res = {}
        for smi, child in self.children.items():
            if not len(child.children):
                res[smi] = child
            else:
                child._gacRecurse(res, terminalOnly=True)
        return res

    def getUltimateParents(self):
        """ returns all the nodes in the hierarchy tree that contain this
            node as a child
        """
        if not self.parents:
            res = [self]
        else:
            res = []
            for p in self.parents.values():
                for uP in p.getUltimateParents():
                    if uP not in res:
                        res.append(uP)
        return res

    def _gacRecurse(self, res, terminalOnly=False):
        for smi, child in self.children.items():
            if not terminalOnly or not len(child.children):
                res[smi] = child
            child._gacRecurse(res, terminalOnly=terminalOnly)

    def __del__(self):
        self.children = {}
        self.parents = {}
        self.mol = None


def RecapDecompose_for1(mol, allNodes=None, minFragmentSize=0, onlyUseReactions=None):
    """ returns the recap decomposition for a molecule """
    mSmi = Chem.MolToSmiles(mol, 1)

    if allNodes is None:
        allNodes = {}
    if mSmi in allNodes:
        return allNodes[mSmi]

    res = RecapHierarchyNode1(mol)
    res.smiles = mSmi
    activePool = {mSmi: res}
    allNodes[mSmi] = res
    count = 0
    while activePool:
        nSmi = next(iter(activePool))
        node = activePool.pop(nSmi)
        if not node.mol:
            continue
        for rxnIdx, reaction in enumerate(reactions):
            if onlyUseReactions and rxnIdx not in onlyUseReactions:
                continue
            # print '  .',nSmi
            # print '         !!!!',rxnIdx,nSmi,reactionDefs[rxnIdx]
            ps = reaction.RunReactants((node.mol,))
            # print '    ',len(ps)
            if ps:
                for prodSeq in ps:
                    seqOk = True
                    # we want to disqualify small fragments, so sort the product sequence by size
                    # and then look for "forbidden" fragments
                    tSeq = [(prod.GetNumAtoms(onlyExplicit=True), idx)
                            for idx, prod in enumerate(prodSeq)]
                    tSeq.sort()
                    ts = [(x, prodSeq[y]) for x, y in tSeq]
                    prodSeq = ts
                    for nats, prod in prodSeq:
                        try:
                            Chem.SanitizeMol(prod)
                        except Exception:
                            continue
                        pSmi = Chem.MolToSmiles(prod, 1)
                        if minFragmentSize > 0:
                            nDummies = pSmi.count('*')
                            if nats - nDummies < minFragmentSize:
                                seqOk = False
                                break
                        # don't forget after replacing dummy atoms to remove any empty
                        # branches:
                        elif count > 1500:
                            seqOk = False
                            break
                        elif pSmi.replace('*', '').replace('()', '') in ('', 'C', 'CC','CCC'):
                            seqOk = False
                            break
                        prod.pSmi = pSmi
                        prod.Molwt = get_exact_mass(Chem.MolFromSmiles(pSmi))
                        prod.broke = True
                        prod.dd = count

                    if seqOk:
                        for nats, prod in prodSeq:
                            if hasattr(prod, 'pSmi'):
                                pSmi = prod.pSmi
                            # print '\t',nats,pSmi
                                if pSmi not in allNodes:
                                    pNode = RecapHierarchyNode1(prod)
                                    pNode.smiles = pSmi
                                    pNode.parents[nSmi] = weakref.proxy(node)
                                    node.children[pSmi] = pNode
                                    #
                                    # if get_exact_mass(Chem.MolFromSmiles(pSmi)) > 50:
                                    activePool[pSmi] = pNode
                                    allNodes[pSmi] = pNode
                                else:
                                    pNode = allNodes[pSmi]
                                    Moll = pNode.mol
                                    Moll.Molwt = get_exact_mass(Chem.MolFromSmiles(pSmi))
                                    # Moll.broke = False
                                    Moll.dd = count
                                    pNode.parents[nSmi] = weakref.proxy(node)
                                    node.children[pSmi] = pNode
                        # print '                >>an:',allNodes.keys()
                    count += 1
    return res
