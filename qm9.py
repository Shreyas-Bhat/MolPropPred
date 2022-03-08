# -*- coding: utf-8 -*-


## import libraries
import torch
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
from deepchem.feat.graph_features import *
from rdkit.Chem import AllChem
from IPython.display import display
from ipywidgets import FloatProgress
import deepchem as dc
from rdkit import Chem, DataStructs
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from collections import OrderedDict
import matplotlib.pyplot as plt
import sys
import os


"""## Parameters """

# N = 10000# number of molecules in the dataset
N = 133885
D = 75     # hidden dimension of each atom
E = 6      # dimension of each edge
T = 3      # number of time steps the message phase will run for
P = 32     # dimensions of the output from the readout phase, the penultimate output before the target layer
V = 12     # dimensions of the molecular targets or tasks

# TRAIN_SIZE = 8000
TRAIN_SIZE = 100000
# VALID_SIZE = 1000
VALID_SIZE = 10000
# TEST_SIZE  = 1000
TEST_SIZE = 23885
BATCH_SIZE = 20
NUM_EPOCHS = 3

DF = np.random.uniform(0.01, 1)
LR = np.random.uniform(1e-5, 5e-4)
LF = DF * LR
# from google.colab import drive
# drive.mount('/content/gdrive')

print('decay factor          : %.6f' % (DF))
print('initial learning rate : %.6f' % (LR))
print('final learning rate   : %.6f' % (LF))

# from google.colab import files
# uploaded = files.upload()

qm9 = pd.read_csv('data/qm9.csv')

qm9.head()

len(qm9)

# list = []
# for i in range(20001,30001):
#     list.append(i)
# qm9 = qm9.iloc[list]

# len(list)

chemical_accuracy_dict = {'mu': [0.1],
                          'alpha': [0.1],
                          'homo': [0.043],
                          'lumo': [0.043],
                          'gap': [0.043],
                          'r2': [1.2],
                          'zpve': [0.0012],
                          'u0': [0.043],
                          'u298': [0.043],
                          'h298': [0.043],
                          'g298': [0.043],
                          'cv': [0.50]}

chemical_accuracy = pd.DataFrame(chemical_accuracy_dict)

chemical_accuracy

structures = ['smiles']
tasks = ['mu', 'alpha', 'homo', 'lumo', 'gap', 'r2',
         'zpve', 'u0', 'u298', 'h298', 'g298', 'cv']

X = qm9[structures]
y = qm9[tasks]

y.head()

y.describe()

scaler = StandardScaler()

y = pd.DataFrame(scaler.fit_transform(y), index=y.index, columns=y.columns)

y.describe()


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=143)
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=VALID_SIZE, random_state=143)

dfn = pd.DataFrame(columns=['col1', 'col2', 'col3', 'col4', 'col5',
                   'col6', 'col7', 'col8', 'col9', 'col10', 'col11', 'col12'])

dfn

"""## Code"""


def batch_mse_loss(pred, true):
    return F.mse_loss(pred, true) / BATCH_SIZE


def valid_mse_loss(pred, true):
    return (F.mse_loss(pred, true)).detach() / VALID_SIZE


scale_batch_to_train = BATCH_SIZE / TRAIN_SIZE

# what does this do?


class MasterEdge(nn.Module):

    def __init__(self):
        super(MasterEdge, self).__init__()

        self.l1 = nn.Linear(D, P)
        nn.init.kaiming_normal_(self.l1.weight)
        self.l2 = nn.Linear(P, 2*E)
        nn.init.kaiming_normal_(self.l2.weight)
        self.l3 = nn.Linear(2*E, E)
        nn.init.kaiming_normal_(self.l3.weight)

    def forward(self, x):
        return F.elu(self.l3(F.elu(self.l2(F.elu(self.l1(x))))))


master_edge_learner = MasterEdge()

# def dfs(adjacency_matrix,visited_array,i):
#   visited_array[i] = 1
#   for j in range(len(visited_array)):
#     if(!visited_array[j] && adjacency_matrix[i][j]==1):
#       dfs(adjacency_matrix,visited_matrix,j)

# printing all cycles in an undirected graph
# Function to mark the vertex with different colors for different cycles


def dfs_cycle(u, p, color, mark, par, cyclenumber, g):
    # already (completely) visited vertex.
    if(color[u] == 2):
        return
    # seen vertex, but was not completely visited -> cycle detected.
    # backtrack based on parents to find the complete cycle.
    if(color[u] == 1):
        cyclenumber = cyclenumber + 1
        cur = p
        mark[cur] = cyclenumber
        # backtrack the vertex which are
        # in the current cycle thats found
        while(cur != u):
            # print(cur,u)
            cur = par[cur]
            mark[cur] = cyclenumber
        return

    par[u] = p
    # partially visited.
    color[u] = 1
    # simple dfs on graph
    for j in range(len(g[u])):
        if(g[u][j][1] == par[u]):
            continue
        dfs_cycle(g[u][j][1], u, color, mark, par, cyclenumber, g)
    # completely visited
    color[u] = 2

# function to print all cycles


def cycles_list_function(edges_1, mark, cycle_number, cycles):
    # push the edges that into the cycle adjacency list
    for i in range(edges_1):
        if(mark[i] != 0):
            # print(mark[i],i)
            cycles[mark[i]].append(i)
    # for i in range(cycle_number+1):
        # print(cycles[i])


def construct_multigraph(smile):
    g = OrderedDict({})
    h = OrderedDict({})
    #h[-1] = 0
    molecule = Chem.MolFromSmiles(smile)
    #mol_matrix = [['0','1','0','0','0'],['1','0','2','0','1'],['0','2','0','1','1'],['0','0','1','0','0'],['0','1','1','0','0']]
    for i in range(molecule.GetNumAtoms()):
        atom_i = molecule.GetAtomWithIdx(i)
        atom_i_featurized = dc.feat.graph_features.atom_features(atom_i)
        atom_i_tensorized = torch.FloatTensor(atom_i_featurized).view(1, D)
        h[i] = atom_i_tensorized
        #h[-1] += h[i]
        master_edge = master_edge_learner(h[i])
        g.setdefault(i, [])
        # .append((master_edge, -1))
        #g.setdefault(-1, [])
        # .append((master_edge, i))
        for j in range(molecule.GetNumAtoms()):
            bond_ij = molecule.GetBondBetweenAtoms(i, j)
            if bond_ij:  # bond_ij is None when there is no bond.
                #atom_j = molecule.GetAtomWithIdx(j)
                #atom_j_featurized = dc.feat.graph_features.atom_features(atom_j)
                #atom_j_tensorized = torch.FloatTensor(atom_j_featurized).view(1, 75)
                bond_ij_featurized = dc.feat.graph_features.bond_features(
                    bond_ij).astype(int)
                bond_ij_tensorized = torch.FloatTensor(
                    bond_ij_featurized).view(1, E)
                g.setdefault(i, []).append((bond_ij_tensorized, j))
    # novelty
    edges = molecule.GetNumBonds()
    mark = [0]*molecule.GetNumAtoms()
    par = [-2]*molecule.GetNumAtoms()
    color = [0]*molecule.GetNumAtoms()
    cyclenumber = 0
    cycles_list = [[], [], []]
    dfs_cycle(0, -1, color, mark, par, cyclenumber, g)
    cycles_list_function(molecule.GetNumAtoms(), mark, max(mark), cycles_list)
    # print(max(mark))
    num_of_atoms = molecule.GetNumBonds()
    for i in range(len(cycles_list)):
        if(len(cycles_list[i]) >= 3):
            h[len(h)] = 0
            for j in range(len(cycles_list[i])):
                # print(len(h)-1)
                h[len(h)-1] += h[cycles_list[i][j]]  # semi master node
                master_edge = master_edge_learner(h[cycles_list[i][j]])
                g[cycles_list[i][j]].append((master_edge, len(h)-1))
                g.setdefault(
                    len(h)-1, []).append((master_edge, cycles_list[i][j]))
    h[-1] = 0
    # print(cycles_list)
    for x in range(num_of_atoms, len(h)-1):
        # print(len(h))
        h[-1] += h[x]
        master_edge = master_edge_learner(h[x])
        g[x].append((master_edge, -1))
        g.setdefault(-1, []).append((master_edge, x))
    return g, h


class EdgeMappingNeuralNetwork(nn.Module):

    def __init__(self):
        super(EdgeMappingNeuralNetwork, self).__init__()

        self.fc1 = nn.Linear(E, D)
        nn.init.kaiming_normal_(self.fc1.weight)
        self.fc2 = nn.Linear(1, D)
        nn.init.kaiming_normal_(self.fc2.weight)

    def f1(self, x):
        return F.elu(self.fc1(x))

    def f2(self, x):
        return F.elu(self.fc2(x.permute(1, 0)))

    def forward(self, x):
        return self.f2(self.f1(x))


class MessagePhase(nn.Module):

    def __init__(self):
        super(MessagePhase, self).__init__()
        self.A = EdgeMappingNeuralNetwork()
        self.U = {i: nn.GRUCell(D, D) for i in range(T)}

    def forward(self, smile):

        g, h = construct_multigraph(smile)
        g0, h0 = construct_multigraph(smile)

        for k in range(T):
            h = OrderedDict(
                {
                    v:
                    self.U[k](
                        sum(torch.matmul(h[w], self.A(e_vw))
                            for e_vw, w in en),
                        h[v]
                    )
                    for v, en in g.items()
                }
            )

        return h, h0


class Readout(nn.Module):

    def __init__(self):
        super(Readout, self).__init__()

        self.i1 = nn.Linear(2*D, 2*P)
        nn.init.kaiming_normal_(self.i1.weight)
        self.i2 = nn.Linear(2*P, P)
        nn.init.kaiming_normal_(self.i2.weight)

        self.j1 = nn.Linear(D, P)
        nn.init.kaiming_normal_(self.j1.weight)

    def i(self, h_v, h0_v):
        return F.elu(self.i2(F.elu(self.i1(torch.cat([h_v, h0_v], dim=1)))))

    def j(self, h_v):
        return F.elu(self.j1(h_v))

    def r(self, h, h0):
        return sum(torch.sigmoid(self.i(h[v], h0[v])) * self.j(h[v]) for v in h.keys())

    def forward(self, h, h0):
        return self.r(h, h0)


class MPNN(nn.Module):
    def __init__(self):
        super(MPNN, self).__init__()

        self.M = MessagePhase()
        self.R = Readout()

        self.p1 = nn.Linear(P, P)
        nn.init.kaiming_normal_(self.p1.weight)
        self.p2 = nn.Linear(P, P)
        nn.init.kaiming_normal_(self.p2.weight)
        self.p3 = nn.Linear(P, V)
        nn.init.kaiming_normal_(self.p3.weight)

    def p(self, ro):
        return F.elu(self.p3(F.elu(self.p2(F.elu(self.p1(ro))))))

    def forward(self, smile):
        h, h0 = self.M(smile)
        embed = self.R(h, h0)
        return self.p(embed)


model = MPNN()
optimizer = optim.Adam(model.parameters(), lr=LR)

# construct_multigraph("C1(C)C(Br)C1")

for epoch in range(NUM_EPOCHS):
    print("epoch [%d/%d]" % (epoch+1, NUM_EPOCHS))
    train_loss = 0
    train_bar = FloatProgress(min=0, max=TRAIN_SIZE)
    display(train_bar)
    for batch in range(0, TRAIN_SIZE, BATCH_SIZE):
        batch_loss = 0
        optimizer.zero_grad()
        for sample in range(BATCH_SIZE):
            # print(index)
            index = sample + batch
            print(index)
            smile = X_train.iloc[index]['smiles']
            # print('beep1')
            y_hat = model(smile)
            # print('beep1')
            y_tru = torch.Tensor(y_train.iloc[index].values.reshape(1, V))
            print("y_train[index]: ", y_train.iloc[index])
            y_hat1 = y_hat
            print("y_hat: ", y_hat1.reshape(V, 1))
#             df.loc[len(df)] = y_hat
            if(epoch == 2):
                y_hat1 = y_hat.detach().numpy()
                df1 = pd.DataFrame(y_hat1, columns=[
                                   'col1', 'col2', 'col3', 'col4', 'col5', 'col6', 'col7', 'col8', 'col9', 'col10', 'col11', 'col12'])
                dfn = dfn.append(df1)
            batch_loss += batch_mse_loss(y_hat, y_tru)
            train_bar.value += 1
        train_loss += (batch_loss * scale_batch_to_train).detach()
        batch_loss.backward()
        optimizer.step()
    valid_loss = 0
    accu_check = 0
    valid_bar = FloatProgress(min=0, max=VALID_SIZE)
    display(valid_bar)
    for sample in range(VALID_SIZE):
        index = sample
        smile = X_val.iloc[index]['smiles']
        y_hat = model(smile)
        y_tru = torch.Tensor(y_val.iloc[index].values.reshape(1, V))
        valid_loss += valid_mse_loss(y_hat, y_tru)
        accu_check += np.abs(scaler.inverse_transform(y_hat.detach()) -
                             scaler.inverse_transform(y_tru.detach())) / VALID_SIZE
        valid_bar.value += 1
    print('train_loss [%4.2f]' % (train_loss.item()))
    print('valid_loss [%4.2f]' % (valid_loss.item()))
    print(accu_check)


