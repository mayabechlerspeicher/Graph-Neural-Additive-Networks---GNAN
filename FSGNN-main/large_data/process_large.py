import numpy as np
import torch
import pickle
from ogb.nodeproppred.dataset_pyg import PygNodePropPredDataset
import scipy.sparse as sp
from torch_geometric.utils import to_undirected
from torch_sparse import SparseTensor


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    rowsum = (rowsum==0)*1+rowsum
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features

def sys_normalized_adjacency_i(adj):
   adj = sp.coo_matrix(adj)
   adj = adj + sp.eye(adj.shape[0])
   row_sum = np.array(adj.sum(1))
   row_sum=(row_sum==0)*1+row_sum
   d_inv_sqrt = np.power(row_sum, -0.5).flatten()
   d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
   d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
   return d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt).tocoo()

def sys_normalized_adjacency(adj):
   adj = sp.coo_matrix(adj)
   #adj = adj + sp.eye(adj.shape[0])
   row_sum = np.array(adj.sum(1))
   row_sum=(row_sum==0)*1+row_sum
   d_inv_sqrt = np.power(row_sum, -0.5).flatten()
   d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
   d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
   return d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt).tocoo()

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


#Load dataset
dataset = PygNodePropPredDataset('ogbn-papers100M')
data = dataset[0]
split_idx = dataset.get_idx_split()

#get split indices
train_idx, valid_idx, test_idx = split_idx['train'], split_idx['valid'], split_idx['test']

edge_index = data.edge_index
N = data.num_nodes

#Load edges and create adjacency
row,col = edge_index
adj = SparseTensor(row=row,col=col,sparse_sizes=(N,N))
adj = adj.to_scipy(layout='csr')
print("Getting undirected matrix...")
adj = adj + adj.transpose()
print("Saving unnormalized adjacency matrix")
sp.save_npz('100M_undirected_adj.npz',adj)

#adj = sp.load_npz('100M_undirected_adj.npz')

feat = data.x.numpy()

feat = torch.from_numpy(feat).float()
print("Normalizing matrix A...")
adj_mat = sys_normalized_adjacency(adj)
sp.save_npz('normalized_adj.npz',adj_mat)
adj_mat = sparse_mx_to_torch_sparse_tensor(adj_mat)
print("Normalizing matrix A_i...")
adj_mat_i =  sys_normalized_adjacency_i(adj)
sp.save_npz('normalized_adj_i.npz',adj_mat_i)
adj_mat_i = sparse_mx_to_torch_sparse_tensor(adj_mat_i)

#creating 3-hop features

list_mat_train = []
list_mat_valid = []
list_mat_test = []
#Adding just features X
list_mat_train.append(feat[train_idx,:])
list_mat_valid.append(feat[valid_idx,:])
list_mat_test.append(feat[test_idx,:])

agg_feat = feat
agg_feat_i = feat

#Calculate features for 3 hops
for i in range(4):
    agg_feat = torch.spmm(adj_mat,agg_feat)
    agg_feat_i = torch.spmm(adj_mat_i,agg_feat_i)
    list_mat_train.append(agg_feat[train_idx,:])
    list_mat_valid.append(agg_feat[valid_idx,:])
    list_mat_test.append(agg_feat[test_idx,:])
    list_mat_train.append(agg_feat_i[train_idx,:])
    list_mat_valid.append(agg_feat_i[valid_idx,:])
    list_mat_test.append(agg_feat_i[test_idx,:])

with open('training.pickle',"wb") as fopen:
    pickle.dump(list_mat_train,fopen)

with open('validation.pickle',"wb") as fopen:
    pickle.dump(list_mat_valid,fopen)

with open('test.pickle',"wb") as fopen:
    pickle.dump(list_mat_test,fopen)

#save labels
labels = data.y.data
print(f"Type: {type(labels)},{labels.shape}")

labels_train = labels[train_idx].reshape(-1).long()
labels_valid = labels[valid_idx].reshape(-1).long()
labels_test = labels[test_idx].reshape(-1).long()

with open('labels.pickle',"wb") as fopen:
    pickle.dump([labels_train,labels_valid,labels_test],fopen)



