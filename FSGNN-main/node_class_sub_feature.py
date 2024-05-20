from __future__ import division
from __future__ import print_function
import time
import random
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from process import *
from utils import *
from model_feat_selection import *
import uuid
import pickle

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=42, help='Random seed.') #Default seed same as GCNII
parser.add_argument('--epochs', type=int, default=1500, help='Number of epochs to train.')
parser.add_argument('--layer', type=int, default=3, help='Number of layers.')
parser.add_argument('--hidden', type=int, default=64, help='hidden dimensions.')
parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (1 - keep probability).')
parser.add_argument('--patience', type=int, default=100, help='Patience')
parser.add_argument('--data', default='cora', help='dateset')
parser.add_argument('--dev', type=int, default=0, help='device id')
parser.add_argument('--layer_norm',type=int, default=1, help='layer norm')
parser.add_argument('--w_fc2',type=float, default=0.0005, help='Weight decay layer-2')
parser.add_argument('--w_fc1',type=float, default=0.0005, help='Weight decay layer-1')
parser.add_argument('--lr_fc',type=float, default=0.02, help='Learning rate 2 fully connected layers')
parser.add_argument('--hop_idx', type=int, default=0, help='Select hop combinations')
parser.add_argument('--is_relu', type=int, default=1, help='Non-linearity between layers')
parser.add_argument('--agg_oper', type=str, default='cat', help='Aggregation operation for hop features')

args = parser.parse_args()
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
num_layer = args.layer
is_relu = bool(args.is_relu)

#Dictionary stores various combinations of hops
with open("hop_select.pickle","rb") as fopen:
    dict_param_select = pickle.load(fopen)
hop_select_param = dict_param_select[args.hop_idx]

layer_norm = bool(int(args.layer_norm))
print("==========================")
print(f"Dataset: {args.data}")
print(f"Dropout:{args.dropout}, layer_norm: {layer_norm}")
print(f"w_fc2:{args.w_fc2}, w_fc1:{args.w_fc1}, lr_fc:{args.lr_fc}")


cudaid = "cuda:"+str(args.dev)
device = torch.device(cudaid)
checkpt_file = 'pretrained/'+uuid.uuid4().hex+'.pt'


def train_step(model,optimizer,labels,list_mat,idx_train):
    model.train()
    optimizer.zero_grad()
    output = model(list_mat, layer_norm, is_relu)
    acc_train = accuracy(output[idx_train], labels[idx_train].to(device))
    loss_train = F.nll_loss(output[idx_train], labels[idx_train].to(device))
    loss_train.backward()
    optimizer.step()
    return loss_train.item(),acc_train.item()


def validate_step(model,labels,list_mat,idx_val):
    model.eval()
    with torch.no_grad():
        output = model(list_mat, layer_norm, is_relu)
        loss_val = F.nll_loss(output[idx_val], labels[idx_val].to(device))
        acc_val = accuracy(output[idx_val], labels[idx_val].to(device))
        return loss_val.item(),acc_val.item()

def test_step(model,labels,list_mat,idx_test):
    model.load_state_dict(torch.load(checkpt_file))
    model.eval()
    with torch.no_grad():
        output = model(list_mat, layer_norm, is_relu)
        loss_test = F.nll_loss(output[idx_test], labels[idx_test].to(device))
        acc_test = accuracy(output[idx_test], labels[idx_test].to(device))
        #print(mask_val)
        return loss_test.item(),acc_test.item()


def train(datastr,splitstr):
    adj, adj_i, features, labels, idx_train, idx_val, idx_test, num_features, num_labels = full_load_data(datastr,splitstr)
    features = features.to(device)

    adj = adj.to(device)
    adj_i = adj_i.to(device)
    list_mat = []
    list_mat.append(features)
    no_loop_mat = features
    loop_mat = features

    for ii in range(args.layer):
        no_loop_mat = torch.spmm(adj, no_loop_mat)
        loop_mat = torch.spmm(adj_i, loop_mat)
        list_mat.append(no_loop_mat)
        list_mat.append(loop_mat)

    #Selecting hops features out of all
    # For example, selecting X, AX, (A+I)^2X 
    list_mat = [list_mat[hop_sel] for hop_sel in hop_select_param]

    model = FSGNN(nfeat=num_features,
                nlayers=len(list_mat),
                nhidden=args.hidden,
                nclass=num_labels,
                dropout=args.dropout,agg_oper=args.agg_oper).to(device)


    optimizer_sett = [
        {'params': model.fc2.parameters(), 'weight_decay': args.w_fc2, 'lr': args.lr_fc},
        {'params': model.fc1.parameters(), 'weight_decay': args.w_fc1, 'lr': args.lr_fc},
    ]

    optimizer = optim.Adam(optimizer_sett)

    bad_counter = 0
    best = 999999999
    for epoch in range(args.epochs):
        loss_tra,acc_tra = train_step(model,optimizer,labels,list_mat,idx_train)
        loss_val,acc_val = validate_step(model,labels,list_mat,idx_val)
        #Uncomment following lines to see loss and accuracy values
        '''
        if(epoch+1)%1 == 0:

            print('Epoch:{:04d}'.format(epoch+1),
                'train',
                'loss:{:.3f}'.format(loss_tra),
                'acc:{:.2f}'.format(acc_tra*100),
                '| val',
                'loss:{:.3f}'.format(loss_val),
                'acc:{:.2f}'.format(acc_val*100))
        '''        

        if loss_val < best:
            best = loss_val
            torch.save(model.state_dict(), checkpt_file)
            bad_counter = 0
        else:
            bad_counter += 1

        if bad_counter == args.patience:
            break
    test_out = test_step(model,labels,list_mat,idx_test)
    acc = test_out[1]


    return acc*100

t_total = time.time()
acc_list = []

for i in range(10):
    datastr = args.data
    splitstr = 'splits/'+args.data+'_split_0.6_0.2_'+str(i)+'.npz'
    accuracy_data = train(datastr,splitstr)
    acc_list.append(accuracy_data)


    ##print(i,": {:.2f}".format(acc_list[-1]))

print("Train cost: {:.4f}s".format(time.time() - t_total))
#print("Test acc.:{:.2f}".format(np.mean(acc_list)))
print(f"Test accuracy: {np.mean(acc_list)}, {np.round(np.std(acc_list),2)}")

