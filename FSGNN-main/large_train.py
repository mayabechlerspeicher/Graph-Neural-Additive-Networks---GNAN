from __future__ import division
from __future__ import print_function
import time
import random
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from utils import *
from model import *
import uuid
import pickle
from collections import Counter

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=0, help='Random seed.')
parser.add_argument('--epochs', type=int, default=1500, help='Number of epochs to train.')
parser.add_argument('--layer', type=int, default=3, help='Number of layers.')
parser.add_argument('--hidden', type=int, default=64, help='hidden dimensions.')
parser.add_argument('--dropout', type=float, default=0.6, help='Dropout rate (1 - keep probability).')
parser.add_argument('--patience', type=int, default=300, help='Patience')
parser.add_argument('--dev', type=int, default=0, help='device id')
parser.add_argument('--test', action='store_true', default=False, help='evaluation on test set.')
parser.add_argument('--layer_norm',type=int, default=1, help='layer norm')
parser.add_argument('--wd_fc1',type=float, default=1e-05, help='Weight decay layer-1')
parser.add_argument('--wd_fc2',type=float, default=1e-06, help='Weight decay layer-2')
parser.add_argument('--wd_fc3',type=float, default=9e-06, help='Weight decay layer-3')
parser.add_argument('--wd_att',type=float, default=0.1, help='Weight decay scalar')
parser.add_argument('--lr_fc1',type=float, default=0.00005, help='Learning rate fc layer-1')
parser.add_argument('--lr_fc2',type=float, default=0.0002, help='Learning rate fc layer-2')
parser.add_argument('--lr_fc3',type=float, default=0.00002, help='Learning rate fc layer-3')
parser.add_argument('--lr_att',type=float, default=0.0001, help='Learning rate scalar')
parser.add_argument('--batch_size',type=int, default=4096, help='Batch size')
parser.add_argument('--dp1',type=float, default=0.5, help='Dropout-1')
parser.add_argument('--dp2',type=float, default=0.6, help='Dropout-2')



args = parser.parse_args()
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

num_layer = args.layer
layer_norm = bool(int(args.layer_norm))
batch_size = args.batch_size


print(f"========================")
print(f"ogbn-papers100M: Seed {args.seed}")
print(f"Dropout      :: dropout1: {args.dp1}, dropout2:{args.dp2}, layer_norm: {layer_norm}")
print(f"Learning Rate:: lr_fc1: {args.lr_fc1}, lr_fc2: {args.lr_fc2}, lr_fc3: {args.lr_fc3}, lr_att: {args.lr_att}")
print(f"Weight Decay :: wd1: {args.wd_fc1}, wd2: {args.wd_fc2}, wd3: {args.wd_fc3}, w_att:{args.wd_att}")
print(f"Batch size   :: {batch_size}")

data_path = './large_data/'

with open(data_path+"training.pickle","rb") as fopen:
    train_data = pickle.load(fopen)

with open(data_path+"validation.pickle","rb") as fopen:
    valid_data = pickle.load(fopen)

with open(data_path+"test.pickle","rb") as fopen:
    test_data = pickle.load(fopen)

with open(data_path+"labels.pickle","rb") as fopen:
    labels = pickle.load(fopen)

cudaid = "cuda:"+str(args.dev)
device = torch.device(cudaid)


train_data = [mat.to(device) for mat in train_data[:9]]
valid_data = [mat.to(device) for mat in valid_data[:9]]
#test_data = [mat.to(device) for mat in test_data[:9]]
train_labels = labels[0].reshape(-1).long().to(device)
valid_labels = labels[1].reshape(-1).long().to(device)
test_labels = labels[2].reshape(-1).long().to(device)


num_features = train_data[0].shape[1]
num_labels = int(train_labels.max()) + 1
#print(num_labels)
num_layer = args.layer
checkpt_file = 'pretrained/'+uuid.uuid4().hex+'.pt'
print(cudaid,checkpt_file)

model = FSGNN_Large(nfeat=num_features,
            nlayers=2*args.layer + 1,
            nhidden=args.hidden,
            nclass=num_labels,
            dp1=args.dp1,dp2=args.dp2).to(device)



optimizer_sett = [
    {'params': model.wt1.parameters(), 'weight_decay': args.wd_fc1, 'lr': args.lr_fc1},
    {'params': model.fc2.parameters(), 'weight_decay': args.wd_fc2, 'lr': args.lr_fc2},
    {'params': model.fc3.parameters(), 'weight_decay': args.wd_fc3, 'lr': args.lr_fc3},
    {'params': model.att, 'weight_decay': args.wd_att, 'lr': args.lr_att},
    ]

optimizer = optim.Adam(optimizer_sett)


def create_batch(input_data):
    num_sample = input_data[0].shape[0]
    list_bat = []
    for i in range(0,num_sample,batch_size):
        if (i+batch_size)<num_sample:
            list_bat.append((i,i+batch_size))
        else:
            list_bat.append((i,num_sample))
    return list_bat



def train(st,end):
    model.train()
    optimizer.zero_grad()
    output = model(train_data,layer_norm,st,end)
    acc_train = accuracy(output, train_labels[st:end])
    loss_train = F.nll_loss(output, train_labels[st:end])
    loss_train.backward()
    optimizer.step()
    return loss_train.item(),acc_train.item()


def validate(st,end):
    model.eval()
    with torch.no_grad():
        output = model(valid_data,layer_norm,st,end)
        loss_val = F.nll_loss(output, valid_labels[st:end])
        acc_val = accuracy(output, valid_labels[st:end],batch=True)
        return loss_val.item(),acc_val.item()

def test(st,end):
    model.eval()
    torch.cuda.empty_cache()
    with torch.no_grad():
        output = model(test_data,layer_norm,st,end)
        loss_test = F.nll_loss(output, test_labels[st:end])
        acc_test = accuracy(output, test_labels[st:end],batch=True)
        return loss_test.item(),acc_test.item()

list_bat_train = create_batch(train_data)
list_bat_val = create_batch(valid_data)
list_bat_test = create_batch(test_data)
t_total = time.time()

bad_counter = 0
best = 999999999
best_epoch = 0
acc = 0
valid_num = valid_data[0].shape[0]
test_num = test_data[0].shape[0]
for epoch in range(args.epochs):
    list_loss = []
    list_acc = []
    random.shuffle(list_bat_train)
    for st,end in list_bat_train:
        loss_tra,acc_tra = train(st,end)
        list_loss.append(loss_tra)
        list_acc.append(acc_tra)
    loss_tra = np.round(np.mean(list_loss),4)
    acc_tra = np.round(np.mean(list_acc),4)

    list_loss_val = []
    list_acc_val = []
    for st,end in list_bat_val:
        loss_val,acc_val = validate(st,end)
        list_loss_val.append(loss_val)
        list_acc_val.append(acc_val)

    loss_val = np.mean(list_loss_val)
    acc_val = (np.sum(list_acc_val))/valid_num

    #Uncomment to see losses
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
        best_epoch = epoch
        acc = acc_val
        torch.save(model.state_dict(), checkpt_file)
        bad_counter = 0
    else:
        bad_counter += 1

    if bad_counter == args.patience:
        break


#Following lines only needed if GPU memory is not enough
#for loading all data and model training. Otherwise, test
#data can be loaded to GPU earlier.
del train_data
del valid_data
test_data = [mat.to(device) for mat in test_data[:9]]
torch.cuda.empty_cache()

if args.test:
    list_loss_test = []
    list_acc_test = []
    model.load_state_dict(torch.load(checkpt_file))
    for st,end in list_bat_test:
        loss_test,acc_test = test(st,end)
        list_loss_test.append(loss_test)
        list_acc_test.append(acc_test)
    acc_test = (np.sum(list_acc_test))/test_num


print("Train cost: {:.4f}s".format(time.time() - t_total))
print('Load {}th epoch'.format(best_epoch))

if args.test:
    print(f"Valdiation accuracy: {np.round(acc*100,2)}, Test accuracy: {np.round(acc_test*100,2)}")
else:
    print(f"Valdiation accuracy: {np.round(acc*100,2)}")







