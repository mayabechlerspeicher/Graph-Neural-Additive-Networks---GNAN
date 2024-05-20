import torch.nn as nn
import torch
import math
import numpy as np
import torch.nn.functional as F
from torch.nn.parameter import Parameter


class FSGNN(nn.Module):
    def __init__(self,nfeat,nlayers,nhidden,nclass,dropout,agg_oper):
        super(FSGNN,self).__init__()
        if agg_oper == 'cat':
            self.fc2 = nn.Linear(nhidden*nlayers,nclass)
        elif agg_oper == 'sum':
            self.fc2 = nn.Linear(nhidden,nclass)
        self.dropout = dropout
        self.act_fn = nn.ReLU()
        self.fc1 = nn.ModuleList([nn.Linear(nfeat,int(nhidden)) for _ in range(nlayers)])
        self.agg_oper = agg_oper



    def forward(self,list_mat,layer_norm,is_relu):

        list_out = list()
        for ind, mat in enumerate(list_mat):
            tmp_out = self.fc1[ind](mat)
            if layer_norm == True:
                tmp_out = F.normalize(tmp_out,p=2,dim=1)

            list_out.append(tmp_out)

        if self.agg_oper == 'sum':
            device = list_out[0].get_device()
            final_mat = torch.zeros_like(list_out[0]).cuda(device)
            for mat in list_out:
                final_mat += mat
        elif self.agg_oper == 'cat':
            final_mat = torch.cat(list_out, dim=1)
        if is_relu == True: 
            out = self.act_fn(final_mat)
        else:
            out = final_mat
        out = F.dropout(out,self.dropout,training=self.training)
        out = self.fc2(out)


        return F.log_softmax(out, dim=1)


if __name__ == '__main__':
    pass






