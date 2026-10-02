"""Numeric-only inference for independently trained P1 structural baselines."""
import numpy as np

def predict_tree(x,assets):
    total=np.zeros(len(x));roots=assets['roots']
    for root in roots:
        node=np.full(len(x),root,dtype=np.int64)
        for _ in range(assets['left'].size):
            active=np.flatnonzero(assets['left'][node]>=0)
            if not len(active):break
            current=node[active]
            goes_left=x[active,assets['feature'][current]]<=assets['threshold'][current]
            node[active]=np.where(goes_left,assets['left'][current],assets['right'][current])
        else:raise ValueError('cyclic tree asset')
        total+=assets['value'][node]
    return total/len(roots)

def predict(x,assets,family):
    if family=='trees':return predict_tree(x.astype(np.float32),assets)
    if family=='krr':
        x=(x-assets['mean'])/assets['scale'];train=assets['train_x']
        squared=np.maximum(np.sum(x*x,axis=1)[:,None]+np.sum(train*train,axis=1)[None,:]-2*x@train.T,0.)
        return np.exp(-float(assets['gamma'])*squared)@assets['dual_coef']
    x=(x-assets['x_mean'])/assets['x_scale'];members=[]
    for k in range(int(assets['members'])):
        value=x.copy()
        for j in range(int(assets['layers'])):
            value=value@assets[f'w_{k}_{j}']+assets[f'b_{k}_{j}']
            if j<int(assets['layers'])-1:value=np.maximum(value,0.)
        members.append(value.ravel()*float(assets['y_scale'])+float(assets['y_mean']))
    return np.mean(members,axis=0)
