"""Target-free P5 geometry and documented raw116 atom-channel pooling."""
import numpy as np
from geometry_features import ELEMENTS,featurize as geometry

def featurize(row,view):
    symbols=row['atomic_symbols']
    base=np.r_[geometry(symbols,row['coordinates_angstrom'],'geometry'),float(row['total_charge'])]
    if view=='geometry_charge':return base
    if view not in ('density_mean','density_mean_std'):raise ValueError('unknown descriptor')
    raw=np.asarray(row['density_difference_power_spectrum_raw116'],dtype=float)
    if raw.shape!=(len(symbols),116) or not np.isfinite(raw).all() or (raw < -1e-8).any():
        raise ValueError('invalid raw116 atom-channel array')
    blocks=[]
    for element in ELEMENTS:
        values=raw[np.asarray(symbols)==element]
        blocks.extend([float(len(values))])
        blocks.extend(values.mean(axis=0).tolist() if len(values) else [0.]*116)
        if view=='density_mean_std':blocks.extend(values.std(axis=0).tolist() if len(values) else [0.]*116)
    return np.r_[base,blocks]

def predict(x,assets):
    x=np.asarray(x,dtype=np.float32)
    total=np.zeros((len(x),len(assets['y_mean'])))
    for root in assets['roots']:
        node=np.full(len(x),root,dtype=np.int64)
        for _ in range(len(assets['left'])):
            active=np.flatnonzero(assets['left'][node]>=0)
            if not len(active):break
            current=node[active]
            left=x[active,assets['feature'][current]]<=assets['threshold'][current]
            node[active]=np.where(left,assets['left'][current],assets['right'][current])
        else:raise ValueError('cyclic model tree')
        total+=assets['value'][node]
    return total/len(assets['roots'])*assets['y_scale']+assets['y_mean']
