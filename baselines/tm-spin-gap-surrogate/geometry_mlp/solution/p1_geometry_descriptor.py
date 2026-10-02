"""Explicit, rotation/translation/permutation-invariant structural descriptor.

No ground state, source name or target-derived electronic feature is read.
"""
from collections import Counter
import numpy as np

ELEMENT_Z={'H':1,'B':5,'C':6,'N':7,'O':8,'F':9,'Si':14,'P':15,'S':16,'Cl':17,
           'Cr':24,'Mn':25,'Fe':26,'Co':27,'Ni':28,'Se':34,'Br':35,'I':53}
ELEMENTS=tuple(ELEMENT_Z)
METALS=('Cr','Mn','Fe','Co','Ni')
MAX_ATOMS=256

def formula_key(symbols):
 counts=Counter(symbols)
 if set(counts)-set(ELEMENTS):raise ValueError('unsupported element')
 return '|'.join(f'{e}:{counts[e]}' for e in ELEMENTS if counts[e])

def featurize(record):
 symbols=record['atomic_symbols'];xyz=np.asarray(record['coordinates_angstrom'],dtype=float)
 if xyz.shape!=(len(symbols),3) or not np.isfinite(xyz).all() or len(symbols)>MAX_ATOMS:raise ValueError('invalid geometry')
 metal=[i for i,e in enumerate(symbols) if e in METALS]
 if len(metal)!=1:raise ValueError('requires one target metal')
 z=np.array([ELEMENT_Z[e] for e in symbols]);counts=Counter(symbols)
 distance=np.linalg.norm(xyz[:,None,:]-xyz[None,:,:],axis=2)
 if np.min(distance+np.eye(len(z))*1e10)<=.1:raise ValueError('coincident atom centers')
 features=[counts[e] for e in ELEMENTS]+[int(symbols[metal[0]]==m) for m in METALS]
 features += [record['total_charge'],record['low_spin_multiplicity'],record['high_spin_multiplicity']]
 radii=distance[metal[0]];array=np.array(symbols)
 for e in ELEMENTS:
  values=radii[(array==e)&(np.arange(len(z))!=metal[0])]
  features += [float(values.mean()),float(values.std()),float(values.min()),float(values.max())] if len(values) else [0.]*4
 # A Coulomb-matrix spectrum uses only nuclear charges and distances.
 # It is a structural descriptor, not an electronic-structure energy.
 matrix=z[:,None]*z[None,:]/(distance+np.eye(len(z)))
 np.fill_diagonal(matrix,.5*z**2.4)
 spectrum=np.linalg.eigvalsh(matrix);spectrum=spectrum[np.argsort(-np.abs(spectrum),kind='stable')]
 features += np.pad(spectrum,(0,MAX_ATOMS-len(z))).tolist()
 return np.array(features,dtype=float)
