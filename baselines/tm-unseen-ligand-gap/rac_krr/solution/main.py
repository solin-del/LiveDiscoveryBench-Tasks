import json,sys
from pathlib import Path
import numpy as np
CORES=('cr3','cr2','mn3','mn2','fe3','fe2','co3','co2')
model=np.load(Path(__file__).with_name('model.npz'),allow_pickle=False)
for line in sys.stdin:
 request=json.loads(line);records=request['records']
 x=np.array([r['standard_racs']+[int(r['metal'].lower()+str(r['oxidation_state'])==c) for c in CORES] for r in records])
 x=(x-model['mean'])/model['scale'];train=model['train_x']
 squared=np.maximum(np.sum(x*x,axis=1)[:,None]+np.sum(train*train,axis=1)[None,:]-2*x@train.T,0.)
 pred=np.exp(-float(model['gamma'])*squared)@model['dual_coef']
 response={'predictions':[{'sample_id':r['sample_id'],'predicted_gap_kcal_mol':float(p)} for r,p in zip(records,pred)]}
 print(json.dumps(response,allow_nan=False),flush=True)
