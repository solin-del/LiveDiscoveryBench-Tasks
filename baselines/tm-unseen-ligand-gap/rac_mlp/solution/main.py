import json,sys
from pathlib import Path
import numpy as np
CORES=('cr3','cr2','mn3','mn2','fe3','fe2','co3','co2')
model=np.load(Path(__file__).with_name('model.npz'),allow_pickle=False)
for line in sys.stdin:
 r=json.loads(line);records=r['records']
 x=np.array([v['standard_racs']+[int(v['metal'].lower()+str(v['oxidation_state'])==c) for c in CORES] for v in records])
 x=(x-model['x_mean'])/model['x_scale'];pred=[]
 for k in range(int(model['members'])):
  a=x
  for j in range(int(model['layers'])):
   a=a@model['w_'+str(k)+'_'+str(j)]+model['b_'+str(k)+'_'+str(j)]
   if j+1<int(model['layers']):a=np.maximum(a,0.)
  pred.append(a[:,0]*float(model['y_scale'])+float(model['y_mean']))
 y=np.mean(pred,axis=0)
 print(json.dumps({'predictions':[{'sample_id':v['sample_id'],'predicted_gap_kcal_mol':float(p)} for v,p in zip(records,y)]},allow_nan=False),flush=True)
