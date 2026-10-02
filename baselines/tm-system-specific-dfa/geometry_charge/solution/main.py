import json,sys
from pathlib import Path
import numpy as np
from p5_defined_features import featurize,predict
with np.load(Path(__file__).with_name('model.npz'),allow_pickle=False) as f:
 assets={name:f[name] for name in f.files}
methods=assets['method_ids'].tolist();view=str(assets['view'])
for line in sys.stdin:
 r=json.loads(line)
 if r['action']!='select_method' or r['method_ids']!=methods:raise ValueError('registry mismatch')
 x=np.array([featurize(v,view) for v in r['records']]);values=predict(x,assets)
 print(json.dumps({'recommendations':[{'sample_id':row['sample_id'],'method_id':methods[int(i)]} for row,i in zip(r['records'],np.argmin(values,axis=1))]},allow_nan=False),flush=True)
