"""Retrain the published P3 baselines from authorized train/development inputs.

Never reads verifier labels. Frozen inference files are supplied separately.
"""
import argparse
import json
from pathlib import Path
import shutil
import numpy as np
from sklearn.kernel_ridge import KernelRidge
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

CORES=('cr3','cr2','mn3','mn2','fe3','fe2','co3','co2')
SPECS=[([128,64],.01),([128,64],.1),([256,128],.01),([256,128],.1)]

def data(path):
 rows=json.loads(path.read_text())['records']
 x=np.array([r['standard_racs']+[int(r['metal'].lower()+str(r['oxidation_state'])==c) for c in CORES] for r in rows])
 y=np.array([r['gap_kcal_mol'] for r in rows]);return x,y

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--input',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--baseline',choices=['rac_krr','rac_mlp'],required=True);args=parser.parse_args()
 train,y=data(args.input/'data/training.json');dev,dev_y=data(args.input/'data/development.json')
 if len(train)!=956 or len(dev)!=235:raise ValueError('baseline input partition mismatch')
 directory=args.output/'solution';directory.mkdir(parents=True,exist_ok=False)
 scaler=StandardScaler().fit(train);x=scaler.transform(train)
 if args.baseline=='rac_krr':
  # Fixed development-selected KRR parameters from the baseline registry.
  model=KernelRidge(kernel='rbf',alpha=.001,gamma=.001).fit(x,y)
  assets=dict(mean=scaler.mean_,scale=scaler.scale_,train_x=x,dual_coef=model.dual_coef_,gamma=.001)
 else:
  ys=StandardScaler().fit(y[:,None]);target=ys.transform(y[:,None])[:,0];scores=[];best=None
  for index,(hidden,alpha) in enumerate(SPECS):
   models=[MLPRegressor(hidden_layer_sizes=tuple(hidden),alpha=alpha,random_state=seed,activation='relu',solver='adam',learning_rate_init=.001,batch_size=128,max_iter=500,tol=1e-7,n_iter_no_change=40,early_stopping=False).fit(x,target) for seed in [61031,61032,61033]]
   prediction=np.mean([m.predict(scaler.transform(dev))*ys.scale_[0]+ys.mean_[0] for m in models],axis=0)
   score=float(np.mean(np.abs(prediction-dev_y)));scores.append(dict(index=index,development_mae=score))
   if best is None or score<best[0]:best=(score,index,models)
  score,index,models=best
  assets=dict(x_mean=scaler.mean_,x_scale=scaler.scale_,y_mean=ys.mean_[0],y_scale=ys.scale_[0],members=len(models),layers=len(models[0].coefs_))
  for k,model in enumerate(models):
   for j,(weight,bias) in enumerate(zip(model.coefs_,model.intercepts_)):assets[f'w_{k}_{j}']=weight;assets[f'b_{k}_{j}']=bias
  (args.output/'development_selection.json').write_text(json.dumps(dict(candidates=scores,selected_index=index),indent=2)+'\n')
 np.savez(directory/'model.npz',**assets)
 supplied=Path(__file__).parent/args.baseline/'solution/main.py'
 shutil.copyfile(supplied,directory/'main.py')
 print('Trained from authorized inputs; no final targets read.')

if __name__=='__main__':
 with threadpool_limits(limits=1):main()
