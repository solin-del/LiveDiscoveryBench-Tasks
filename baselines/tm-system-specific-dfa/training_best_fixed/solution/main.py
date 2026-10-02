import json,sys
for line in sys.stdin:
 r=json.loads(line)
 print(json.dumps({'recommendations':[{'sample_id':v['sample_id'],'method_id':'m06-l_hfx_30'} for v in r['records']]},allow_nan=False),flush=True)
