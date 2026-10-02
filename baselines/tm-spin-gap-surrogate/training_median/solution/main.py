import json,sys
for line in sys.stdin:
 r=json.loads(line)
 print(json.dumps({'predictions':[{'sample_id':v['sample_id'],'predicted_gap_kcal_mol':-33.44683746856504} for v in r['records']]},allow_nan=False),flush=True)
