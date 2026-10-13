import json
import sys
import numpy as np
for line in sys.stdin:
    request = json.loads(line)
    if request['action'] == 'initialize':
        candidates = request['candidates']
        ids = [r['candidate_id'] for r in candidates]
        index = {v: i for i, v in enumerate(ids)}
        sampled = [index[r['candidate_id']] for r in request['initial_labels']]
        pool = set(range(len(ids))) - set(sampled)
        seed = request['seed']
        rng = np.random.default_rng(seed)
        response = {'status': 'ready'}
    else:
        feedback = request['feedback']
        if feedback is not None and (not feedback['cached']):
            i = index[feedback['candidate_id']]
            pool.remove(i)
            sampled.append(i)
        remaining = sorted(pool)
        if not remaining or request['remaining'] == 0:
            response = {'candidate_id': None}
        else:
            response = {'candidate_id': ids[int(rng.choice(remaining))]}
    print(json.dumps(response, allow_nan=False), flush=True)
