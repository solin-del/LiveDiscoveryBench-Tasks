import json
import sys
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
CORES = ('cr3', 'cr2', 'mn3', 'mn2', 'fe3', 'fe2', 'co3', 'co2')
for line in sys.stdin:
    request = json.loads(line)
    if request['action'] == 'initialize':
        candidates = request['candidates']
        ids = [r['candidate_id'] for r in candidates]
        index = {v: i for i, v in enumerate(ids)}
        x = np.array(
            [
                r['standard_racs'] + [
                    int(r['metal'].lower() + str(r['oxidation_state']) == c) for c in CORES
                ] for r in candidates
            ]
        )
        sampled = [index[r['candidate_id']] for r in request['initial_labels']]
        values = [r['gap_kcal_mol'] for r in request['initial_labels']]
        pool = set(range(len(ids))) - set(sampled)
        seed = request['seed']
        model = None
        response = {'status': 'ready'}
    else:
        feedback = request['feedback']
        if feedback is not None and (not feedback['cached']):
            i = index[feedback['candidate_id']]
            pool.remove(i)
            sampled.append(i)
            values.append(feedback['gap_kcal_mol'])
        remaining = sorted(pool)
        if not remaining or request['remaining'] == 0:
            response = {'candidate_id': None}
        else:
            # Fit once from the condition's initial labels, then keep the ranking fixed.
            if model is None:
                model = ExtraTreesRegressor(
                    n_estimators=128,
                    min_samples_leaf=2,
                    max_features=1.0,
                    n_jobs=1,
                    random_state=seed
                ).fit(x[sampled], np.array(values))
                tree = np.stack([t.predict(x) for t in model.estimators_])
                mean = tree.mean(axis=0)
                priority = -np.abs(mean)
            response = {'candidate_id': ids[remaining[int(np.argmax(priority[remaining]))]]}
    print(json.dumps(response, allow_nan=False), flush=True)
