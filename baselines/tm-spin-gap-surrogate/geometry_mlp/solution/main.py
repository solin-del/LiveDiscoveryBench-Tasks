import json
import sys
from pathlib import Path
import numpy as np
from p1_geometry_descriptor import featurize
from p1_frozen_predictor import predict
with np.load(Path(__file__).with_name('model.npz'), allow_pickle=False) as archive:
    assets = {name: archive[name] for name in archive.files}
for line in sys.stdin:
    request = json.loads(line)
    records = request['records']
    values = predict(np.array([featurize(r) for r in records]), assets, 'mlp')
    print(
        json.dumps(
            {'predictions': [
                {'sample_id': r['sample_id'], 'predicted_gap_kcal_mol': float(v)}
                for r, v in zip(records, values)
            ]},
            allow_nan=False,
        ),
        flush=True,
    )
