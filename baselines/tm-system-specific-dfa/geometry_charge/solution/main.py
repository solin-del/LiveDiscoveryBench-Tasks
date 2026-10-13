"""Select methods with a frozen numeric tree ensemble over JSON-lines."""

import json
from pathlib import Path
import sys

import numpy as np

from p5_defined_features import featurize, predict

with np.load(Path(__file__).with_name('model.npz'), allow_pickle=False) as archive:
    assets = {name: archive[name] for name in archive.files}
methods = assets['method_ids'].tolist()
view = str(assets['view'])

for line in sys.stdin:
    request = json.loads(line)
    if request['action'] != 'select_method' or request['method_ids'] != methods:
        raise ValueError('registry mismatch')
    features = np.array([featurize(row, view) for row in request['records']])
    predicted_errors = predict(features, assets)
    # np.argmin selects the first minimum in the frozen registry order.
    selected_indices = np.argmin(predicted_errors, axis=1)
    recommendations = [
        {'sample_id': row['sample_id'], 'method_id': methods[int(index)]}
        for row, index in zip(request['records'], selected_indices)
    ]
    print(json.dumps({'recommendations': recommendations}, allow_nan=False), flush=True)
