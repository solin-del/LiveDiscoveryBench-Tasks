"""Return the method with the lowest MAE on the frozen training rows."""

import json
import sys

for line in sys.stdin:
    request = json.loads(line)
    recommendations = [
        {'sample_id': row['sample_id'], 'method_id': 'm06-l_hfx_30'}
        for row in request['records']
    ]
    print(json.dumps({'recommendations': recommendations}, allow_nan=False), flush=True)
