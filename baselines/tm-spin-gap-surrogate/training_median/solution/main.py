"""Return the frozen training-median gap for every requested record."""
import json
import sys


for line in sys.stdin:
    request = json.loads(line)
    print(
        json.dumps(
            {
                'predictions': [
                    {
                        'sample_id': record['sample_id'],
                        'predicted_gap_kcal_mol': -33.44683746856504,
                    }
                    for record in request['records']
                ]
            },
            allow_nan=False,
        ),
        flush=True,
    )
