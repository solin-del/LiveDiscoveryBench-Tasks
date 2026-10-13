"""Pool raw116 atom channels and evaluate frozen numeric tree ensembles."""

import numpy as np

from geometry_features import ELEMENTS, featurize as geometry


def featurize(row, view):
    """Append element counts and raw channel moments to geometry and charge.

    Population standard deviations use NumPy's default ddof=0. Absent elements
    contribute zeros. Atom channels must follow the supplied atom order.
    """
    symbols = row['atomic_symbols']
    base = np.r_[
        geometry(symbols, row['coordinates_angstrom'], 'geometry'),
        float(row['total_charge']),
    ]
    if view == 'geometry_charge':
        return base
    if view not in ('density_mean', 'density_mean_std'):
        raise ValueError('unknown descriptor')

    raw = np.asarray(row['density_difference_power_spectrum_raw116'], dtype=float)
    if (
        raw.shape != (len(symbols), 116)
        or not np.isfinite(raw).all()
        or (raw < -1e-8).any()
    ):
        raise ValueError('invalid raw116 atom-channel array')

    blocks = []
    for element in ELEMENTS:
        values = raw[np.asarray(symbols) == element]
        blocks.extend([float(len(values))])
        blocks.extend(values.mean(axis=0).tolist() if len(values) else [0.0] * 116)
        if view == 'density_mean_std':
            blocks.extend(values.std(axis=0).tolist() if len(values) else [0.0] * 116)
    return np.r_[base, blocks]


def predict(features, assets):
    """Average exported tree outputs and undo the frozen target scaling."""
    features = np.asarray(features, dtype=np.float32)
    total = np.zeros((len(features), len(assets['y_mean'])))
    for root in assets['roots']:
        nodes = np.full(len(features), root, dtype=np.int64)
        for _ in range(len(assets['left'])):
            active = np.flatnonzero(assets['left'][nodes] >= 0)
            if not len(active):
                break
            current = nodes[active]
            go_left = (
                features[active, assets['feature'][current]]
                <= assets['threshold'][current]
            )
            nodes[active] = np.where(
                go_left, assets['left'][current], assets['right'][current]
            )
        else:
            raise ValueError('cyclic model tree')
        total += assets['value'][nodes]
    return total / len(assets['roots']) * assets['y_scale'] + assets['y_mean']
