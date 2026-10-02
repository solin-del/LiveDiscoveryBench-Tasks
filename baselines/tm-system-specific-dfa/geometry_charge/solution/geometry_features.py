"""XYZ-only construction baselines, invariant to rigid motion and atom order."""
from collections import Counter
from pathlib import Path

import numpy as np

# Fixed before held-out evaluation. Unknown elements fail, never silently drop.
ELEMENTS = ('H', 'C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Br', 'I', 'Cr', 'Mn', 'Fe', 'Co')
METALS = ('Cr', 'Mn', 'Fe', 'Co')


def read_xyz(path):
    lines = Path(path).read_text(encoding='utf-8').splitlines()
    n = int(lines[0])
    if n < 2 or n > 200 or len(lines) != n+2:
        raise ValueError('XYZ atom count mismatch')
    symbols, coordinates = [], []
    for line in lines[2:]:
        fields = line.split()
        if len(fields) != 4 or fields[0] not in ELEMENTS:
            raise ValueError('invalid or unsupported XYZ atom')
        symbols.append(fields[0])
        coordinates.append([float(x) for x in fields[1:]])
    coordinates = np.array(coordinates)
    if not np.isfinite(coordinates).all():
        raise ValueError('nonfinite geometry')
    return symbols, coordinates


def formula_key(symbols):
    counts = Counter(symbols)
    return '|'.join(f'{element}:{counts[element]}' for element in ELEMENTS if counts[element])


def featurize(symbols, coordinates, view):
    coordinates = np.asarray(coordinates, dtype=float)
    if coordinates.shape != (len(symbols), 3) or not np.isfinite(coordinates).all():
        raise ValueError('invalid coordinate array')
    if any(s not in ELEMENTS for s in symbols):
        raise ValueError('unsupported element')
    metal_indices = [i for i, s in enumerate(symbols) if s in METALS]
    if len(metal_indices) != 1:
        raise ValueError('requires exactly one target metal')
    metal = metal_indices[0]
    counts = Counter(symbols)
    features = [counts[element] for element in ELEMENTS]
    if view == 'composition':
        return np.array(features, dtype=float)
    if view != 'geometry':
        raise ValueError('unknown feature view')
    radii = np.linalg.norm(coordinates - coordinates[metal], axis=1)
    nonmetal = np.arange(len(symbols)) != metal
    if (radii[nonmetal] <= 0.1).any():
        raise ValueError('coincident or near-coincident atom centers')
    symbols = np.array(symbols)
    # Metal-centered radial moments, per distinct element, in angstrom.
    for element in ELEMENTS:
        r = radii[(symbols == element) & nonmetal]
        features.extend([r.mean(), r.std(), r.min(), r.max()] if len(r) else [0]*4)
    # Six closest heavy atoms: distance-only diagnostics, not assigned bonds.
    closest = np.sort(radii[nonmetal & (symbols != 'H')])[:6]
    if len(closest) != 6:
        raise ValueError('requires at least six nonmetal heavy atoms')
    features.extend(closest)
    return np.array(features, dtype=float)
