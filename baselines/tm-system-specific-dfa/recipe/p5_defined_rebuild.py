"""Rebuild the frozen recommender from its public training rows."""

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil

import numpy as np
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from p5_defined_features import featurize, predict

SERVER = '''"""Select methods with a frozen numeric tree ensemble over JSON-lines."""

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
'''


def export(model, methods, view):
    """Flatten fitted trees into arrays for inference without sklearn objects."""
    columns = {name: [] for name in ('left', 'right', 'feature', 'threshold', 'value')}
    roots = []
    offset = 0
    for estimator in model.regressor_.estimators_:
        tree = estimator.tree_
        roots.append(offset)
        columns['left'].append(
            np.where(tree.children_left < 0, -1, tree.children_left + offset)
        )
        columns['right'].append(
            np.where(tree.children_right < 0, -1, tree.children_right + offset)
        )
        columns['feature'].append(tree.feature)
        columns['threshold'].append(tree.threshold)
        columns['value'].append(tree.value[:, :, 0])
        offset += tree.node_count
    assets = {name: np.concatenate(values, axis=0) for name, values in columns.items()}
    assets.update(
        roots=np.array(roots),
        y_mean=model.transformer_.mean_,
        y_scale=model.transformer_.scale_,
        method_ids=np.array(methods),
        view=np.array(view),
    )
    return assets


def fit_public(training, recipe, view):
    """Fit the declared configuration without searching or reading final rows."""
    methods = training['method_ids']
    rows = training['records']
    features = np.array([featurize(row, view) for row in rows])
    errors = np.array([
        [
            abs(row['method_gaps_kcal_mol'][method] - row['reference_gap_kcal_mol'])
            for method in methods
        ]
        for row in rows
    ])
    selected = recipe['selections'][view]
    if selected['target'] != 'absolute_error' or selected['spec']['kind'] != 'extra_trees':
        raise ValueError('unsupported frozen alternative configuration')
    model = TransformedTargetRegressor(
        regressor=ExtraTreesRegressor(
            n_estimators=256,
            n_jobs=1,
            random_state=20260930,
            min_samples_leaf=selected['spec']['min_samples_leaf'],
            max_features=selected['spec']['max_features'],
        ),
        transformer=StandardScaler(),
    ).fit(features, errors)
    arrays = export(model, methods, view)
    difference = float(np.max(np.abs(predict(features, arrays) - model.predict(features))))
    if difference > 1e-10:
        raise ValueError('numeric model inference mismatch')
    return arrays, difference


def save(folder, assets):
    """Write the numeric model and its self-contained JSON-lines entry."""
    solution = folder / 'solution'
    solution.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(solution / 'model.npz', **assets)
    for name in ('p5_defined_features.py', 'geometry_features.py'):
        shutil.copyfile(Path(__file__).parent / name, solution / name)
    (solution / 'main.py').write_text(SERVER, encoding='utf-8')
    return hashlib.sha256((solution / 'model.npz').read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('training', 'recipe', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('use a fresh reconstruction directory')

    versions = {
        name: importlib.metadata.version(name)
        for name in ('numpy', 'scipy', 'scikit-learn', 'threadpoolctl')
    }
    expected = dict(
        numpy='2.3.2', scipy='1.17.1',
        **{'scikit-learn': '1.7.2', 'threadpoolctl': '3.7.0'},
    )
    if versions != expected:
        raise ValueError('pinned numerical dependencies required')

    training = json.loads(args.training.read_text(encoding='utf-8'))
    recipe = json.loads(args.recipe.read_text(encoding='utf-8'))
    training_bytes = args.training.read_bytes().replace(b'\r\n', b'\n')
    if hashlib.sha256(training_bytes).hexdigest() != recipe['training_sha256']:
        raise ValueError('frozen training hash mismatch')

    args.output.mkdir(parents=True)
    report = dict(
        requires_final_data=False,
        searches_configurations=False,
        training_count=len(training['records']),
        versions=versions,
        models={},
    )
    for view in ('geometry_charge', recipe['density_view']):
        assets, difference = fit_public(training, recipe, view)
        report['models'][view] = dict(
            asset_sha256=save(args.output / view, assets),
            maximum_numeric_training_difference=difference,
        )
    (args.output / 'rebuild_report.json').write_text(
        json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8'
    )
    print('P5_PUBLIC_INPUT_NUMERIC_REBUILD_PASS', flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        main()
