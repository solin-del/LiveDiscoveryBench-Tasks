"""Rebuild frozen P1 baselines from the agent-visible training/development JSON.

This module has no construction-directory import and never opens final data.
It is copied, with the two numeric modules, into the local recipe candidate.
"""
import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
import shutil
import time
import warnings
from pathlib import Path

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.kernel_ridge import KernelRidge
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from p1_geometry_descriptor import featurize
from p1_frozen_predictor import predict

PINNED = {"numpy": "2.3.2", "scipy": "1.17.1", "scikit-learn": "1.7.2", "threadpoolctl": "3.7.0"}
EXPECTED_INPUTS = {
    "training": "b9be943ad6adbdaf9cf7853b59443efc3d5eb5e272cfb70efbc790f813acea6c",
    "development": "a91403e9734e302b752567536d4b2b5e505eb29e38080957d0d052c95cbf00aa",
}
FROZEN = {
    "trees": {"n_estimators": 256, "n_jobs": 1, "random_state": 61041,
              "min_samples_leaf": 2, "max_features": 1.0},
    "krr": {"kernel": "rbf", "alpha": 0.001, "gamma": 0.001},
    "mlp": {"hidden_layer_sizes": (256, 128), "alpha": 0.01,
            "activation": "relu", "solver": "adam", "learning_rate_init": 0.001,
            "batch_size": 128, "max_iter": 500, "tol": 1e-7,
            "n_iter_no_change": 40, "early_stopping": False},
    "mlp_seeds": [61041, 61042, 61043],
}
FIELDS = ("atomic_symbols", "coordinates_angstrom", "total_charge",
          "low_spin_multiplicity", "high_spin_multiplicity")
SERVER = '''import json,sys
from pathlib import Path
import numpy as np
from p1_geometry_descriptor import featurize
from p1_frozen_predictor import predict
with np.load(Path(__file__).with_name('model.npz'),allow_pickle=False) as archive:
    assets={name:archive[name] for name in archive.files}
for line in sys.stdin:
    request=json.loads(line);records=request['records']
    values=predict(np.array([featurize(r) for r in records]),assets,FAMILY)
    print(json.dumps({'predictions':[{'sample_id':r['sample_id'],'predicted_gap_kcal_mol':float(v)} for r,v in zip(records,values)]},allow_nan=False),flush=True)
'''


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def input_sha256(path):
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                    encoding="utf-8")


def load_public(path):
    value = json.loads(path.read_text(encoding="utf-8"),
                       parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
    if not isinstance(value, dict) or not isinstance(value.get("records"), list):
        raise ValueError("Expected the public training/development records object")
    records = value["records"]
    if not records:
        raise ValueError("Empty public split")
    targets = []
    for record in records:
        target = record.get("gap_kcal_mol")
        if isinstance(target, bool) or not isinstance(target, (int, float)) or not math.isfinite(target):
            raise ValueError("Expected a finite numeric public training/development target")
        targets.append(float(target))
    # Ignore identifiers and all target-derived/source-only fields when featurizing.
    features = np.array([featurize({key: record[key] for key in FIELDS}) for record in records])
    return records, features, np.array(targets)


def tree_arrays(model):
    columns = {key: [] for key in ("left", "right", "feature", "threshold", "value")}
    roots, offset = [], 0
    for estimator in model.estimators_:
        tree = estimator.tree_
        roots.append(offset)
        columns["left"].append(np.where(tree.children_left < 0, -1, tree.children_left + offset))
        columns["right"].append(np.where(tree.children_right < 0, -1, tree.children_right + offset))
        columns["feature"].append(tree.feature)
        columns["threshold"].append(tree.threshold)
        columns["value"].append(tree.value.ravel())
        offset += tree.node_count
    arrays = {key: np.concatenate(value) for key, value in columns.items()}
    arrays["roots"] = np.array(roots)
    return arrays


def save_solution(output, family, arrays):
    folder = output / ("geometry_" + family) / "solution"
    folder.mkdir(parents=True, exist_ok=False)
    np.savez(folder / "model.npz", **arrays)
    module_dir = Path(__file__).resolve().parent
    for name in ("p1_geometry_descriptor.py", "p1_frozen_predictor.py"):
        shutil.copyfile(module_dir / name, folder / name)
    (folder / "main.py").write_text(SERVER.replace("FAMILY", repr(family)), encoding="utf-8")
    return folder


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training", type=Path, required=True)
    parser.add_argument("--development", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    actual_versions = {package: importlib.metadata.version(package) for package in PINNED}
    if actual_versions != PINNED:
        raise ValueError("Pinned numerical runtime required: " + repr(actual_versions))
    for split in EXPECTED_INPUTS:
        if input_sha256(getattr(args, split)) != EXPECTED_INPUTS[split]:
            raise ValueError("Expected the frozen public input JSON for " + split)
    if args.output.exists():
        raise ValueError("Use a new output directory; frozen baseline files are not overwritten")
    started = time.monotonic()
    train, x, y = load_public(args.training)
    _, dx, dy = load_public(args.development)
    args.output.mkdir(parents=True)
    print("Public descriptors ready; fitting the frozen configurations", flush=True)
    x_scaler = StandardScaler().fit(x)
    scaled_x, scaled_dx = x_scaler.transform(x), x_scaler.transform(dx)
    y_scaler = StandardScaler().fit(y[:, None])
    scaled_y = y_scaler.transform(y[:, None])[:, 0]
    results = {}
    model = ExtraTreesRegressor(**FROZEN["trees"]).fit(x, y)
    arrays = tree_arrays(model)
    expected = model.predict(dx)
    folder = save_solution(args.output, "trees", arrays)
    numeric = predict(dx, arrays, "trees")
    results["trees"] = {"development_mae": float(np.mean(np.abs(numeric - dy))),
                         "numeric_inference_max_error": float(np.max(np.abs(numeric - expected))),
                         "asset_sha256": sha256(folder / "model.npz")}
    print("Frozen trees rebuilt", flush=True)
    model = KernelRidge(**FROZEN["krr"]).fit(scaled_x, y)
    arrays = dict(mean=x_scaler.mean_, scale=x_scaler.scale_, train_x=scaled_x,
                  dual_coef=model.dual_coef_, gamma=FROZEN["krr"]["gamma"])
    expected = model.predict(scaled_dx)
    folder = save_solution(args.output, "krr", arrays)
    numeric = predict(dx, arrays, "krr")
    results["krr"] = {"development_mae": float(np.mean(np.abs(numeric - dy))),
                       "numeric_inference_max_error": float(np.max(np.abs(numeric - expected))),
                       "asset_sha256": sha256(folder / "model.npz")}
    print("Frozen kernel ridge model rebuilt", flush=True)
    members, warnings_by_member = [], []
    for seed in FROZEN["mlp_seeds"]:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            model = MLPRegressor(**FROZEN["mlp"], random_state=seed).fit(scaled_x, scaled_y)
        members.append(model)
        warnings_by_member.append([str(item.message) for item in caught])
        print("Frozen neural member fitted", seed, model.n_iter_, flush=True)
    arrays = dict(x_mean=x_scaler.mean_, x_scale=x_scaler.scale_, y_mean=y_scaler.mean_[0],
                  y_scale=y_scaler.scale_[0], members=len(members), layers=len(members[0].coefs_))
    for k, member in enumerate(members):
        for j, (weights, bias) in enumerate(zip(member.coefs_, member.intercepts_)):
            arrays[f"w_{k}_{j}"] = weights
            arrays[f"b_{k}_{j}"] = bias
    expected = np.mean([member.predict(scaled_dx) * y_scaler.scale_[0] + y_scaler.mean_[0]
                        for member in members], axis=0)
    folder = save_solution(args.output, "mlp", arrays)
    numeric = predict(dx, arrays, "mlp")
    results["mlp"] = {"development_mae": float(np.mean(np.abs(numeric - dy))),
                       "numeric_inference_max_error": float(np.max(np.abs(numeric - expected))),
                       "training_iterations": [int(member.n_iter_) for member in members],
                       "convergence_warnings": warnings_by_member,
                       "asset_sha256": sha256(folder / "model.npz")}
    folder = args.output / "training_median" / "solution"
    folder.mkdir(parents=True)
    median = float(np.median(y))
    (folder / "main.py").write_text(
        "import json,sys\nfor line in sys.stdin:\n r=json.loads(line)\n print(json.dumps({'predictions':"
        "[{'sample_id':v['sample_id'],'predicted_gap_kcal_mol':" + repr(median) +
        "} for v in r['records']]},allow_nan=False),flush=True)\n", encoding="utf-8")
    results["training_median"] = {"value": median, "development_mae": float(np.mean(np.abs(dy - median)))}
    for family in ("trees", "krr", "mlp"):
        if results[family]["numeric_inference_max_error"] > 1e-8:
            raise ValueError("Numeric inference verification failed: " + family)
    dump(args.output / "rebuild_report.json", {
        "protocol": "2026-10-02-p1-synthetic-public-input-rebuild-v1", "selected_parameters": FROZEN,
        "selection": "Parameters were frozen after the original development-only comparison. "
                     "This recipe does not search parameters or refit on development labels.",
        "input_sha256": {"training": input_sha256(args.training), "development": input_sha256(args.development)},
        "counts": {"training": len(train), "development": len(dy)},
        "fitting_fields": list(FIELDS) + ["training gap_kcal_mol"],
        "development_targets_used_for_reporting_only": True,
        "requires_final_data_or_labels": False,
        "requires_construction_ids_or_feature_cache": False,
        "versions": actual_versions, "python": platform.python_version(),
        "results": results, "elapsed_seconds": time.monotonic() - started,
        "release_ready": False,
        "release_gates": ["Explicit scientific review of the changed generated-geometry B3LYP protocol",
                          "Public frozen baseline and recipe links",
                          "Actual operator SDK/image compatibility"],
    })
    print("P1_PUBLIC_INPUT_REBUILD_PASS", flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
