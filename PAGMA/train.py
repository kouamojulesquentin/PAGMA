"""Phase 2: Training and comparison of FAIL/PASS classifiers on the datasets.

Key points
- Split BY CIRCUIT (filename) and stratified by fault class: the same DUT (its 5 corners, etc.) is never
  both in operation and under test at the same time. The split depends only on (seed, DUT): it is identical from one stimulus to the next,
  so the comparison between stimuli is fair.
- The `nominal` rows serve as a reference (--residual option) and are excluded from the training and test sets.
- Cumulative results are stored in runs/<circuit>/results/summary.csv.
"""
import csv
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

MODEL_NAMES = ("logreg", "svm", "random_forest", "gboost", "mlp")
DEFAULT_MODELS = ("logreg", "random_forest", "mlp")
LABELS = ("hard_label", "spec_label", "checker_label", "fault_class")
FEATURES = ("raw", "stats", "raw+stats")


def make_model(name, seed):
    if name == "logreg":
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced"))
    if name == "svm":
        return make_pipeline(StandardScaler(), SVC(kernel="rbf", C=10.0, class_weight="balanced"))
    if name == "random_forest":
        return RandomForestClassifier(n_estimators=300, class_weight="balanced", n_jobs=-1, random_state=seed)
    if name == "gboost":
        return GradientBoostingClassifier(n_estimators=200, random_state=seed)
    if name == "mlp":
        return make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(128, 64), early_stopping=True,
                                                             max_iter=400, random_state=seed))
    raise ValueError("modele inconnu: %s (choix: %s)" % (name, ", ".join(MODEL_NAMES)))


def stat_features(X):
    """Compact descriptors of a waveform: moments, extrema, slope, energy per frequency band."""
    d = np.diff(X, axis=1)
    n = X.shape[1]
    cols = [X.mean(1), X.std(1), X.min(1), X.max(1), np.ptp(X, axis=1), np.sqrt((X ** 2).mean(1)),
            np.abs(d).mean(1), d.max(1), d.min(1), X.argmax(1) / float(n), X.argmin(1) / float(n), X[:, -1] - X[:, 0]]
    spec = np.abs(np.fft.rfft(X - X.mean(1, keepdims=True), axis=1))
    cols += [np.log1p(b.sum(1)) for b in np.array_split(spec, 8, axis=1)]
    return np.column_stack(cols)


def load_dataset(path):
    df = pd.read_csv(str(path), dtype={"filename": str, "corner": str, "fault_class": str})
    df["corner"] = df["corner"].fillna("")
    feats = [c for c in df.columns if c.startswith("t_")]
    if not feats:
        raise ValueError("%s : No t_* columns (invalid dataset)" % path)
    X = df[feats].to_numpy(dtype=float)
    if not np.isfinite(X).all():
        raise ValueError("%s : Infinite values in waveforms" % path)
    return df, X


def nominal_reference(df, X):
    ref = {}
    for corner in sorted(df["corner"].unique()):
        m = ((df["fault_class"] == "nominal") & (df["corner"] == corner)).to_numpy()
        if not m.any():
            raise ValueError("No nominal line for the '%s' corner: --residual not possible"
                             "(Run 'dataset' again without setting --limit too low)" % corner)
        ref[corner] = X[m].mean(0)
    return ref


def build_features(X, mode):
    if mode == "raw":
        return X
    if mode == "stats":
        return stat_features(X)
    return np.hstack([X, stat_features(X)])


def assign_folds(df, n_folds, seed):
    """Fold by DUT, stratify by fault_class; depends only on (seed, DUT names)."""
    rng = np.random.RandomState(seed)
    fold_of = {}
    for cls in sorted(df["fault_class"].unique()):
        files = sorted(df.loc[df["fault_class"] == cls, "filename"].unique())
        for rank, i in enumerate(rng.permutation(len(files))):
            fold_of[files[i]] = rank % n_folds
    return df["filename"].map(fold_of).to_numpy()


def _score(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)


def evaluate(model, X, y, classes_te, binary):
    pred = model.predict(X)
    r = {"accuracy": accuracy_score(y, pred), "balanced_acc": balanced_accuracy_score(y, pred)}
    if binary:
        r.update(precision=precision_score(y, pred, zero_division=0) if _has_zd() else precision_score(y, pred),
                 recall=recall_score(y, pred), f1=f1_score(y, pred))
        r["roc_auc"] = roc_auc_score(y, _score(model, X)) if len(set(y)) > 1 else float("nan")
        for c in sorted(set(classes_te)):     
            r["pred_fail_%s" % c] = float(np.mean(pred[classes_te == c] == 1))
    else:
        r["f1"] = f1_score(y, pred, average="macro")
        r["precision"] = precision_score(y, pred, average="macro")
        r["recall"] = recall_score(y, pred, average="macro")
        r["roc_auc"] = float("nan")
    r["_cm"] = confusion_matrix(y, pred).tolist()
    return r


def _has_zd():
    import sklearn
    return tuple(int(x) for x in sklearn.__version__.split(".")[:2]) >= (0, 22)


def train_one_dataset(path, label, models, features, residual, n_folds, run_all_folds, seed, model_dir):
    df, X = load_dataset(path)
    if label not in df.columns or (label != "fault_class" and df[label].isna().any()):
        raise ValueError("%s : Missing or incomplete '%s' label in the dataset" % (Path(path).name, label))
    ref = None
    if residual:
        ref = nominal_reference(df, X)
        X = X - np.vstack([ref[c] for c in df["corner"]])
    pop = (df["fault_class"] != "nominal").to_numpy()
    df, X = df[pop].reset_index(drop=True), X[pop]
    F = build_features(X, features)
    classes_all = df["fault_class"].to_numpy()
    binary = label != "fault_class"
    if binary:
        y = df[label].astype(int).to_numpy()
    else:
        names = sorted(set(classes_all))
        y = np.array([names.index(c) for c in classes_all])
    if len(set(y)) < 2:
        raise ValueError("Only one class in '%s': training not possible" % label)
    folds = assign_folds(df, n_folds, seed)
    stim = Path(path).stem.replace("DATASET_", "")
    out = []
    for name in models:
        res = []
        for k in (range(n_folds) if run_all_folds else [0]):
            te, tr = folds == k, folds != k
            if len(set(y[tr])) < 2 or te.sum() == 0:
                continue
            t0 = time.time()
            mdl = make_model(name, seed)
            mdl.fit(F[tr], y[tr])
            r = evaluate(mdl, F[te], y[te], classes_all[te], binary)
            r["fit_s"] = time.time() - t0
            r["n_train"], r["n_test"] = int(tr.sum()), int(te.sum())
            res.append(r)
            if not run_all_folds:
                model_dir.mkdir(parents=True, exist_ok=True)
                joblib.dump({"model": mdl, "label": label, "features": features, "residual": residual,
                             "nominal_ref": ref, "stimulus": stim, "n_points": X.shape[1]},
                            str(model_dir / ("%s_%s_%s.joblib" % (stim, name, label))))
        if not res:
            print("  ! %s/%s : No usable folds (not enough loops?)" % (stim, name))
            continue
        keys = [k for k in res[0] if not k.startswith("_") and k not in ("n_train", "n_test")]
        row = {"stimulus": stim, "label": label, "features": features, "residual": int(residual), "model": name,
               "folds": len(res), "n_train": res[0]["n_train"], "n_test": res[0]["n_test"]}
        for k in keys:
            vals = np.array([r.get(k, np.nan) for r in res], dtype=float)
            row[k] = float(np.nanmean(vals)) if not np.isnan(vals).all() else float("nan")
            if len(res) > 1 and k in ("accuracy", "f1", "roc_auc"):
                row[k + "_std"] = float(np.nanstd(vals))
        row["_cm"] = res[0]["_cm"]
        out.append(row)
        print("  %-9s %-13s acc=%.3f f1=%.3f auc=%.3f  (train=%d test=%d, %d pli%s, %.1fs)" % (
            stim, name, row["accuracy"], row["f1"], row["roc_auc"], row["n_train"], row["n_test"], len(res),
            "s" if len(res) > 1 else "", sum(r["fit_s"] for r in res)))
    return out


def run(circuit, dataset_paths, label="hard_label", models=DEFAULT_MODELS, features="raw", residual=False,
        cv=0, test_size=0.2, seed=0, results_dir=None, model_dir=None):
    n_folds = cv if cv and cv > 1 else max(2, int(round(1.0 / test_size)))
    results_dir = Path(results_dir)
    results = []
    print("[%s] train : label=%s features=%s residual=%s modeles=%s | %s" % (
        circuit, label, features, residual, ",".join(models),
        ("cross validation %d plis" % n_folds) if cv and cv > 1 else ("holdout %d%%" % round(100.0 / n_folds))))
    for p in dataset_paths:
        results += train_one_dataset(p, label, models, features, residual, n_folds, bool(cv and cv > 1), seed, model_dir)
    if not results:
        return []
    results_dir.mkdir(parents=True, exist_ok=True)
    cols = []
    for r in results:
        for k in r:
            if k != "_cm" and k not in cols:
                cols.append(k)
    path = results_dir / "summary.csv"
    # on relit l'historique pour garder un en-tete coherent si les colonnes changent
    old = []
    if path.exists():
        with open(str(path), newline="", encoding="utf-8") as f:
            old = list(csv.DictReader(f))
    allcols = list(cols)
    for r in old:
        for k in r:
            if k not in allcols:
                allcols.append(k)
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(str(path), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["date"] + allcols, extrasaction="ignore", restval="")
        w.writeheader()
        w.writerows(old)
        for r in results:
            w.writerow(dict(date=stamp, **dict((k, v) for k, v in r.items() if k != "_cm")))
    print("\nClassement (F1) :")
    print("  %-10s %-13s %-9s %6s %6s %6s" % ("stimulus", "modele", "features", "acc", "f1", "auc"))
    for r in sorted(results, key=lambda r: -r["f1"]):
        print("  %-10s %-13s %-9s %6.3f %6.3f %6.3f" % (r["stimulus"], r["model"], r["features"], r["accuracy"],
                                                        r["f1"], r["roc_auc"]))
    print("\nCumulative Results -> %s" % path)
    return results
