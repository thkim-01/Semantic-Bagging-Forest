# flake8: noqa

"""Multi-dataset evaluation for Semantic Bagging Forest.

This script evaluates the bagging ensemble on multiple MoleculeNet-style
classification datasets included in `data/`.

By default it runs one representative task per dataset to keep runtime practical.
Use `--all-tasks` to evaluate all tasks for multi-task datasets (Tox21, SIDER).
"""

import argparse
import csv
import os
import re
import sys
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from typing import Optional


sys.path.append(str(Path(__file__).parent.parent))

from src.ontology.molecule_ontology import MoleculeOntology
from src.ontology.smiles_converter import MolecularFeatureExtractor
from src.sdt.logic_forest import SemanticForest


def _safe_name(name: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(name))
    return s[:80].strip("_") or "task"


def _get_git_commit_info() -> tuple[str, str]:
    """Return (short_sha, subject) for current git HEAD.

    If git is unavailable (e.g., not a repo), returns empty strings.
    """
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        msg = subprocess.run(
            ["git", "log", "-1", "--pretty=%s"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        return sha, msg
    except Exception:
        return "", ""


def _normalize_binary_labels(series: pd.Series) -> pd.Series:
    """Normalize common binary label encodings.

    - drops missing values
    - treats -1 as missing (common in some datasets)
    - supports {-1, 1} by mapping to {0, 1} after dropping missing
    """
    y = pd.to_numeric(series, errors="coerce")
    # In some MoleculeNet dumps, -1 is used as missing.
    y = y.replace(-1, np.nan)
    y = y.dropna()

    uniq = set(y.unique().tolist())
    if uniq == {1.0, 0.0}:
        return y.astype(int)
    if uniq == {1.0, -1.0}:
        # If -1 survived (shouldn't), map it.
        return y.replace(-1, 0).astype(int)
    if uniq == {1.0} or uniq == {0.0}:
        return y.astype(int)

    # Best-effort: if values are already 0/1-like
    if uniq.issubset({0.0, 1.0}):
        return y.astype(int)

    raise ValueError(f"Non-binary labels found: {sorted(uniq)}")


def populate_ontology(
    onto: MoleculeOntology,
    extractor: MolecularFeatureExtractor,
    df: pd.DataFrame,
    smiles_col: str,
    label_col: str,
    subset_name: str,
):
    instances = []
    labels = []

    for idx, row in df.iterrows():
        try:
            smi = row[smiles_col]
            feats = extractor.extract_features(smi)
            mol_id = f"Mol_{subset_name}_{idx}"
            label_val = int(row[label_col])
            inst = onto.add_molecule_instance(mol_id, feats, label=label_val)
            instances.append(inst)
            labels.append(label_val)
        except Exception:
            # Skip invalid SMILES / feature extraction failures.
            continue

    return instances, labels


def evaluate_task(
    dataset_key: str,
    dataset_name: str,
    csv_path: str,
    smiles_col: str,
    label_col: str,
    feature_cache_path: Optional[str],
    split_criterion: str,
    n_estimators: int,
    max_depth: int,
    min_samples_split: int,
    min_samples_leaf: int,
    sample_size: int,
    test_size: float,
    random_state: int,
):
    df = pd.read_csv(csv_path)

    if smiles_col not in df.columns:
        raise ValueError(f"Missing smiles column '{smiles_col}' in {csv_path}")
    if label_col not in df.columns:
        raise ValueError(f"Missing label column '{label_col}' in {csv_path}")

    # Normalize labels (drop missing)
    y_norm = _normalize_binary_labels(df[label_col])
    df = df.loc[y_norm.index].copy()
    df[label_col] = y_norm

    if sample_size and len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=random_state).reset_index(drop=True)

    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df[label_col] if df[label_col].nunique() > 1 else None,
    )

    onto_path = Path("ontology") / f"temp_bagging_{dataset_key}_{_safe_name(label_col)}.owl"
    if onto_path.exists():
        onto_path.unlink()

    onto = MoleculeOntology(str(onto_path))
    extractor = MolecularFeatureExtractor(cache_path=feature_cache_path)

    try:
        train_instances, _ = populate_ontology(
            onto,
            extractor,
            train_df,
            smiles_col,
            label_col,
            subset_name="Train",
        )
        test_instances, test_labels = populate_ontology(
            onto,
            extractor,
            test_df,
            smiles_col,
            label_col,
            subset_name="Test",
        )

        # If feature extraction filtered too much, bail out.
        if len(train_instances) < max(min_samples_split, 50) or len(test_instances) < 50:
            return {
                "dataset": dataset_name,
                "task": label_col,
                "n_train": len(train_instances),
                "n_test": len(test_instances),
                "auc": np.nan,
                "acc": np.nan,
                "note": "too_few_valid_instances",
            }

        forest = SemanticForest(
            onto,
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            class_weight="balanced",
            verbose=False,
            learner_kwargs={
                "split_criterion": split_criterion,
            },
        )
        forest.fit(train_instances)

        probs = forest.predict_proba(test_instances)
        preds = forest.predict(test_instances)

        acc = accuracy_score(test_labels, preds)
        try:
            auc = roc_auc_score(test_labels, probs)
        except ValueError:
            auc = 0.5

        return {
            "dataset": dataset_name,
            "task": label_col,
            "n_train": len(train_instances),
            "n_test": len(test_instances),
            "auc": float(auc),
            "acc": float(acc),
            "note": "",
        }
    finally:
        extractor.close()


def _append_result(out_path: Path, res: dict) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "dataset",
        "task",
        "n_train",
        "n_test",
        "auc",
        "acc",
        "note",
        "commit_sha",
        "commit_message",
    ]
    write_header = not out_path.exists() or out_path.stat().st_size == 0

    with out_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow({k: res.get(k, "") for k in fieldnames})


def _ensure_commit_columns(out_path: Path, commit_sha: str, commit_msg: str) -> None:
    """Ensure existing CSV has commit columns; fill missing values.

    This keeps resume behavior intact while making old rows traceable.
    """
    if not out_path.exists() or out_path.stat().st_size == 0:
        return

    try:
        df = pd.read_csv(out_path)
    except Exception:
        return

    changed = False
    if "commit_sha" not in df.columns:
        df["commit_sha"] = ""
        changed = True
    if "commit_message" not in df.columns:
        df["commit_message"] = ""
        changed = True

    # Fill blanks/NaNs with current commit info so historical rows are still
    # traceable to "the code version that produced this file".
    if commit_sha:
        before = df["commit_sha"].isna().sum() + (df["commit_sha"] == "").sum()
        df["commit_sha"] = df["commit_sha"].fillna("")
        df.loc[df["commit_sha"] == "", "commit_sha"] = commit_sha
        after = df["commit_sha"].isna().sum() + (df["commit_sha"] == "").sum()
        if before != after:
            changed = True

    if commit_msg:
        before = df["commit_message"].isna().sum() + (df["commit_message"] == "").sum()
        df["commit_message"] = df["commit_message"].fillna("")
        df.loc[df["commit_message"] == "", "commit_message"] = commit_msg
        after = df["commit_message"].isna().sum() + (df["commit_message"] == "").sum()
        if before != after:
            changed = True

    if changed:
        # Keep original columns order as much as possible; append commit columns.
        cols = list(df.columns)
        # Ensure commit columns are at the end for readability
        for c in ["commit_sha", "commit_message"]:
            if c in cols:
                cols.remove(c)
        cols += ["commit_sha", "commit_message"]
        df = df[cols]
        df.to_csv(out_path, index=False)


def _write_dataset_averages(in_path: Path, out_path: Path) -> None:
    """Write macro averages per dataset over completed tasks.

    - Excludes rows with NaN metrics from mean/std computations.
    - Keeps counts so partial/resumed runs are still meaningful.
    """
    if not in_path.exists() or in_path.stat().st_size == 0:
        return

    try:
        df = pd.read_csv(in_path)
    except Exception:
        return

    required = {
        "dataset",
        "task",
        "n_train",
        "n_test",
        "auc",
        "acc",
        "note",
        "commit_sha",
        "commit_message",
    }
    if not required.issubset(set(df.columns)):
        return

    # Coerce numeric columns
    for c in ["n_train", "n_test", "auc", "acc"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Only compute stats on rows with finite metrics.
    valid = df[df["auc"].notna() & df["acc"].notna()].copy()

    rows = []
    group_cols = ["commit_sha", "commit_message", "dataset"]

    for (csha, cmsg, dataset), df_ds in df.groupby(group_cols):
        df_valid = valid[
            (valid["commit_sha"] == csha)
            & (valid["commit_message"] == cmsg)
            & (valid["dataset"] == dataset)
        ]

        rows.append(
            {
                "commit_sha": csha,
                "commit_message": cmsg,
                "dataset": dataset,
                "n_tasks": int(df_ds["task"].nunique()),
                "n_rows": int(len(df_ds)),
                "n_valid": int(len(df_valid)),
                "auc_mean": float(df_valid["auc"].mean()) if len(df_valid) else np.nan,
                "auc_std": float(df_valid["auc"].std(ddof=0)) if len(df_valid) else np.nan,
                "acc_mean": float(df_valid["acc"].mean()) if len(df_valid) else np.nan,
                "acc_std": float(df_valid["acc"].std(ddof=0)) if len(df_valid) else np.nan,
                "n_train_mean": float(df_valid["n_train"].mean()) if len(df_valid) else np.nan,
                "n_test_mean": float(df_valid["n_test"].mean()) if len(df_valid) else np.nan,
            }
        )

    # Overall macro average across all valid rows per commit
    if len(valid):
        for (csha, cmsg), df_valid in valid.groupby(["commit_sha", "commit_message"]):
            df_all = df[(df["commit_sha"] == csha) & (df["commit_message"] == cmsg)]
            rows.append(
                {
                    "commit_sha": csha,
                    "commit_message": cmsg,
                    "dataset": "ALL",
                    "n_tasks": int(df_all["task"].nunique()),
                    "n_rows": int(len(df_all)),
                    "n_valid": int(len(df_valid)),
                    "auc_mean": float(df_valid["auc"].mean()),
                    "auc_std": float(df_valid["auc"].std(ddof=0)),
                    "acc_mean": float(df_valid["acc"].mean()),
                    "acc_std": float(df_valid["acc"].std(ddof=0)),
                    "n_train_mean": float(df_valid["n_train"].mean()),
                    "n_test_mean": float(df_valid["n_test"].mean()),
                }
            )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df = pd.DataFrame(rows)
    out_df = out_df.sort_values(["dataset"]).reset_index(drop=True)
    out_df.to_csv(out_path, index=False)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Semantic Bagging Forest across datasets"
    )
    parser.add_argument("--n-estimators", type=int, default=5)
    parser.add_argument("--max-depth", type=int, default=10)
    parser.add_argument("--min-samples-split", type=int, default=20)
    parser.add_argument("--min-samples-leaf", type=int, default=5)
    parser.add_argument("--sample-size", type=int, default=2000)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--split-criterion",
        default="gain_ratio",
        choices=["information_gain", "gain_ratio", "c45_gain_ratio"],
        help=(
            "Split criterion for tree growth. 'gain_ratio' matches C4.5's "
            "gain ratio; 'information_gain' matches ID3-style info gain."
        ),
    )
    parser.add_argument(
        "--all-tasks",
        action="store_true",
        help="Evaluate all tasks for multi-task datasets (Tox21, SIDER).",
    )
    parser.add_argument(
        "--datasets",
        default=None,
        help=(
            "Comma-separated dataset keys to run (e.g., 'clintox' or 'tox21,sider'). "
            "If omitted, runs all datasets."
        ),
    )
    parser.add_argument(
        "--out",
        default=str(Path("output") / "semantic_forest_benchmark.csv"),
        help="Output CSV path.",
    )

    parser.add_argument(
        "--out-avg",
        default=None,
        help=(
            "Optional output CSV path for per-dataset macro averages over tasks. "
            "If omitted, uses '<out>_avg.csv'."
        ),
    )

    parser.add_argument(
        "--run-message",
        default=None,
        help=(
            "Override the commit_message written to result CSVs. "
            "Useful when you pre-decide the commit message and want the CSV "
            "to match it 1:1."
        ),
    )
    parser.add_argument(
        "--run-sha",
        default=None,
        help=(
            "Override the commit_sha written to result CSVs. "
            "Normally detected from git; only set this if you know what you're doing."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output CSV if it already exists (disables resume).",
    )
    parser.add_argument(
        "--feature-cache-dir",
        default=str(Path("output") / "feature_cache"),
        help=(
            "Directory for persistent SMILES->features cache (SQLite). "
            "Set to empty string to disable caching."
        ),
    )
    args = parser.parse_args()

    commit_sha, commit_msg = _get_git_commit_info()
    if args.run_sha is not None:
        commit_sha = str(args.run_sha)
    if args.run_message is not None:
        commit_msg = str(args.run_message)

    out_path = Path(args.out)
    out_avg_path = (
        Path(args.out_avg)
        if args.out_avg
        else out_path.with_name(out_path.stem + "_avg.csv")
    )
    if args.overwrite and out_path.exists():
        out_path.unlink()

    # Backfill commit columns for existing output (helps tracking + keeps resume).
    _ensure_commit_columns(out_path, commit_sha, commit_msg)

    completed: set[tuple[str, str]] = set()
    if out_path.exists() and not args.overwrite:
        try:
            prev = pd.read_csv(out_path)
            if {"dataset", "task"}.issubset(set(prev.columns)):
                # Resume is commit-aware: only skip tasks already computed for
                # the current commit_sha.
                if commit_sha and "commit_sha" in prev.columns:
                    prev = prev[prev["commit_sha"] == commit_sha]
                completed = set(zip(prev["dataset"], prev["task"]))
        except Exception:
            completed = set()

    datasets = [
        {
            "key": "bbbp",
            "name": "BBBP",
            "path": "data/bbbp/BBBP.csv",
            "smiles": "smiles",
            "tasks": ["p_np"],
        },
        {
            "key": "bace",
            "name": "BACE",
            "path": "data/bace/bace.csv",
            "smiles": "smiles",
            "tasks": ["Class"],
        },
        {
            "key": "clintox",
            "name": "ClinTox",
            "path": "data/clintox/clintox.csv",
            "smiles": "smiles",
            "tasks": ["CT_TOX", "FDA_APPROVED"],
            "default_task": "CT_TOX",
        },
        {
            "key": "hiv",
            "name": "HIV",
            "path": "data/hiv/HIV.csv",
            "smiles": "smiles",
            "tasks": ["HIV_active"],
        },
        {
            "key": "tox21",
            "name": "Tox21",
            "path": "data/tox21/tox21.csv",
            "smiles": "smiles",
            "tasks": [
                "NR-AR",
                "NR-AR-LBD",
                "NR-AhR",
                "NR-Aromatase",
                "NR-ER",
                "NR-ER-LBD",
                "NR-PPAR-gamma",
                "SR-ARE",
                "SR-ATAD5",
                "SR-HSE",
                "SR-MMP",
                "SR-p53",
            ],
            "default_task": "SR-p53",
        },
        {
            "key": "sider",
            "name": "SIDER",
            "path": "data/sider/sider.csv",
            "smiles": "smiles",
            "tasks": None,  # determined from header
            "default_task": "Hepatobiliary disorders",
        },
    ]

    if args.datasets:
        available_keys = [d["key"] for d in datasets]
        wanted = {
            s.strip().lower()
            for s in str(args.datasets).split(",")
            if s.strip()
        }
        datasets = [ds for ds in datasets if ds["key"].lower() in wanted]
        if not datasets:
            raise ValueError(
                f"No datasets matched --datasets={args.datasets!r}. "
                f"Available: {available_keys}"
            )

    try:
        for ds in datasets:
            tasks = ds.get("tasks")
            if tasks is None:
                # SIDER: all columns except smiles
                df_cols = pd.read_csv(ds["path"], nrows=1).columns.tolist()
                tasks = [c for c in df_cols if c != ds["smiles"]]

            if not args.all_tasks:
                # Use one representative task for multi-task datasets.
                if "default_task" in ds:
                    tasks = [ds["default_task"]]

            for task in tasks:
                if (ds["name"], task) in completed:
                    print(f"Skipping (already in output): {ds['name']}/{task}")
                    continue

                try:
                    cache_dir = str(args.feature_cache_dir).strip()
                    cache_path = None
                    if cache_dir:
                        cache_path = str(Path(cache_dir) / f"{ds['key']}.sqlite3")

                    res = evaluate_task(
                        dataset_key=ds["key"],
                        dataset_name=ds["name"],
                        csv_path=ds["path"],
                        smiles_col=ds["smiles"],
                        label_col=task,
                        feature_cache_path=cache_path,
                        split_criterion=args.split_criterion,
                        n_estimators=args.n_estimators,
                        max_depth=args.max_depth,
                        min_samples_split=args.min_samples_split,
                        min_samples_leaf=args.min_samples_leaf,
                        sample_size=args.sample_size,
                        test_size=args.test_size,
                        random_state=args.random_state,
                    )
                except Exception as e:
                    res = {
                        "dataset": ds["name"],
                        "task": task,
                        "n_train": 0,
                        "n_test": 0,
                        "auc": np.nan,
                        "acc": np.nan,
                        "note": f"error: {e}",
                    }

                # Attach commit info for traceability
                res["commit_sha"] = commit_sha
                res["commit_message"] = commit_msg

                _append_result(out_path, res)
                completed.add((ds["name"], task))
                print(
                    f"{res['dataset']}/{res['task']}: "
                    f"AUC={res['auc']}, ACC={res['acc']} "
                    f"(train={res['n_train']}, test={res['n_test']}) {res['note']}"
                )
    finally:
        # Always refresh the macro-average file, even on interruption.
        _write_dataset_averages(out_path, out_avg_path)
        print(f"\nSaved: {out_path}")
        print(f"Saved averages: {out_avg_path}")


if __name__ == "__main__":
    main()
