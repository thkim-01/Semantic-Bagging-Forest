"""Extract ontology-driven refinements from DTO.owl for a given CSV/dataset.

This script is meant as a practical bridge:
- Load DTO.owl as the base ontology
- Create Molecule individuals from a CSV (via RDKit feature extraction)
- Generate candidate refinements from the ontology + observed instance structure
- Dump the resulting refinement list to a text file

Examples:
    # 1) Direct CSV mode
    python experiments/extract_dto_refinements.py --csv data/bbbp/BBBP.csv --smiles-col smiles --label-col p_np --limit 500

    # 2) Dataset mode (uses experiments/benchmark_config.py)
    python experiments/extract_dto_refinements.py --dataset bbbp --target p_np --limit 500

Notes:
- Refinements are data-dependent (thresholds and observed concept types are derived from the
  provided instances), but the *schema* comes from DTO.owl.
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional

# Add project root to path so `import src.*` works when running as a script
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

import pandas as pd

from src.ontology.molecule_ontology import MoleculeOntology
from src.ontology.smiles_converter import MolecularFeatureExtractor
from src.sdt.logic_refinement import (
    OntologyRefinementGenerator,
    save_refinements_json,
)


def _load_from_dataset_config(dataset: str, target: Optional[str]):
    """Return (csv_path, smiles_col, targets_cfg, targets) from benchmark_config."""
    # Local import keeps this script usable from other contexts.
    from experiments.benchmark_config import DATASET_CONFIG

    if dataset not in DATASET_CONFIG:
        available = sorted(DATASET_CONFIG.keys())
        raise ValueError(
            f"Unknown dataset '{dataset}'. Available: {available}"
        )

    cfg = DATASET_CONFIG[dataset]
    csv_path = Path(cfg['path'])
    smiles_col = cfg.get('smiles_col', 'smiles')

    targets_cfg = cfg.get('targets')
    if targets_cfg == 'ALL_EXCEPT_SMILES':
        # We'll resolve actual targets after loading the CSV.
        targets = []
    elif isinstance(targets_cfg, list):
        targets = targets_cfg
    else:
        targets = [str(targets_cfg)]

    if target is not None:
        targets = [target]

    return csv_path, smiles_col, targets_cfg, targets


def _build_instances(
    onto: MoleculeOntology,
    df: pd.DataFrame,
    smiles_col: str,
    label_col: Optional[str],
) -> List:
    extractor = MolecularFeatureExtractor()
    instances = []
    failed = 0

    for i, row in df.iterrows():
        try:
            smi = row[smiles_col]
            feats = extractor.extract_features(smi)

            if label_col is None:
                label = 0
            else:
                label = int(row[label_col])

            inst = onto.add_molecule_instance(f"Mol_{i}", feats, label=label)
            instances.append(inst)
        except Exception:
            failed += 1
            continue

    if failed:
        print(f"Skipped {failed} rows due to SMILES/feature errors")

    return instances


def main() -> int:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--csv", help="Path to CSV under data/")
    group.add_argument(
        "--dataset",
        help=(
            "Dataset key from experiments/benchmark_config.py "
            "(e.g., bbbp, bace) "
            "or 'all'"
        ),
    )

    parser.add_argument(
        "--target",
        default=None,
        help=(
            "Target/label column (dataset mode). If omitted, runs all targets."
        ),
    )

    parser.add_argument(
        "--smiles-col",
        default=None,
        help="SMILES column name",
    )
    parser.add_argument("--label-col", default=None, help="Label column name")
    parser.add_argument(
        "--limit",
        type=int,
        default=500,
        help="Number of rows to use (keeps refinement generation fast)",
    )
    parser.add_argument(
        "--out",
        default=None,
        help=(
            "Output file path (CSV mode). "
            "In dataset mode, outputs are written under "
            "output/dto_refinements/<dataset>/<target>.txt by default."
        ),
    )

    args = parser.parse_args()

    if args.csv:
        # Use a throwaway ontology file name; we do not need to save it.
        onto = MoleculeOntology("ontology/_tmp_dto_extract.owl")
        generator = OntologyRefinementGenerator(onto)

        csv_path = Path(args.csv)
        if not csv_path.exists():
            raise FileNotFoundError(str(csv_path))

        df = pd.read_csv(csv_path)
        smiles_col = args.smiles_col or "smiles"
        label_col = args.label_col

        if smiles_col not in df.columns:
            raise ValueError(
                f"SMILES column '{smiles_col}' not found in {csv_path}"
            )
        if label_col is not None and label_col not in df.columns:
            raise ValueError(
                f"Label column '{label_col}' not found in {csv_path}"
            )

        df = df.dropna(subset=[smiles_col])
        if label_col is not None:
            df = df.dropna(subset=[label_col])
        if args.limit and len(df) > args.limit:
            df = df.head(args.limit)

        instances = _build_instances(onto, df, smiles_col, label_col)
        if not instances:
            raise RuntimeError(
                "No valid instances could be built from the CSV"
            )

        refinements = generator.generate_refinements(onto.Molecule, instances)

        out_path = Path(args.out or "output/dto_refinements.txt")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            f.write(f"CSV: {csv_path}\n")
            f.write(f"Instances used: {len(instances)}\n")
            f.write(f"Refinements: {len(refinements)}\n")
            f.write("\n")
            for r in refinements:
                f.write(str(r) + "\n")

        # Also save machine-readable JSON for static mode
        json_path = str(out_path.with_suffix('.json'))
        save_refinements_json(
            refinements,
            json_path,
            metadata={
                'mode': 'csv',
                'csv': str(csv_path),
                'smiles_col': smiles_col,
                'label_col': label_col,
                'instances_used': len(instances),
            },
        )

        print(f"Wrote {len(refinements)} refinements to {out_path}")
        return 0

    # Dataset mode
    from experiments.benchmark_config import DATASET_CONFIG

    datasets = [args.dataset]
    if args.dataset == 'all':
        datasets = list(DATASET_CONFIG.keys())

    for dataset in datasets:
        csv_path, cfg_smiles, targets_cfg, targets = _load_from_dataset_config(
            dataset,
            args.target,
        )
        if not csv_path.exists():
            print(f"[SKIP] Missing CSV: {csv_path}")
            continue

        df = pd.read_csv(csv_path)
        smiles_col = args.smiles_col or cfg_smiles
        if smiles_col not in df.columns:
            print(f"[SKIP] {dataset}: SMILES column '{smiles_col}' not found")
            continue

        # Resolve "ALL_EXCEPT_SMILES" targets now that we have columns.
        if targets_cfg == 'ALL_EXCEPT_SMILES':
            targets = [
                c for c in df.columns
                if c not in (smiles_col, 'mol_id')
            ]
            if args.target is not None:
                targets = [args.target]

        if not targets:
            print(f"[SKIP] {dataset}: no targets resolved")
            continue

        base_out_dir = Path("output") / "dto_refinements" / str(dataset)
        base_out_dir.mkdir(parents=True, exist_ok=True)

        # Clean SMILES NaNs once.
        df = df.dropna(subset=[smiles_col])
        if args.limit and len(df) > args.limit:
            df = df.head(args.limit)

        for label_col in targets:
            if label_col not in df.columns:
                continue

            df_target = df.dropna(subset=[label_col])
            if df_target.empty:
                continue

            # IMPORTANT: isolate per (dataset,target) so individuals don't
            # accumulate.
            tmp_onto_path = (
                f"ontology/_tmp_dto_extract_{dataset}_{label_col}.owl"
            )
            onto = MoleculeOntology(tmp_onto_path)
            generator = OntologyRefinementGenerator(onto)

            instances = _build_instances(
                onto,
                df_target,
                smiles_col,
                label_col,
            )
            if not instances:
                continue

            refinements = generator.generate_refinements(
                onto.Molecule,
                instances,
            )
            out_path = base_out_dir / f"{label_col}.txt"

            with out_path.open("w", encoding="utf-8") as f:
                f.write(f"Dataset: {dataset}\n")
                f.write(f"CSV: {csv_path}\n")
                f.write(f"SMILES column: {smiles_col}\n")
                f.write(f"Target: {label_col}\n")
                f.write(f"Instances used: {len(instances)}\n")
                f.write(f"Refinements: {len(refinements)}\n")
                f.write("\n")
                for r in refinements:
                    f.write(str(r) + "\n")

            json_path = str(out_path.with_suffix('.json'))
            save_refinements_json(
                refinements,
                json_path,
                metadata={
                    'mode': 'dataset',
                    'dataset': dataset,
                    'csv': str(csv_path),
                    'smiles_col': smiles_col,
                    'target': label_col,
                    'instances_used': len(instances),
                },
            )

            print(f"[{dataset}/{label_col}] {len(refinements)} -> {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
