
import argparse
from pathlib import Path
import pandas as pd
import os
from src.ontology.molecule_ontology import MoleculeOntology
from src.ontology.smiles_converter import MolecularFeatureExtractor
from src.sdt.logic_learner import LogicSDTLearner
from src.sdt.learner import SDTLearner
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


def verify_dataset(
    name,
    csv_path,
    label_col,
    smiles_col='smiles',
    refinement_mode: str = 'dynamic',
    refinement_file: str = '',
):
    logger.info(f"\n{'='*50}\nStarting Benchmark for {name}\n{'='*50}")

    # 1. Load Data
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        logger.error(f"{csv_path} not found.")
        return None

    # Handle irregular headers/types
    if name == 'Clintox':
        # Clintox has multiple tasks, we pick 'CT_TOX' typically
        if 'CT_TOX' not in df.columns:
            # Fallback or check columns
            pass
    
    # Drop NaNs in label
    df = df.dropna(subset=[label_col])
    
    # Limit size for speed if dataset is huge (HIV is large ~40k)
    if len(df) > 2000:
        logger.info(f"Dataset {name} is large ({len(df)}). Sampling 2000 for benchmark.")
        df_sample = df.sample(n=2000, random_state=42).reset_index(drop=True)
    else:
        df_sample = df
    
    logger.info(f"Loaded {len(df_sample)} samples.")

    # Split
    train_df, test_df = train_test_split(
        df_sample,
        test_size=0.3,
        random_state=42,
    )
    logger.info(f"Train size: {len(train_df)}, Test size: {len(test_df)}")

    # ---------------------------------------------------------
    # PART A: Logic SDT
    # ---------------------------------------------------------
    onto_path = f"ontology/temp_{name}_logic.owl"
    if os.path.exists(onto_path):
        os.remove(onto_path)
        
    onto = MoleculeOntology(onto_path)
    extractor = MolecularFeatureExtractor()

    def populate_ontology(dataframe, subset_name):
        instances = []
        valid_indices = []
        for idx, row in dataframe.iterrows():
            try:
                smi = row[smiles_col]
                feats = extractor.extract_features(smi)
                
                mol_id = f"Mol_{name}_{subset_name}_{idx}"
                label_val = int(row[label_col])
                
                inst = onto.add_molecule_instance(
                    mol_id,
                    feats,
                    label=label_val,
                )
                instances.append(inst)
                valid_indices.append(idx)
            except Exception:
                pass
        return instances, valid_indices

    logger.info(f"[{name}] Populating Ontology...")
    train_instances, train_valid_idx = populate_ontology(train_df, "Train")
    test_instances, test_valid_idx = populate_ontology(test_df, "Test")
    
    logger.info(f"[{name}] Training Logic SDT...")
    # Using previous optimal params or standard
    logic_learner = LogicSDTLearner(
        onto,
        max_depth=7,
        min_samples_split=20,
        min_samples_leaf=10,
        class_weight='balanced',
        verbose=False,
        refinement_mode=refinement_mode,
        refinement_file=refinement_file,
    )
    logic_tree = logic_learner.fit(train_instances)

    # Predict Logic
    def predict_logic(tree, instances):
        preds = []
        probs = []
        for inst in instances:
            node = tree.root
            while not node.is_leaf:
                if logic_learner.refinement_generator.instance_satisfies_refinement(
                    inst,
                    node.refinement,
                ):
                    node = node.left_child
                else:
                    node = node.right_child
            preds.append(node.predicted_label)
            total = sum(node.label_counts.values())
            prob1 = node.label_counts.get(1, 0) / total if total > 0 else 0.0
            probs.append(prob1)
        return preds, probs

    logic_preds, logic_probs = predict_logic(logic_tree, test_instances)
    test_labels = [inst.hasLabel[0] for inst in test_instances]
    
    try:
        logic_auc = roc_auc_score(test_labels, logic_probs)
    except ValueError:
        logic_auc = 0.5  # Single class case
        
    logic_acc = accuracy_score(test_labels, logic_preds)
    logger.info(f"[{name}] Logic Accuracy: {logic_acc:.4f}")
    
    # ---------------------------------------------------------
    # PART B: Legacy SDT
    # ---------------------------------------------------------
    # Prepare feature matrix
    class InstanceObj:
        def __init__(self, f_dict, l):
            self.__dict__.update(f_dict)
            self.label = l
            
        def satisfies_refinement(self, refinement) -> bool:
            prop, operator, value = refinement
            if not hasattr(self, prop): return False
            feature_value = getattr(self, prop)
            if operator == '==': return feature_value == value
            elif operator == '>': return feature_value > value
            elif operator == '>=': return feature_value >= value
            elif operator == '<': return feature_value < value
            elif operator == '<=': return feature_value <= value
            elif operator == 'contains':
                if isinstance(feature_value, list): return value in feature_value
                return False
            return False

    def get_feature_matrix(dataframe, valid_indices):
        data_list = []
        labels_list = []
        subset = dataframe.loc[valid_indices]
        for _, row in subset.iterrows():
            try:
                feats = extractor.extract_features(row[smiles_col])
                flat_feats = feats.copy() 
                # keep functional_groups
                data_list.append(InstanceObj(flat_feats, int(row[label_col])))
                labels_list.append(int(row[label_col]))
            except: pass
        return data_list, labels_list

    legacy_train_data, _ = get_feature_matrix(train_df, train_valid_idx)
    legacy_test_data, legacy_test_labels = get_feature_matrix(test_df, test_valid_idx)
    
    logger.info(f"[{name}] Training Legacy SDT...")
    legacy_learner = SDTLearner(max_depth=7, min_samples_split=20, min_samples_leaf=10, class_weight='balanced')
    legacy_tree = legacy_learner.fit(legacy_train_data)
    
    def predict_legacy(tree, instances):
        preds = []
        probs = []
        for inst in instances:
            node = tree.root
            while not node.is_leaf:
                if inst.satisfies_refinement(node.refinement.to_tuple()):
                    node = node.left_child
                else:
                    node = node.right_child
            preds.append(node.predicted_label)
            if hasattr(node, 'label_counts'):
                 total = sum(node.label_counts.values())
                 prob1 = node.label_counts.get(1, 0) / total if total > 0 else 0.0
            else:
                 prob1 = float(node.predicted_label)
            probs.append(prob1)
        return preds, probs

    legacy_preds, legacy_probs = predict_legacy(legacy_tree, legacy_test_data)
    try:
        legacy_auc = roc_auc_score(legacy_test_labels, legacy_probs)
    except ValueError:
        legacy_auc = 0.5
    legacy_acc = accuracy_score(legacy_test_labels, legacy_preds)
    logger.info(f"[{name}] Legacy Accuracy: {legacy_acc:.4f}")

    logger.info(f"[{name}] Logic AUC: {logic_auc:.4f} | Legacy AUC: {legacy_auc:.4f}")
    
    return {
        'dataset': name,
        'logic_auc': logic_auc,
        'legacy_auc': legacy_auc,
        'delta': logic_auc - legacy_auc
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify Logic SDT on multiple datasets")
    parser.add_argument(
        "--refinement-mode",
        choices=["dynamic", "static"],
        default="dynamic",
        help="Use dynamic generation or reuse saved refinements (static).",
    )
    parser.add_argument(
        "--refinement-base-dir",
        default=str(Path("output") / "dto_refinements"),
        help=(
            "Base dir containing per-dataset refinement JSON files, e.g. "
            "output/dto_refinements/<dataset>/<target>.json"
        ),
    )
    args = parser.parse_args()

    datasets = [
        {
            'name': 'BACE',
            'key': 'bace',
            'path': 'data/bace/bace.csv',
            'label': 'Class',
            'smiles': 'mol',
        },
        {
            'name': 'ClinTox',
            'key': 'clintox',
            'path': 'data/clintox/clintox.csv',
            'label': 'CT_TOX',
            'smiles': 'smiles',
        },
        {
            'name': 'HIV',
            'key': 'hiv',
            'path': 'data/hiv/HIV.csv',
            'label': 'HIV_active',
            'smiles': 'smiles',
        },
    ]
    
    results = []
    with open("multi_benchmark_results.txt", "w", encoding="utf-8") as f:
        f.write("Dataset,LogicAUC,LegacyAUC,Delta\n")
        
        for ds in datasets:
            refinement_file = ''
            if args.refinement_mode == 'static':
                refinement_file = str(
                    Path(args.refinement_base_dir)
                    / ds['key']
                    / f"{ds['label']}.json"
                )
                if not Path(refinement_file).exists():
                    logger.error(
                        "Missing refinement file for %s/%s: %s",
                        ds['key'],
                        ds['label'],
                        refinement_file,
                    )
                    continue

            res = verify_dataset(
                ds['name'],
                ds['path'],
                ds['label'],
                ds['smiles'],
                refinement_mode=args.refinement_mode,
                refinement_file=refinement_file,
            )
            if res:
                results.append(res)
                line = f"{res['dataset']},{res['logic_auc']:.4f},{res['legacy_auc']:.4f},{res['delta']:.4f}"
                print(line)
                f.write(line + "\n")
