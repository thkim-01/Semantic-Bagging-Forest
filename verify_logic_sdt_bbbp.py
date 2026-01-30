
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from src.ontology.molecule_ontology import MoleculeOntology
from src.ontology.smiles_converter import MolecularFeatureExtractor
from src.sdt.logic_learner import LogicSDTLearner
from src.sdt.learner import SDTLearner
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def verify_bbbp_logic_sdt(refinement_mode: str, refinement_file: str):
    logger.info("Starting BBBP Logic SDT Verification...")

    # 1. Load Data
    try:
        df = pd.read_csv("data/bbbp/BBBP.csv")
    except FileNotFoundError:
        logger.error("BBBP.csv not found in data/bbbp/")
        return

    # Sample for speed (limitation test)
    # df_sample = df.sample(n=200, random_state=42).reset_index(drop=True)
    df_sample = df # Use full dataset
    logger.info(f"Loaded {len(df_sample)} samples from BBBP.")

    # Split
    train_df, test_df = train_test_split(df_sample, test_size=0.3, random_state=42)
    logger.info(f"Train size: {len(train_df)}, Test size: {len(test_df)}")

    # ---------------------------------------------------------
    # PART A: Logic SDT (Ontology-Driven)
    # ---------------------------------------------------------
    logger.info("\n=== PART A: Logic SDT (Ontology-Driven) ===")
    
    # Initialize Ontology
    onto = MoleculeOntology("bbbp_logic_test.owl")
    extractor = MolecularFeatureExtractor()

    def populate_ontology(dataframe, subset_name):
        instances = []
        valid_indices = []
        for idx, row in dataframe.iterrows():
            try:
                feats = extractor.extract_features(row['smiles'])
                # Add instance
                mol_id = f"Mol_{subset_name}_{idx}"
                inst = onto.add_molecule_instance(mol_id, feats, label=int(row['p_np']))
                instances.append(inst)
                valid_indices.append(idx)
            except Exception as e:
                # logger.warning(f"Failed to process SMILES at index {idx}: {e}")
                pass
        return instances, valid_indices

    logger.info("Populating Ontology with Train Data...")
    train_instances, train_valid_idx = populate_ontology(train_df, "Train")
    logger.info(f"Populated {len(train_instances)} training instances.")

    logger.info("Populating Ontology with Test Data...")
    test_instances, test_valid_idx = populate_ontology(test_df, "Test")
    logger.info(f"Populated {len(test_instances)} testing instances.")

    # Train Logic SDT
    logger.info("Training Logic SDT...")
    # Using strict depth/split to force logic discovery
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

    # Predict Logic SDT
    def predict_logic(tree, instances):
        preds = []
        probs = []
        for inst in instances:
            node = tree.root
            while not node.is_leaf:
                if logic_learner.refinement_generator.instance_satisfies_refinement(inst, node.refinement):
                    node = node.left_child
                else:
                    node = node.right_child
            
            preds.append(node.predicted_label)
            # Probability evaluation
            total = sum(node.label_counts.values())
            prob1 = node.label_counts.get(1, 0) / total if total > 0 else 0.0
            probs.append(prob1)
        return preds, probs

    logic_preds, logic_probs = predict_logic(logic_tree, test_instances)
    
    # Get ground truth labels
    test_labels = [inst.hasLabel[0] for inst in test_instances]
    
    logic_auc = roc_auc_score(test_labels, logic_probs)
    logic_acc = accuracy_score(test_labels, logic_preds)
    logger.info(f"Logic SDT Results -> AUC: {logic_auc:.4f}, Accuracy: {logic_acc:.4f}")
    
    # Print Rules
    logger.info("Logic SDT Tree Rules:")
    traverse(logic_tree.root)


    # ---------------------------------------------------------
    # PART B: Legacy SDT (Feature-Based)
    # ---------------------------------------------------------
    logger.info("\n=== PART B: Legacy SDT (Feature-Based) ===")
    
    # Prepare Feature Matrix for Legacy SDT
    # We need to compute features for the SAME valid instances to be fair
    # Or just use the original DF if indexes match.
    # We'll re-extract features into a DataFrame compatible with legacy SDT
    
    def get_feature_matrix(dataframe, valid_indices):
        # Extract features again or reuse? 
        # Legacy SDT expects a list of dicts or DataFrame works?
        # SDTLearner.fit expects 'instances'. 
        # In legacy code, instances were typically dicts or objects with __dict__?
        # Let's check sdt_learner.py usage. It usually takes list of dicts or objects.
        # We will use list of dicts.
        
        data_list = []
        labels_list = []
        
        subset = dataframe.loc[valid_indices]
        for _, row in subset.iterrows():
            try:
                feats = extractor.extract_features(row['smiles'])
                # Flatten complex types if legacy doesn't handle them?
                # Legacy generic SDT handles numeric/categorical.
                # functional_groups is a list. Legacy SDT might choke or treat as string?
                # We'll convert list to string or indicator columns?
                # For fairness, let's keep numeric features + converted string features.
                
                flat_feats = feats.copy()
                # Keep functional_groups as list for legacy 'contains' check
                
                class InstanceObj:
                    def __init__(self, f_dict, l):
                        self.__dict__.update(f_dict)
                        self.label = l
                        
                    def satisfies_refinement(self, refinement) -> bool:
                        """Check if instance satisfies refinement tuple (prop, op, val)"""
                        prop, operator, value = refinement
                        
                        if not hasattr(self, prop):
                            return False
                        
                        feature_value = getattr(self, prop)
                        
                        if operator == '==':
                            return feature_value == value
                        elif operator == '>':
                            return feature_value > value
                        elif operator == '>=':
                            return feature_value >= value
                        elif operator == '<':
                            return feature_value < value
                        elif operator == '<=':
                            return feature_value <= value
                        elif operator == 'contains':
                            # functional_groups list
                            if isinstance(feature_value, list):
                                return value in feature_value
                            return False
                        return False

                data_list.append(InstanceObj(flat_feats, int(row['p_np'])))
                labels_list.append(int(row['p_np']))
            except:
                pass
        return data_list, labels_list

    legacy_train_data, _ = get_feature_matrix(train_df, train_valid_idx)
    legacy_test_data, legacy_test_labels = get_feature_matrix(test_df, test_valid_idx)
    
    logger.info(f"Prepared {len(legacy_train_data)} legacy training instances.")
    
    # Train Legacy SDT
    logger.info("Training Legacy SDT...")
    legacy_learner = SDTLearner(max_depth=7, min_samples_split=20, min_samples_leaf=10, class_weight='balanced')
    legacy_tree = legacy_learner.fit(legacy_train_data)
    
    # Predict Legacy
    # Legacy SDT assumes predict returns labels? Or we can traverse?
    # SDTLearner returns a tree.
    # We need to implement predict traversal for legacy tree object
    
    def predict_legacy(tree, instances):
        preds = []
        probs = []
        for inst in instances:
            node = tree.root
            while not node.is_leaf:
                # Use satisfies_refinement on instance
                if inst.satisfies_refinement(node.refinement.to_tuple()):
                    node = node.left_child
                else:
                    node = node.right_child
            
            preds.append(node.predicted_label)
            # Prob (assuming legacy tree node has proper counts)
            # Check legacy TreeNode implementation
            if hasattr(node, 'label_counts'):
                 total = sum(node.label_counts.values())
                 prob1 = node.label_counts.get(1, 0) / total if total > 0 else 0.0
            else:
                 prob1 = float(node.predicted_label) # Fallback
            probs.append(prob1)
        return preds, probs

    legacy_preds, legacy_probs = predict_legacy(legacy_tree, legacy_test_data)
    
    legacy_auc = roc_auc_score(legacy_test_labels, legacy_probs)
    legacy_acc = accuracy_score(legacy_test_labels, legacy_preds)
    
    logger.info(f"Legacy SDT Results -> AUC: {legacy_auc:.4f}, Accuracy: {legacy_acc:.4f}")

    logger.info("\n=== COMPARISON ===")
    logger.info(f"Logic SDT AUC: {logic_auc:.4f}")
    logger.info(f"Legacy SDT AUC: {legacy_auc:.4f}")
    logger.info(f"Delta (Logic - Legacy): {logic_auc - legacy_auc:.4f}")
    
    with open("final_results.txt", "w", encoding="utf-8") as f:
        f.write("=== FINAL BBBP BENCHMARK RESULTS ===\n")
        f.write(f"Logic SDT AUC: {logic_auc:.4f}\n")
        f.write(f"Logic SDT Accuracy: {logic_acc:.4f}\n")
        f.write(f"Legacy SDT AUC: {legacy_auc:.4f}\n")
        f.write(f"Legacy SDT Accuracy: {legacy_acc:.4f}\n")
        f.write(f"Delta (Logic - Legacy): {logic_auc - legacy_auc:.4f}\n")

def traverse(node, indent=0):
    prefix = "  " * indent
    if node.is_leaf:
        print(f"{prefix}Leaf: Class {node.predicted_label} (n={node.num_instances})")
    else:
        print(f"{prefix}Node {node.node_id}: {node.refinement} (Gain={node.entropy:.3f})")
        traverse(node.left_child, indent + 1)
        traverse(node.right_child, indent + 1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify Logic SDT on BBBP")
    parser.add_argument(
        "--refinement-mode",
        choices=["dynamic", "static"],
        default="dynamic",
        help="Use dynamic generation or reuse saved refinements (static).",
    )
    parser.add_argument(
        "--refinement-file",
        default=str(Path("output") / "dto_refinements" / "bbbp" / "p_np.json"),
        help="Path to saved refinements JSON (required for static mode).",
    )
    args = parser.parse_args()

    if args.refinement_mode == "static" and not Path(args.refinement_file).exists():
        logger.error(
            "Static mode requested but refinement file not found: %s",
            args.refinement_file,
        )
        raise SystemExit(2)

    verify_bbbp_logic_sdt(args.refinement_mode, args.refinement_file)
