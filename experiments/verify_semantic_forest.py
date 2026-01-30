# flake8: noqa

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
from src.ontology.molecule_ontology import MoleculeOntology
from src.ontology.smiles_converter import MolecularFeatureExtractor
from src.sdt.logic_forest import SemanticForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def verify_forest():
    logger.info("Starting Semantic Forest Verification on BBBP...")
    
    # Load BBBP
    df = pd.read_csv("data/bbbp/BBBP.csv")
    df_sample = df # Full dataset
    
    # Split
    train_df, test_df = train_test_split(df_sample, test_size=0.2, random_state=42)
    
    # Ontology
    onto_path = "ontology/temp_forest_bbbp.owl"
    if os.path.exists(onto_path): os.remove(onto_path)
    onto = MoleculeOntology(onto_path)
    extractor = MolecularFeatureExtractor()
    
    def populate(dataframe, name):
        instances = []
        for idx, row in dataframe.iterrows():
            try:
                feats = extractor.extract_features(row['smiles'])
                inst = onto.add_molecule_instance(f"Mol_{name}_{idx}", feats, label=int(row['p_np']))
                instances.append(inst)
            except: pass
        return instances
        
    logger.info("Populating Ontology...")
    train_instances = populate(train_df, "Train")
    test_instances = populate(test_df, "Test")
    
    # Train Forest
    logger.info("Training Semantic Forest (5 estimators)...")
    # Using Depth 10 as it was best single tree
    forest = SemanticForest(
        onto,
        n_estimators=5,
        max_depth=10,
        min_samples_split=20,
        min_samples_leaf=5,
        class_weight='balanced',
        verbose=True,
        learner_kwargs={
            # Use C4.5-style gain ratio scoring for splits
            'split_criterion': 'gain_ratio',
        },
    )
    forest.fit(train_instances)
    
    # Evaluate
    logger.info("Evaluating...")
    probs = forest.predict_proba(test_instances)
    preds = forest.predict(test_instances)
    
    test_labels = [inst.hasLabel[0] for inst in test_instances]
    auc = roc_auc_score(test_labels, probs)
    acc = accuracy_score(test_labels, preds)
    
    logger.info(f"Forest Results -> AUC: {auc:.4f}, Accuracy: {acc:.4f}")
    
    with open("forest_results.txt", "w") as f:
        f.write(f"Forest AUC: {auc:.4f}\n")
        f.write(f"Forest Accuracy: {acc:.4f}\n")

if __name__ == "__main__":
    verify_forest()
