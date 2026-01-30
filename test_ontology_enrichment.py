
from src.ontology.molecule_ontology import MoleculeOntology

def test_ontology():
    print("Initializing Ontology...")
    onto_manager = MoleculeOntology("test_enriched.owl")
    
    # Mock features
    features = {
        'molecular_weight': 150.0,
        'num_atoms': 20,
        'num_heavy_atoms': 10,
        'num_rotatable_bonds': 2,
        'num_hba': 1,
        'num_hbd': 1,
        'num_rings': 1,
        'num_aromatic_rings': 1,
        'has_aromatic': True,
        'logp': 2.5,
        'tpsa': 40.0,
        'obeys_lipinski': True,
        'mw_category': 'Low',
        'logp_category': 'Moderate',
        'tpsa_category': 'Low',
        'functional_groups': ['Alcohol', 'Benzene']
    }
    
    print("Adding molecule instance...")
    mol = onto_manager.add_molecule_instance("TestMol_1", features, label=1)
    
    print(f"Molecule created: {mol}")
    print(f"Substructures (Object Property - hasSubstructure): {mol.hasSubstructure}")
    print(f"Functional Groups (Object Property - hasFunctionalGroupRel): {mol.hasFunctionalGroupRel}")
    
    # Verify Object Properties
    # Check specific property first
    has_alcohol_rel = any(isinstance(sub, onto_manager.Alcohol) for sub in mol.hasFunctionalGroupRel)
    has_benzene_rel = any(isinstance(sub, onto_manager.BenzeneRing) for sub in mol.hasFunctionalGroupRel)
    
    if has_alcohol_rel and has_benzene_rel:
        print("SUCCESS: Object properties linked correctly via hasFunctionalGroupRel.")
    else:
        print("FAILURE: Object properties not linked via hasFunctionalGroupRel.")
        
    # Check parent property aggregation (Owlready2 might not do this automatically on instances)
    if mol.hasSubstructure:
         print("Parent property hasSubstructure is populated.")
    else:
         print("Parent property hasSubstructure is EMPTY (expected if inference not run).")

if __name__ == "__main__":
    test_ontology()
