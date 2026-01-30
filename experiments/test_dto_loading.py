
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.ontology.molecule_ontology import MoleculeOntology

def test_loading():
    print("Testing MoleculeOntology with DTO...")
    if not os.path.exists("ontology/DTO.xrdf"):
        print("DTO.xrdf not found!")
        return

    try:
        # Initialize
        onto_manager = MoleculeOntology("ontology/DTO.xrdf")
        print("Initialization complete.")
        
        # Check Enrichment
        print(f"Molecule Class: {onto_manager.Molecule}")
        print(f"AromaticMolecule Class: {onto_manager.AromaticMolecule}")
        print(f"BenzeneRing Class: {onto_manager.BenzeneRing}")
        
        # Check if they are actually in the ontology
        print(f"Classes in ontology: {len(list(onto_manager.onto.classes()))}")
        
        # Test adding an instance
        print("Adding dummy instance...")
        feats = {
            'molecular_weight': 100.0,
            'num_atoms': 10,
            'num_heavy_atoms': 5,
            'num_rotatable_bonds': 2,
            'num_hba': 1,
            'num_hbd': 1,
            'num_rings': 1,
            'num_aromatic_rings': 1,
            'has_aromatic': True,
            'logp': 1.5,
            'tpsa': 20.0,
            'obeys_lipinski': True,
            'mw_category': 'Low',
            'logp_category': 'Low',
            'tpsa_category': 'Low',
            'functional_groups': ['Benzene', 'Alcohol']
        }
        inst = onto_manager.add_molecule_instance("TestMol_1", feats, label=1)
        print(f"Instance created: {inst}")
        print(f"Instance is_a: {inst.is_a}")
        print(f"Instance functional groups: {inst.hasFunctionalGroupRel}")
        
        print("DTO Loading Test Success!")
        
    except Exception as e:
        print(f"Test Failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_loading()
