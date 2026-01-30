
from owlready2 import *
import os

def check_dto():
    path = "ontology/DTO.xrdf"
    if not os.path.exists(path):
        print(f"File not found: {path}")
        return

    print(f"Loading {path}...")
    try:
        onto = get_ontology(path).load()
        print("Load successful.")
        
        print(f"Classes: {len(list(onto.classes()))}")
        print(f"Properties: {len(list(onto.properties()))}")
        
        # Check for specific chemical terms
        terms = ["Molecule", "Chemical", "Structure", "Aromatic", "Alcohol", "Benzene"]
        found = []
        for c in onto.classes():
            for t in terms:
                if t.lower() in c.name.lower():
                    found.append(c.name)
        
        print("\nPotentially relevant classes found:")
        for f in sorted(list(set(found)))[:20]:
            print(f"- {f}")
            
    except Exception as e:
        print(f"Error loading ontology: {e}")

if __name__ == "__main__":
    check_dto()
