"""
Molecule Ontology: 화학 온톨로지 구조 정의
"""
from typing import List, Dict, Set, Optional
from owlready2 import *
import os


class MoleculeOntology:
    """화학 분자 온톨로지를 관리하는 클래스"""
    
    def __init__(
        self,
        ontology_path: str = "ontology/DTO.xrdf",
        base_dto_path: Optional[str] = None,
    ):
        """
        Args:
            ontology_path: 작업용/저장용 온톨로지 파일 경로.
                - 파일이 이미 존재하면 해당 파일을 로드합니다.
                - 파일이 없으면 base DTO(기본: DTO.owl 또는 DTO.xrdf)를 로드한 뒤
                  확장합니다.
            base_dto_path: 신규 온톨로지 생성 시 기반으로 사용할 DTO 파일 경로(선택).
        """

        # Target ontology path (workspace-specific ontology for the dataset)
        self.ontology_path = ontology_path
        self.base_dto_path = (
            base_dto_path or self._resolve_default_base_dto_path()
        )
        self.onto = None
        self._load_and_enrich_ontology()

    @staticmethod
    def _resolve_default_base_dto_path() -> Optional[str]:
        """Prefer DTO.owl, fall back to DTO.xrdf if present."""
        candidates = [
            os.path.join("ontology", "DTO.owl"),
            os.path.join("ontology", "DTO.xrdf"),
            os.path.join("ontology", "DTO.xml"),
        ]
        for p in candidates:
            if os.path.exists(p):
                return p
        return None
    
    def _load_and_enrich_ontology(self):
        """Load DTO and inject Chemical Ontology structure"""
        
        # 1. Load an ontology
        # - If ontology_path already exists (previously generated dataset ontology), load it.
        # - Otherwise, load base DTO.owl (or DTO.xrdf) and extend it.
        loaded_from = None
        if os.path.exists(self.ontology_path):
            loaded_from = self.ontology_path
        elif self.base_dto_path and os.path.exists(self.base_dto_path):
            loaded_from = self.base_dto_path

        if loaded_from:
            print(f"Loading base ontology from {loaded_from}...")
            try:
                self.onto = get_ontology(loaded_from).load()
            except Exception as e:
                print(
                    f"Failed to load ontology from {loaded_from}: {e}. Creating new."
                )
                self.onto = get_ontology("http://www.semanticweb.org/molecule/ontology")
        else:
            print("No ontology file found. Creating new ontology base.")
            self.onto = get_ontology("http://www.semanticweb.org/molecule/ontology")
            
        # 2. Enrich with Chemical Classes
        with self.onto:
            # Check if classes already exist to avoid duplication if re-loading
            if not self.onto['Molecule']:
                class Molecule(Thing):
                    """Center Class: Molecule (Enriched into DTO)"""
                    pass
            else:
                Molecule = self.onto.Molecule

            # Helper to safely creation
            def get_or_create(name, parent):
                c = self.onto[name]
                if not c:
                    with self.onto:
                        return type(name, (parent,), {})
                return c
            
            self.AromaticMolecule = get_or_create('AromaticMolecule', Molecule)
            self.NonAromaticMolecule = get_or_create('NonAromaticMolecule', Molecule)
            
            self.Substructure = get_or_create('Substructure', Thing)
            self.FunctionalGroup = get_or_create('FunctionalGroup', self.Substructure)
            self.RingSystem = get_or_create('RingSystem', self.Substructure)
            
            # Functionals
            groups = ['Alcohol', 'Amine', 'Carboxyl', 'Carbonyl', 'Ether', 'Ester', 'Amide', 'Nitro', 'Halogen']
            for g in groups:
                 setattr(self, g, get_or_create(g, self.FunctionalGroup))
            
            # Rings
            self.BenzeneRing = get_or_create('BenzeneRing', self.RingSystem)
            self.Heterocycle = get_or_create('Heterocycle', self.RingSystem)

            # Properties
            # We use 'search' or just define. Ideally unique names.
            with self.onto:
                class hasSubstructure(Molecule >> self.Substructure): pass
                class hasFunctionalGroupRel(hasSubstructure): range = [self.FunctionalGroup]
                class hasRingSystem(hasSubstructure): range = [self.RingSystem]
                
                # Data Properties
                if not self.onto['hasMolecularWeight']:
                    class hasMolecularWeight(Molecule >> float): pass
                    class hasNumAtoms(Molecule >> int): pass
                    class hasNumHeavyAtoms(Molecule >> int): pass
                    class hasNumRotatableBonds(Molecule >> int): pass
                    class hasNumHBA(Molecule >> int): pass
                    class hasNumHBD(Molecule >> int): pass
                    class hasNumRings(Molecule >> int): pass
                    class hasNumAromaticRings(Molecule >> int): pass
                    class hasAromaticity(Molecule >> bool): pass
                    class hasLogP(Molecule >> float): pass
                    class hasTPSA(Molecule >> float): pass
                    class obeysLipinski(Molecule >> bool): pass
                    class hasMWCategory(Molecule >> str): pass
                    class hasLogPCategory(Molecule >> str): pass
                    class hasTPSACategory(Molecule >> str): pass
                    class hasLabel(Molecule >> int): pass

        # Expose
        self.Molecule = Molecule
        self.AromaticMolecule = self.AromaticMolecule
        self.NonAromaticMolecule = self.NonAromaticMolecule
        self.Substructure = self.Substructure
        self.FunctionalGroup = self.FunctionalGroup
        self.RingSystem = self.RingSystem
        
        self.hasSubstructure = self.onto.hasSubstructure
        self.hasFunctionalGroupRel = self.onto.hasFunctionalGroupRel
        self.hasRingSystem = self.onto.hasRingSystem
        
        # Expose subclasses
        self.Alcohol = self.onto.Alcohol
        self.Amine = self.onto.Amine
        self.Carboxyl = self.onto.Carboxyl
        self.Carbonyl = self.onto.Carbonyl
        self.Ether = self.onto.Ether
        self.Ester = self.onto.Ester
        self.Amide = self.onto.Amide
        self.Nitro = self.onto.Nitro
        self.Halogen = self.onto.Halogen
        self.BenzeneRing = self.onto.BenzeneRing
        self.Heterocycle = self.onto.Heterocycle
    
    def add_molecule_instance(self, mol_id: str, features: Dict, label: int):
        """분자 인스턴스를 온톨로지에 추가"""
        with self.onto:
            mol_instance = self.Molecule(mol_id)
            
            # Data properties 설정
            mol_instance.hasMolecularWeight = [features['molecular_weight']]
            mol_instance.hasNumAtoms = [features['num_atoms']]
            mol_instance.hasNumHeavyAtoms = [features['num_heavy_atoms']]
            mol_instance.hasNumRotatableBonds = [features['num_rotatable_bonds']]
            mol_instance.hasNumHBA = [features['num_hba']]
            mol_instance.hasNumHBD = [features['num_hbd']]
            mol_instance.hasNumRings = [features['num_rings']]
            mol_instance.hasNumAromaticRings = [features['num_aromatic_rings']]
            mol_instance.hasAromaticity = [features['has_aromatic']]
            mol_instance.hasLogP = [features['logp']]
            mol_instance.hasTPSA = [features['tpsa']]
            mol_instance.obeysLipinski = [features['obeys_lipinski']]
            mol_instance.hasMWCategory = [features['mw_category']]
            mol_instance.hasLogPCategory = [features['logp_category']]
            mol_instance.hasTPSACategory = [features['tpsa_category']]
            
            
            # --- Populate Object Properties for True SDT ---
            # Map string features to Ontology Classes
            fg_map = {
                'Alcohol': self.Alcohol,
                'Amine': self.Amine,
                'Carboxyl': self.Carboxyl,
                'Carbonyl': self.Carbonyl,
                'Ether': self.Ether,
                'Ester': self.Ester,
                'Amide': self.Amide,
                'Nitro': self.Nitro,
                'Halogen': self.Halogen,
                'Benzene': self.BenzeneRing
            }

            for fg_name, fg_class in fg_map.items():
                # If the feature extractor detects this group (assuming features has boolean or count)
                # For now, we rely on the 'functional_groups' list which contains names
                if fg_name in features.get('functional_groups', []):
                    # Create an anonymous instance of the functional group
                    # or a specific one if we want to track unique groups.
                    # For SDT, existence is usually enough, so we create a distinct instance per molecule
                    if fg_class:
                        fg_instance = fg_class()
                        mol_instance.hasFunctionalGroupRel.append(fg_instance)
                    
            # Aromatic Rings
            if features['has_aromatic']:
                mol_instance.is_a.append(self.AromaticMolecule)
            else:
                mol_instance.is_a.append(self.NonAromaticMolecule)

            # Label
            mol_instance.hasLabel = [label]
            
            return mol_instance
    
    def save(self):
        """온톨로지를 파일로 저장"""
        self.onto.save(file=self.ontology_path, format="rdfxml")
        print(f"Ontology saved to {self.ontology_path}")
    
    def load(self):
        """온톨로지를 파일에서 로드"""
        if os.path.exists(self.ontology_path):
            self.onto = get_ontology(self.ontology_path).load()
            print(f"Ontology loaded from {self.ontology_path}")
        else:
            print(f"Ontology file not found: {self.ontology_path}")
