"""
SMILES Converter: SMILES 문자열을 분자 특성으로 변환하는 모듈
"""
import json
import os
import sqlite3
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski
from typing import Dict, List, Optional


class MolecularFeatureExtractor:
    """SMILES로부터 의미적 분자 특성을 추출하는 클래스"""
    
    def __init__(self, cache_path: Optional[str] = None):
        """Create a feature extractor.

        Args:
            cache_path: Optional path to a SQLite cache DB. When provided,
                extracted features will be persisted and reused across runs.
        """

        self.cache_path = cache_path
        self._mem_cache: Dict[str, Dict] = {}
        self._db_conn: Optional[sqlite3.Connection] = None
        self._pending_writes: int = 0
        self._commit_every: int = 50

        if self.cache_path:
            self._init_cache_db(self.cache_path)

    def _init_cache_db(self, cache_path: str) -> None:
        Path(os.path.dirname(cache_path) or ".").mkdir(
            parents=True, exist_ok=True
        )
        conn = sqlite3.connect(cache_path)
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS features (
                smiles TEXT PRIMARY KEY,
                features_json TEXT,
                error TEXT
            )
            """
        )
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.commit()
        self._db_conn = conn

    @staticmethod
    def _normalize_smiles_key(smiles: str) -> str:
        """Normalize SMILES key for caching (strip + remove salts)."""
        s = str(smiles).strip()
        if "." in s:
            parts = s.split(".")
            s = max(parts, key=len)
        return s

    def _cache_get(self, key: str) -> Optional[Dict]:
        if key in self._mem_cache:
            return self._mem_cache[key]

        if not self._db_conn:
            return None

        cur = self._db_conn.execute(
            "SELECT features_json, error FROM features WHERE smiles = ?",
            (key,),
        )
        row = cur.fetchone()
        if not row:
            return None

        features_json, error = row
        if error:
            raise ValueError(error)

        if not features_json:
            return None

        features = json.loads(features_json)
        self._mem_cache[key] = features
        return features

    def _cache_set(
        self,
        key: str,
        features: Optional[Dict] = None,
        error: Optional[str] = None,
    ) -> None:
        if features is not None:
            self._mem_cache[key] = features

        if not self._db_conn:
            return

        features_json = (
            json.dumps(features, ensure_ascii=False)
            if features is not None
            else None
        )
        self._db_conn.execute(
            "INSERT OR REPLACE INTO features(smiles, features_json, error) "
            "VALUES (?, ?, ?)",
            (key, features_json, error),
        )
        self._pending_writes += 1
        if self._pending_writes >= self._commit_every:
            self._db_conn.commit()
            self._pending_writes = 0

    def close(self) -> None:
        """Flush pending cache writes and close DB connection."""
        if not self._db_conn:
            return
        try:
            self._db_conn.commit()
        finally:
            self._db_conn.close()
            self._db_conn = None
            self._pending_writes = 0

    def __del__(self):
        try:
            self.close()
        except Exception:
            # Avoid destructor-time exceptions.
            pass
    
    def smiles_to_mol(self, smiles: str) -> Chem.Mol:
        """SMILES 문자열을 RDKit Mol 객체로 변환"""
        # 염(salt) 제거
        smiles = str(smiles).strip()
        if '.' in smiles:
            parts = smiles.split('.')
            # 가장 긴 부분을 주 분자로 선택
            smiles = max(parts, key=len)
        
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles}")
        return mol
    
    def extract_features(self, smiles: str) -> Dict:
        """SMILES로부터 모든 의미적 특성을 추출"""
        key = self._normalize_smiles_key(smiles)
        if self.cache_path:
            cached = self._cache_get(key)
            if cached is not None:
                return cached

        try:
            mol = self.smiles_to_mol(key)
        except Exception as e:
            if self.cache_path:
                self._cache_set(key, error=str(e))
            raise
        
        features = {
            # 기본 속성
            'molecular_weight': Descriptors.MolWt(mol),
            'num_atoms': mol.GetNumAtoms(),
            'num_heavy_atoms': Lipinski.HeavyAtomCount(mol),
            'num_rotatable_bonds': Lipinski.NumRotatableBonds(mol),
            
            # H-bond 특성
            'num_hba': Lipinski.NumHAcceptors(mol),
            'num_hbd': Lipinski.NumHDonors(mol),
            
            # 링 구조
            'num_rings': Lipinski.RingCount(mol),
            'num_aromatic_rings': Lipinski.NumAromaticRings(mol),
            'has_aromatic': Lipinski.NumAromaticRings(mol) > 0,
            
            # LogP (친유성)
            'logp': Descriptors.MolLogP(mol),
            
            # TPSA (Topological Polar Surface Area)
            'tpsa': Descriptors.TPSA(mol),
            
            # Lipinski Rule of Five
            'obeys_lipinski': self._check_lipinski(mol),
            
            # Functional Groups
            'functional_groups': self._detect_functional_groups(mol),
        }
        
        # 카테고리화
        features['mw_category'] = self._categorize_molecular_weight(
            features['molecular_weight']
        )
        features['logp_category'] = self._categorize_logp(features['logp'])
        features['tpsa_category'] = self._categorize_tpsa(features['tpsa'])

        if self.cache_path:
            self._cache_set(key, features=features)

        return features
    
    def _detect_functional_groups(self, mol: Chem.Mol) -> List[str]:
        """분자 내 functional group 검출 (SMARTS 패턴 기반)"""
        detected = []
        
        # SMARTS 패턴 정의
        patterns = {
            'Amine': '[NX3;H2,H1;!$(NC=O)]',
            'Alcohol': '[OX2H]',
            'Carbonyl': '[CX3]=[OX1]',
            'Carboxyl': '[CX3](=O)[OX2H1]',
            'Ether': '[OD2]([#6])[#6]',
            'Ester': '[#6][CX3](=O)[OX2H0][#6]',
            'Amide': '[NX3][CX3](=[OX1])[#6]',
            'Halogen': '[F,Cl,Br,I]',
            'Aromatic': 'c',
            'Sulfur': '[#16]',
            'Nitro': '[$([NX3](=O)=O),$([NX3+](=O)[O-])][!#8]',
        }
        
        for group_name, smarts in patterns.items():
            pattern = Chem.MolFromSmarts(smarts)
            if pattern and mol.HasSubstructMatch(pattern):
                detected.append(group_name)
        
        return detected
    
    def _check_lipinski(self, mol: Chem.Mol) -> bool:
        """Lipinski's Rule of Five 체크"""
        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        hbd = Lipinski.NumHDonors(mol)
        hba = Lipinski.NumHAcceptors(mol)
        
        return (mw <= 500 and logp <= 5 and hbd <= 5 and hba <= 10)
    
    def _categorize_molecular_weight(self, mw: float) -> str:
        """분자량을 카테고리로 분류"""
        if mw < 200:
            return 'Low'
        elif mw < 400:
            return 'Medium'
        else:
            return 'High'
    
    def _categorize_logp(self, logp: float) -> str:
        """LogP를 카테고리로 분류 (친수성/친유성)"""
        if logp < 0:
            return 'Hydrophilic'
        elif logp < 3:
            return 'Moderate'
        else:
            return 'Lipophilic'
    
    def _categorize_tpsa(self, tpsa: float) -> str:
        """TPSA를 카테고리로 분류"""
        if tpsa < 60:
            return 'Low'
        elif tpsa < 140:
            return 'Medium'
        else:
            return 'High'


class MolecularInstance:
    """개별 분자 인스턴스를 나타내는 클래스"""
    
    def __init__(self, mol_id: str, smiles: str, label: int, features: Dict):
        self.mol_id = mol_id
        self.smiles = smiles
        self.label = label
        self.features = features
    
    def __repr__(self):
        return f"MolecularInstance(id={self.mol_id}, label={self.label})"
    
    def satisfies_refinement(self, refinement) -> bool:
        """
        이 분자가 특정 refinement를 만족하는지 확인
        refinement는 (property, operator, value) 형태
        """
        prop, operator, value = refinement
        
        if prop not in self.features:
            return False
        
        feature_value = self.features[prop]
        
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
            # functional_groups와 같은 리스트 속성
            if isinstance(feature_value, list):
                return value in feature_value
            return False
        
        return False
