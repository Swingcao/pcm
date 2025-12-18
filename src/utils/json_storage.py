"""
JSON-based Local Storage
========================
Replaces ChromaDB with a local JSON-based vector storage system.
All data is stored in the unified results folder.
"""

import os
import json
import numpy as np
from typing import List, Dict, Optional, Any, Tuple
from datetime import datetime
from dataclasses import dataclass, asdict


@dataclass
class VectorRecord:
    """A single vector record with embedding and metadata."""
    id: str
    embedding: List[float]
    content: str
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'VectorRecord':
        return cls(
            id=data['id'],
            embedding=data['embedding'],
            content=data['content'],
            metadata=data['metadata']
        )


class JSONVectorStore:
    """
    JSON-based vector storage that replaces ChromaDB.

    Features:
    - Stores embeddings and metadata in JSON files
    - Supports cosine similarity search
    - Automatic persistence to disk
    - No external database dependencies
    """

    def __init__(self, storage_path: str, collection_name: str = "default"):
        """
        Initialize the JSON vector store.

        Args:
            storage_path: Base directory for storage
            collection_name: Name of the collection (subdirectory)
        """
        self.storage_path = storage_path
        self.collection_name = collection_name
        self.collection_path = os.path.join(storage_path, f"{collection_name}.json")

        # In-memory index
        self._records: Dict[str, VectorRecord] = {}
        self._embeddings_matrix: Optional[np.ndarray] = None
        self._id_list: List[str] = []

        # Ensure directory exists
        os.makedirs(storage_path, exist_ok=True)

        # Load existing data
        self._load()

    def _load(self) -> None:
        """Load records from JSON file."""
        if os.path.exists(self.collection_path):
            try:
                with open(self.collection_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for record_data in data.get('records', []):
                        record = VectorRecord.from_dict(record_data)
                        self._records[record.id] = record
                self._rebuild_index()
                print(f"Loaded {len(self._records)} records from {self.collection_path}")
            except Exception as e:
                print(f"Failed to load vector store: {e}")
                self._records = {}

    def _save(self) -> None:
        """Save records to JSON file."""
        data = {
            'collection_name': self.collection_name,
            'record_count': len(self._records),
            'updated_at': datetime.now().isoformat(),
            'records': [r.to_dict() for r in self._records.values()]
        }
        with open(self.collection_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _rebuild_index(self) -> None:
        """Rebuild the numpy matrix index for fast search."""
        if not self._records:
            self._embeddings_matrix = None
            self._id_list = []
            return

        self._id_list = list(self._records.keys())
        embeddings = [self._records[id].embedding for id in self._id_list]
        self._embeddings_matrix = np.array(embeddings)

    def upsert(
        self,
        ids: List[str],
        embeddings: List[List[float]],
        metadatas: List[Dict[str, Any]],
        documents: List[str]
    ) -> None:
        """
        Insert or update records.

        Args:
            ids: List of record IDs
            embeddings: List of embedding vectors
            metadatas: List of metadata dictionaries
            documents: List of document contents
        """
        for id, emb, meta, doc in zip(ids, embeddings, metadatas, documents):
            self._records[id] = VectorRecord(
                id=id,
                embedding=emb,
                content=doc,
                metadata=meta
            )

        self._rebuild_index()
        self._save()

    def update(
        self,
        ids: List[str],
        metadatas: List[Dict[str, Any]]
    ) -> None:
        """Update metadata for existing records."""
        for id, meta in zip(ids, metadatas):
            if id in self._records:
                self._records[id].metadata = meta
        self._save()

    def get(self, ids: List[str]) -> Dict[str, Any]:
        """Get records by IDs."""
        result = {
            'ids': [],
            'embeddings': [],
            'metadatas': [],
            'documents': []
        }

        for id in ids:
            if id in self._records:
                record = self._records[id]
                result['ids'].append(record.id)
                result['embeddings'].append(record.embedding)
                result['metadatas'].append(record.metadata)
                result['documents'].append(record.content)

        return result

    def query(
        self,
        query_embeddings: List[List[float]],
        n_results: int = 10,
        include: List[str] = None
    ) -> Dict[str, Any]:
        """
        Query similar vectors using cosine similarity.

        Args:
            query_embeddings: Query vectors
            n_results: Number of results to return
            include: Fields to include in results

        Returns:
            Dict with ids, distances, metadatas, documents
        """
        if self._embeddings_matrix is None or len(self._id_list) == 0:
            return {
                'ids': [[]],
                'distances': [[]],
                'metadatas': [[]],
                'documents': [[]]
            }

        query_vec = np.array(query_embeddings[0])

        # Normalize for cosine similarity
        query_norm = query_vec / (np.linalg.norm(query_vec) + 1e-10)
        matrix_norms = self._embeddings_matrix / (
            np.linalg.norm(self._embeddings_matrix, axis=1, keepdims=True) + 1e-10
        )

        # Compute cosine similarities
        similarities = np.dot(matrix_norms, query_norm)

        # Convert to distances (1 - similarity for cosine)
        distances = 1 - similarities

        # Get top-k indices
        n_results = min(n_results, len(self._id_list))
        top_indices = np.argsort(distances)[:n_results]

        # Build results
        result_ids = []
        result_distances = []
        result_metadatas = []
        result_documents = []

        for idx in top_indices:
            record_id = self._id_list[idx]
            record = self._records[record_id]
            result_ids.append(record_id)
            result_distances.append(float(distances[idx]))
            result_metadatas.append(record.metadata)
            result_documents.append(record.content)

        return {
            'ids': [result_ids],
            'distances': [result_distances],
            'metadatas': [result_metadatas],
            'documents': [result_documents]
        }

    def count(self) -> int:
        """Return the number of records."""
        return len(self._records)

    def delete(self, ids: List[str]) -> None:
        """Delete records by IDs."""
        for id in ids:
            if id in self._records:
                del self._records[id]
        self._rebuild_index()
        self._save()

    def clear(self) -> None:
        """Clear all records."""
        self._records = {}
        self._embeddings_matrix = None
        self._id_list = []
        self._save()


class ResultsManager:
    """
    Unified results folder manager.

    Organizes all output files into categorized subfolders:
    - results/
      - knowledge_graphs/     # Graph storage (JSON)
      - vector_stores/        # Vector embeddings (JSON)
      - experiments/          # Experiment results
      - intermediate/         # Intermediate processing files
      - logs/                 # Processing logs
    """

    def __init__(self, base_path: str = "./results"):
        """
        Initialize the results manager.

        Args:
            base_path: Base directory for all results
        """
        self.base_path = base_path

        # Define folder structure
        self.folders = {
            'knowledge_graphs': os.path.join(base_path, 'knowledge_graphs'),
            'vector_stores': os.path.join(base_path, 'vector_stores'),
            'experiments': os.path.join(base_path, 'experiments'),
            'intermediate': os.path.join(base_path, 'intermediate'),
            'logs': os.path.join(base_path, 'logs')
        }

        # Create all folders
        self._create_folders()

    def _create_folders(self) -> None:
        """Create all required folders."""
        for folder in self.folders.values():
            os.makedirs(folder, exist_ok=True)

    def get_path(self, category: str, filename: str = None) -> str:
        """
        Get the path for a specific category.

        Args:
            category: One of 'knowledge_graphs', 'vector_stores', 'experiments',
                     'intermediate', 'logs'
            filename: Optional filename to append

        Returns:
            Full path
        """
        if category not in self.folders:
            raise ValueError(f"Unknown category: {category}. Available: {list(self.folders.keys())}")

        path = self.folders[category]
        if filename:
            path = os.path.join(path, filename)
        return path

    def save_json(self, category: str, filename: str, data: Any) -> str:
        """
        Save data as JSON to the specified category.

        Args:
            category: Folder category
            filename: Filename (will add .json if not present)
            data: Data to save

        Returns:
            Full path where file was saved
        """
        if not filename.endswith('.json'):
            filename += '.json'

        filepath = self.get_path(category, filename)

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=str)

        return filepath

    def load_json(self, category: str, filename: str) -> Any:
        """
        Load JSON data from the specified category.

        Args:
            category: Folder category
            filename: Filename

        Returns:
            Loaded data
        """
        if not filename.endswith('.json'):
            filename += '.json'

        filepath = self.get_path(category, filename)

        if not os.path.exists(filepath):
            return None

        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)

    def save_intermediate(
        self,
        name: str,
        data: Any,
        step: Optional[str] = None
    ) -> str:
        """
        Save intermediate processing data.

        Args:
            name: Base name for the file
            data: Data to save
            step: Optional step identifier

        Returns:
            Full path where file was saved
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if step:
            filename = f"{name}_{step}_{timestamp}.json"
        else:
            filename = f"{name}_{timestamp}.json"

        return self.save_json('intermediate', filename, data)

    def save_experiment_result(
        self,
        experiment_name: str,
        results: Dict[str, Any]
    ) -> str:
        """
        Save experiment results.

        Args:
            experiment_name: Name of the experiment
            results: Results dictionary

        Returns:
            Full path where file was saved
        """
        # Create experiment subfolder
        exp_folder = os.path.join(self.folders['experiments'], experiment_name)
        os.makedirs(exp_folder, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"results_{timestamp}.json"
        filepath = os.path.join(exp_folder, filename)

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2, default=str)

        return filepath

    def list_files(self, category: str) -> List[str]:
        """
        List all files in a category.

        Args:
            category: Folder category

        Returns:
            List of filenames
        """
        folder = self.folders.get(category)
        if not folder or not os.path.exists(folder):
            return []

        return [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]

    def get_statistics(self) -> Dict[str, Any]:
        """Get storage statistics."""
        stats = {}
        for name, folder in self.folders.items():
            if os.path.exists(folder):
                files = os.listdir(folder)
                total_size = sum(
                    os.path.getsize(os.path.join(folder, f))
                    for f in files if os.path.isfile(os.path.join(folder, f))
                )
                stats[name] = {
                    'file_count': len(files),
                    'total_size_kb': round(total_size / 1024, 2)
                }
            else:
                stats[name] = {'file_count': 0, 'total_size_kb': 0}
        return stats
