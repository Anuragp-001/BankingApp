"""
Embedding Generator Module
Generates vector embeddings using Euron API.
"""

import numpy as np
import pandas as pd
import requests
from typing import List, Dict, Any, Optional
import hashlib
import time


class EmbeddingGenerator:
    """Generates vector embeddings for banking data using Euron API."""
    
    def __init__(self, api_key: str = "euri-5af13587821689d1c1c8c50ab9fab3e6d04e4800a8489e6bb87b2df2cd408a75"):
        """
        Initialize the embedding generator.
        
        Args:
            api_key: Euron API key
        """
        self.api_key = api_key
        self.api_url = "https://api.euron.one/api/v1/euri/embeddings"
        self.model = "text-embedding-3-small"
        self.embedding_dim = 1536  # text-embedding-3-small dimension
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
    
    def _get_single_embedding(self, text: str) -> np.ndarray:
        """Get embedding for a single text."""
        payload = {
            "input": text,
            "model": self.model
        }
        
        try:
            response = requests.post(self.api_url, headers=self.headers, json=payload, timeout=30)
            response.raise_for_status()
            data = response.json()
            embedding = np.array(data['data'][0]['embedding'])
            return embedding
        except Exception as e:
            print(f"Error getting embedding: {e}")
            # Return zero vector as fallback
            return np.zeros(self.embedding_dim)
    
    def _get_batch_embeddings(self, texts: List[str], batch_size: int = 20) -> List[np.ndarray]:
        """Get embeddings for multiple texts in batches."""
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            # Process batch - Euron API may support batch input
            payload = {
                "input": batch,
                "model": self.model
            }
            
            try:
                response = requests.post(self.api_url, headers=self.headers, json=payload, timeout=60)
                response.raise_for_status()
                data = response.json()
                
                # Extract embeddings from response
                batch_embeddings = [np.array(item['embedding']) for item in data['data']]
                all_embeddings.extend(batch_embeddings)
                
            except Exception as e:
                print(f"Batch embedding error, falling back to single requests: {e}")
                # Fallback to single requests
                for text in batch:
                    embedding = self._get_single_embedding(text)
                    all_embeddings.append(embedding)
                    time.sleep(0.1)  # Rate limiting
            
            # Small delay between batches
            time.sleep(0.2)
        
        return all_embeddings
    
    def generate_embeddings(self, data: pd.DataFrame, text_columns: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Generate embeddings for each row in the DataFrame.
        
        Args:
            data: DataFrame containing the data
            text_columns: Columns to use for generating text representation
            
        Returns:
            List of embedding entries with id, vector, and metadata
        """
        if text_columns is None:
            text_columns = data.columns.tolist()
        
        # Generate text representations for each row
        texts = []
        for idx, row in data.iterrows():
            text = self._row_to_text(row, text_columns)
            texts.append(text)
        
        # Get embeddings in batches
        vectors = self._get_batch_embeddings(texts)
        
        # Create embedding entries
        embeddings = []
        for idx, (row_idx, row) in enumerate(data.iterrows()):
            entry = {
                'id': self._generate_id(row_idx, row),
                'vector': vectors[idx].tolist() if isinstance(vectors[idx], np.ndarray) else vectors[idx],
                'metadata': {
                    'row_index': int(row_idx),
                    'text': texts[idx][:500],  # Truncate for storage
                    **{k: self._serialize_value(v) for k, v in row.to_dict().items()}
                }
            }
            embeddings.append(entry)
        
        return embeddings
    
    def generate_query_embedding(self, query: str) -> np.ndarray:
        """
        Generate embedding for a query string.
        
        Args:
            query: The query text
            
        Returns:
            Numpy array of the embedding
        """
        return self._get_single_embedding(query)
    
    def _row_to_text(self, row: pd.Series, columns: List[str]) -> str:
        """Convert a DataFrame row to a text representation."""
        parts = []
        for col in columns:
            if col in row.index:
                value = row[col]
                if pd.notna(value):
                    if isinstance(value, (int, float)):
                        if 'amount' in col.lower() or 'balance' in col.lower():
                            formatted = f"${value:,.2f}"
                        else:
                            formatted = f"{value:,.2f}" if isinstance(value, float) else str(value)
                    else:
                        formatted = str(value)
                    parts.append(f"{col}: {formatted}")
        
        return " | ".join(parts)
    
    def _generate_id(self, row_idx: int, row: pd.Series) -> str:
        """Generate a unique ID for an embedding entry."""
        content = f"{row_idx}_{row.to_json()}"
        return hashlib.md5(content.encode()).hexdigest()[:16]
    
    def _serialize_value(self, value: Any) -> Any:
        """Serialize a value for JSON storage."""
        if pd.isna(value):
            return None
        if isinstance(value, (np.integer, np.floating)):
            return float(value)
        if isinstance(value, np.ndarray):
            return value.tolist()
        if hasattr(value, 'isoformat'):
            return value.isoformat()
        return str(value)


class EmbeddingEntry:
    """Represents a single embedding entry."""
    
    def __init__(self, id: str, vector: List[float], metadata: Dict[str, Any]):
        self.id = id
        self.vector = np.array(vector)
        self.metadata = metadata
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'vector': self.vector.tolist(),
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EmbeddingEntry':
        return cls(
            id=data['id'],
            vector=data['vector'],
            metadata=data['metadata']
        )
