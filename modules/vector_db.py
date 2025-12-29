"""
Vector Database Module
Stores and retrieves embeddings using Pinecone for similarity search.
"""

import numpy as np
from pinecone import Pinecone, ServerlessSpec
from typing import List, Dict, Any, Optional
import time
import uuid


class VectorDBClient:
    """Interface to Pinecone vector database for similarity search."""
    
    def __init__(self, dimension: int = 1536, 
                 api_key: str = "pcsk_Lwcwo_LzURMduzuFeLZn1yAMgmGZieX59f7imscZTXCHcjoXL6LneKShR1UZxARSGgn1P",
                 index_name: str = "bankdatalens"):
        """
        Initialize the Pinecone vector database.
        
        Args:
            dimension: Dimension of the embedding vectors (1536 for text-embedding-3-small)
            api_key: Pinecone API key
            index_name: Name of the Pinecone index
        """
        self.dimension = dimension
        self.api_key = api_key
        self.index_name = index_name
        self.namespace = f"session_{uuid.uuid4().hex[:8]}"  # Unique namespace per session
        
        # Initialize Pinecone
        self.pc = Pinecone(api_key=self.api_key)
        
        # Initialize index
        self._initialize_index()
        
        # Local metadata cache for faster retrieval
        self.metadata_cache: Dict[str, Dict[str, Any]] = {}
    
    def _initialize_index(self):
        """Initialize or connect to Pinecone index."""
        try:
            # Check if index exists
            existing_indexes = [idx.name for idx in self.pc.list_indexes()]
            
            if self.index_name not in existing_indexes:
                # Create new index
                self.pc.create_index(
                    name=self.index_name,
                    dimension=self.dimension,
                    metric="cosine",
                    spec=ServerlessSpec(
                        cloud="aws",
                        region="us-east-1"
                    )
                )
                # Wait for index to be ready
                time.sleep(5)
            
            # Connect to index
            self.index = self.pc.Index(self.index_name)
            
        except Exception as e:
            print(f"Error initializing Pinecone index: {e}")
            # Fallback to using existing quickstart index
            try:
                self.index = self.pc.Index("quickstart")
                self.index_name = "quickstart"
            except:
                raise Exception(f"Failed to initialize Pinecone: {e}")
    
    def upsert(self, embeddings: List[Dict[str, Any]], batch_size: int = 100) -> int:
        """
        Insert or update embeddings in Pinecone.
        
        Args:
            embeddings: List of embedding entries with 'id', 'vector', and 'metadata'
            batch_size: Number of vectors to upsert per batch
            
        Returns:
            Number of embeddings upserted
        """
        if not embeddings:
            return 0
        
        total_upserted = 0
        
        # Process in batches
        for i in range(0, len(embeddings), batch_size):
            batch = embeddings[i:i + batch_size]
            
            # Prepare vectors for Pinecone
            vectors_to_upsert = []
            for entry in batch:
                # Clean metadata - Pinecone has restrictions on metadata values
                clean_metadata = self._clean_metadata(entry.get('metadata', {}))
                
                vectors_to_upsert.append({
                    'id': entry['id'],
                    'values': entry['vector'],
                    'metadata': clean_metadata
                })
                
                # Cache metadata locally
                self.metadata_cache[entry['id']] = entry.get('metadata', {})
            
            try:
                # Upsert to Pinecone
                self.index.upsert(
                    vectors=vectors_to_upsert,
                    namespace=self.namespace
                )
                total_upserted += len(batch)
                
            except Exception as e:
                print(f"Error upserting batch: {e}")
                continue
            
            # Small delay between batches
            time.sleep(0.1)
        
        return total_upserted
    
    def _clean_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Clean metadata for Pinecone compatibility."""
        clean = {}
        for key, value in metadata.items():
            # Skip None values
            if value is None:
                continue
            
            # Convert to string if needed (Pinecone supports string, number, boolean, list of strings)
            if isinstance(value, (str, int, float, bool)):
                clean[key] = value
            elif isinstance(value, list):
                # Convert list items to strings
                clean[key] = [str(v) for v in value[:10]]  # Limit list length
            else:
                # Convert other types to string
                clean[key] = str(value)[:1000]  # Limit string length
        
        return clean
    
    def query(self, query_vector: np.ndarray, top_k: int = 10, 
              filter_metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Query Pinecone for similar vectors.
        
        Args:
            query_vector: The query embedding
            top_k: Number of results to return
            filter_metadata: Optional metadata filters
            
        Returns:
            List of results with id, score, and metadata
        """
        try:
            # Ensure query vector is a list
            if isinstance(query_vector, np.ndarray):
                query_vector = query_vector.tolist()
            
            # Build filter if provided
            pinecone_filter = None
            if filter_metadata:
                pinecone_filter = {
                    key: {"$eq": value} for key, value in filter_metadata.items()
                }
            
            # Query Pinecone
            results = self.index.query(
                vector=query_vector,
                top_k=top_k,
                namespace=self.namespace,
                include_metadata=True,
                filter=pinecone_filter
            )
            
            # Format results
            formatted_results = []
            for match in results.get('matches', []):
                # Try to get full metadata from cache, fall back to Pinecone metadata
                metadata = self.metadata_cache.get(match['id'], match.get('metadata', {}))
                
                formatted_results.append({
                    'id': match['id'],
                    'score': float(match['score']),
                    'distance': 1 - float(match['score']),  # Convert cosine similarity to distance
                    'metadata': metadata
                })
            
            return formatted_results
            
        except Exception as e:
            print(f"Error querying Pinecone: {e}")
            return []
    
    def delete(self, ids: List[str]) -> int:
        """
        Delete embeddings by ID from Pinecone.
        
        Args:
            ids: List of IDs to delete
            
        Returns:
            Number of entries deleted
        """
        try:
            self.index.delete(ids=ids, namespace=self.namespace)
            
            # Remove from cache
            for id in ids:
                self.metadata_cache.pop(id, None)
            
            return len(ids)
        except Exception as e:
            print(f"Error deleting from Pinecone: {e}")
            return 0
    
    def get_all_metadata(self) -> List[Dict[str, Any]]:
        """Get all cached metadata entries."""
        return [
            {'id': entry_id, 'metadata': metadata}
            for entry_id, metadata in self.metadata_cache.items()
        ]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        try:
            stats = self.index.describe_index_stats()
            namespace_stats = stats.get('namespaces', {}).get(self.namespace, {})
            
            return {
                'total_vectors': namespace_stats.get('vector_count', 0),
                'dimension': self.dimension,
                'metadata_entries': len(self.metadata_cache),
                'index_name': self.index_name,
                'namespace': self.namespace
            }
        except Exception as e:
            return {
                'total_vectors': len(self.metadata_cache),
                'dimension': self.dimension,
                'metadata_entries': len(self.metadata_cache),
                'error': str(e)
            }
    
    def clear(self):
        """Clear all data from the current namespace."""
        try:
            self.index.delete(delete_all=True, namespace=self.namespace)
            self.metadata_cache.clear()
        except Exception as e:
            print(f"Error clearing Pinecone namespace: {e}")
    
    def save(self, path: str):
        """Save metadata cache to disk (Pinecone persists automatically)."""
        import json
        import os
        
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        
        with open(f"{path}.meta", 'w') as f:
            json.dump({
                'metadata_cache': self.metadata_cache,
                'namespace': self.namespace,
                'dimension': self.dimension
            }, f)
    
    def load(self, path: str):
        """Load metadata cache from disk."""
        import json
        
        with open(f"{path}.meta", 'r') as f:
            data = json.load(f)
            self.metadata_cache = data.get('metadata_cache', {})
            self.namespace = data.get('namespace', self.namespace)
            self.dimension = data.get('dimension', self.dimension)
