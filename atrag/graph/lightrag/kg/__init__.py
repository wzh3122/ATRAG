"""
LightRAG Module for ATRAG

This module is based on the original LightRAG project with extensive modifications.

Original Project:
- Repository: https://github.com/HKUDS/LightRAG
- Paper: "LightRAG: Simple and Fast Retrieval-Augmented Generation" (arXiv:2410.05779)
- Authors: Zirui Guo, Lianghao Xia, Yanhua Yu, Tu Ao, Chao Huang
- License: MIT License

Modifications by ATRAG Team:
- Removed global state management for true concurrent processing
- Added stateless interfaces for Celery/Prefect integration
- Implemented instance-level locking mechanism
- Enhanced error handling and stability
- See changelog.md for detailed modifications
"""

# Direct import of storage implementations
from .neo4j_sync_impl import Neo4JSyncStorage
from .pg_ops_sync_graph_storage import PGOpsSyncGraphStorage
from .pg_ops_sync_kv_storage import PGOpsSyncKVStorage
from .pg_ops_sync_vector_storage import PGOpsSyncVectorStorage

# Storage implementation class registry
STORAGES = {
    "Neo4JSyncStorage": Neo4JSyncStorage,
    "PGOpsSyncGraphStorage": PGOpsSyncGraphStorage,
    "PGOpsSyncKVStorage": PGOpsSyncKVStorage,
    "PGOpsSyncVectorStorage": PGOpsSyncVectorStorage,
}


def verify_storage_implementation(storage_type: str, storage_name: str) -> None:
    """Verify if storage implementation is compatible with specified storage type

    Args:
        storage_type: Storage type (KV_STORAGE, GRAPH_STORAGE etc.)
        storage_name: Storage implementation name

    Raises:
        ValueError: If storage implementation is incompatible with storage type or not found
    """
    from ..base import BaseGraphStorage, BaseKVStorage, BaseVectorStorage

    # Check if storage implementation exists in STORAGES
    if storage_name not in STORAGES:
        raise ValueError(f"Storage implementation '{storage_name}' not found in available storages")

    # Define expected base classes for each storage type
    match storage_type:
        case "KV_STORAGE":
            expected_base_class = BaseKVStorage
        case "VECTOR_STORAGE":
            expected_base_class = BaseVectorStorage
        case "GRAPH_STORAGE":
            expected_base_class = BaseGraphStorage
        case _:
            raise ValueError(f"Unknown storage type: {storage_type}")

    # Get the storage class directly from registry
    storage_class = STORAGES[storage_name]

    # Check if the class implements the required base class
    if not issubclass(storage_class, expected_base_class):
        raise ValueError(
            f"Storage implementation '{storage_name}' does not implement the required "
            f"base class '{expected_base_class.__name__}' for storage type '{storage_type}'"
        )
