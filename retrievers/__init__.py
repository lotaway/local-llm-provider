"""Retrievers package with lazy exports to avoid loading heavy optional dependencies."""

from importlib import import_module

_EXPORTS = {
    "HybridRetriever": ("hybrid_retriever", "HybridRetriever"),
    "Reranker": ("reranker", "Reranker"),
    "ExpandedRetriever": ("expanded_retriever", "ExpandedRetriever"),
    "ESBM25Retriever": ("es_retriever", "ESBM25Retriever"),
    "GraphRetriever": ("graph_retriever", "GraphRetriever"),
    "GraphQueryCoordinator": ("graph_coordinator", "GraphQueryCoordinator"),
    "GraphFilteredVectorRetriever": ("chunk_filter", "GraphFilteredVectorRetriever"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute = _EXPORTS[name]
    value = getattr(import_module(f"{__name__}.{module_name}"), attribute)
    globals()[name] = value
    return value
