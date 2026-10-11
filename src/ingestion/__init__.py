"""
Data Ingestion Subsystem for News and Social Media Feeds.
"""
from .data_loader import MultiSourceIngestionEngine, RawItem

__all__ = ["MultiSourceIngestionEngine", "RawItem"]
