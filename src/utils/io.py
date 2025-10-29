"""
File I/O utilities for models and artifacts.
Centralized saving/loading with error handling.
"""

import pickle
import json
import joblib
from pathlib import Path
from typing import Any, Dict, Optional
import logging

logger = logging.getLogger(__name__)


def save_pickle(obj: Any, path: Path) -> None:
    """
    Save object as pickle file.
    
    Args:
        obj: Object to save
        path: Output path
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        with open(path, 'wb') as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info(f"✓ Saved pickle to {path}")
    
    except Exception as e:
        logger.error(f"Failed to save pickle to {path}: {e}")
        raise


def load_pickle(path: Path) -> Any:
    """
    Load object from pickle file.
    
    Args:
        path: Path to pickle file
    
    Returns:
        Loaded object
    """
    if not path.exists():
        raise FileNotFoundError(f"Pickle file not found: {path}")
    
    try:
        with open(path, 'rb') as f:
            obj = pickle.load(f)
        logger.info(f"✓ Loaded pickle from {path}")
        return obj
    
    except Exception as e:
        logger.error(f"Failed to load pickle from {path}: {e}")
        raise


def save_model(model: Any, path: Path, use_joblib: bool = True) -> None:
    """
    Save ML model (using joblib for scikit-learn models).
    
    Args:
        model: Model object to save
        path: Output path
        use_joblib: Use joblib instead of pickle (better for sklearn)
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        if use_joblib:
            joblib.dump(model, path, compress=3)
        else:
            save_pickle(model, path)
        
        logger.info(f"✓ Saved model to {path}")
    
    except Exception as e:
        logger.error(f"Failed to save model to {path}: {e}")
        raise


def load_model(path: Path, use_joblib: bool = True) -> Any:
    """
    Load ML model.
    
    Args:
        path: Path to model file
        use_joblib: Use joblib instead of pickle
    
    Returns:
        Loaded model
    """
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")
    
    try:
        if use_joblib:
            model = joblib.load(path)
        else:
            model = load_pickle(path)
        
        logger.info(f"✓ Loaded model from {path}")
        return model
    
    except Exception as e:
        logger.error(f"Failed to load model from {path}: {e}")
        raise


def save_json(data: Dict, path: Path, indent: int = 2) -> None:
    """
    Save dictionary as JSON file.
    
    Args:
        data: Dictionary to save
        path: Output path
        indent: JSON indentation level
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        with open(path, 'w') as f:
            json.dump(data, f, indent=indent, default=str)
        logger.info(f"✓ Saved JSON to {path}")
    
    except Exception as e:
        logger.error(f"Failed to save JSON to {path}: {e}")
        raise


def load_json(path: Path) -> Dict:
    """
    Load dictionary from JSON file.
    
    Args:
        path: Path to JSON file
    
    Returns:
        Loaded dictionary
    """
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")
    
    try:
        with open(path, 'r') as f:
            data = json.load(f)
        logger.info(f"✓ Loaded JSON from {path}")
        return data
    
    except Exception as e:
        logger.error(f"Failed to load JSON from {path}: {e}")
        raise


def ensure_directory(path: Path) -> None:
    """
    Ensure directory exists, create if needed.
    
    Args:
        path: Directory path
    """
    path.mkdir(parents=True, exist_ok=True)
    logger.debug(f"✓ Directory ensured: {path}")


def get_file_size_mb(path: Path) -> float:
    """
    Get file size in megabytes.
    
    Args:
        path: File path
    
    Returns:
        File size in MB
    """
    if not path.exists():
        return 0.0
    
    size_bytes = path.stat().st_size
    size_mb = size_bytes / (1024 * 1024)
    
    return round(size_mb, 2)


def list_files_by_extension(directory: Path, extension: str) -> list:
    """
    List all files with given extension in directory.
    
    Args:
        directory: Directory to search
        extension: File extension (e.g., '.csv', '.pkl')
    
    Returns:
        List of file paths
    """
    if not directory.exists():
        return []
    
    return sorted(directory.glob(f"*{extension}"))