"""Helper functions to access package resources"""

import json
from pathlib import Path
from importlib.resources import files

def get_resource_path():
    """
    Get the path to the resources directory.
    Works both in development and when installed as a package.
    """
    try:
        # Try using importlib.resources (Python 3.9+)
        # This works when the package is installed
        package_path = files('cuban_lib')
        resources_dir = package_path.parent / 'resources'
    except (TypeError, AttributeError):
        # Fallback for development or if importlib.resources doesn't work
        package_dir = Path(__file__).parent.parent
        resources_dir = package_dir / 'resources'
    
    if not resources_dir.exists():
        raise FileNotFoundError(
            f"Resources directory not found at {resources_dir}. "
            "Make sure the package is installed correctly."
        )
    
    return resources_dir

def load_baseline_coverage(technology='ill'):
    """
    Load baseline coverage data for a specific technology.
    
    Args:
        technology (str): Either 'ill' for Illumina or 'pb' for PacBio
        
    Returns:
        dict: Baseline coverage data by chromosome
        
    Example:
        >>> baseline_cov_ill = load_baseline_coverage('ill')
        >>> baseline_cov_pb = load_baseline_coverage('pb')
    """
    if technology not in ['ill', 'pb']:
        raise ValueError("Technology must be 'ill' (Illumina) or 'pb' (PacBio)")
    
    resource_path = get_resource_path()
    filename = f"baseline_cov_{technology}.json"
    filepath = resource_path / filename
    
    if not filepath.exists():
        raise FileNotFoundError(f"Resource file not found: {filepath}")
    
    with open(filepath, 'r') as f:
        return json.load(f)
