import json
import math
from typing import Dict, Any, Mapping

class GenomeValidationError(Exception):
    """Exception raised when genome validation fails."""
    pass

class GenomeConfigurationError(Exception):
    """Exception raised when genome engine configuration is missing or invalid."""
    pass

def validate_genome(genome: Dict[str, float], config: Dict[str, Any] | None = None) -> None:
    """
    Validates a genome dictionary.
    
    Checks:
    - Input is a mapping/dictionary
    - All keys are strings
    - All values are numeric (integers or floats; rejects booleans)
    - All values are not NaN or infinite
    - Values are within configured bounds if config is provided
    
    Raises GenomeValidationError if validation fails.
    """
    if not isinstance(genome, Mapping):
        raise GenomeValidationError("Genome must be a mapping/dictionary")
        
    for key, val in genome.items():
        if not isinstance(key, str):
            raise GenomeValidationError("Genome keys must be strings")
            
        # Reject booleans explicitly (since isinstance(True, (int, float)) is True)
        if isinstance(val, bool):
            raise GenomeValidationError(f"Genome value for key '{key}' must be numeric, not boolean")
            
        if not isinstance(val, (int, float)):
            raise GenomeValidationError(f"Genome value for key '{key}' must be numeric (int or float)")
            
        if math.isnan(val) or math.isinf(val):
            raise GenomeValidationError(f"Genome value for key '{key}' cannot be NaN or infinite")
            
        if config and key in config:
            field_config = config[key]
            if isinstance(field_config, dict):
                min_val = field_config.get("min")
                max_val = field_config.get("max")
                if min_val is not None:
                    if val < min_val:
                        raise GenomeValidationError(f"Genome value '{key}' ({val}) is below configured min ({min_val})")
                if max_val is not None:
                    if val > max_val:
                        raise GenomeValidationError(f"Genome value '{key}' ({val}) is above configured max ({max_val})")

def normalize_genome(genome: Dict[str, float], config: Dict[str, Any], preserve_unconfigured: bool = True) -> Dict[str, float]:
    """
    Normalizes the genome based on range bounds in configuration.
    
    Normalizes to [0.0, 1.0].
    If field is not in config and preserve_unconfigured is True, returns original value.
    If field is not in config and preserve_unconfigured is False, raises GenomeConfigurationError.
    
    Raises GenomeValidationError if genome is invalid.
    Raises GenomeConfigurationError if bounds are invalid or missing when required.
    """
    validate_genome(genome, None)
    
    normalized_genome = {}
    for key, val in genome.items():
        if key in config:
            field_config = config[key]
            if not isinstance(field_config, dict):
                raise GenomeConfigurationError(f"Configuration for '{key}' must be a dictionary")
                
            min_val = field_config.get("min")
            max_val = field_config.get("max")
            
            if min_val is None or max_val is None:
                raise GenomeConfigurationError(f"Configuration for '{key}' must contain both 'min' and 'max' bounds")
                
            if not isinstance(min_val, (int, float)) or not isinstance(max_val, (int, float)):
                raise GenomeConfigurationError(f"Bounds for '{key}' must be numeric")
                
            if max_val < min_val:
                raise GenomeConfigurationError(f"Max bound ({max_val}) cannot be less than min bound ({min_val}) for '{key}'")
                
            if max_val == min_val:
                normalized = 0.0
            else:
                raw_norm = (val - min_val) / (max_val - min_val)
                # Clip normalized value to [0.0, 1.0]
                normalized = min(max(raw_norm, 0.0), 1.0)
                
            normalized_genome[key] = normalized
        else:
            if preserve_unconfigured:
                normalized_genome[key] = val
            else:
                raise GenomeConfigurationError(f"No range configuration provided for field '{key}'")
                
    return normalized_genome

def calculate_dimension_intensity(field: str, val: float, config: Dict[str, Any]) -> float:
    """
    Calculates the intensity of a single genome field based on config bounds and direction.
    
    Raises GenomeConfigurationError if configuration is missing, incomplete, or direction is not specified.
    """
    if field not in config:
        raise GenomeConfigurationError(f"No configuration provided for field '{field}'")
        
    field_config = config[field]
    if not isinstance(field_config, dict):
        raise GenomeConfigurationError(f"Configuration for '{field}' must be a dictionary")
        
    min_val = field_config.get("min")
    max_val = field_config.get("max")
    direction = field_config.get("suspicious_direction")
    
    if min_val is None or max_val is None:
        raise GenomeConfigurationError(f"Configuration for '{field}' must contain both 'min' and 'max' bounds")
        
    if not isinstance(min_val, (int, float)) or not isinstance(max_val, (int, float)):
        raise GenomeConfigurationError(f"Bounds for '{field}' must be numeric")
        
    if max_val < min_val:
        raise GenomeConfigurationError(f"Max bound ({max_val}) cannot be less than min bound ({min_val}) for '{field}'")
        
    if direction not in ("higher", "lower"):
        raise GenomeConfigurationError(
            f"Field '{field}' must specify 'suspicious_direction' as 'higher' or 'lower' (got {repr(direction)})"
        )
        
    if max_val == min_val:
        norm = 0.0
    else:
        raw_norm = (val - min_val) / (max_val - min_val)
        norm = min(max(raw_norm, 0.0), 1.0)
        
    if direction == "higher":
        return norm
    else:
        return 1.0 - norm

def calculate_aggregate_intensity(genome: Dict[str, float], config: Dict[str, Any]) -> float:
    """
    Calculates aggregate suspicious intensity as average of dimension intensities.
    
    Skips unconfigured genome fields.
    Raises GenomeValidationError if genome validation fails.
    Raises GenomeConfigurationError if no configured fields are present or calculation fails.
    """
    validate_genome(genome, None)
    
    intensities = []
    for key, val in genome.items():
        if key in config:
            intensity = calculate_dimension_intensity(key, val, config)
            intensities.append(intensity)
            
    if not intensities:
        raise GenomeConfigurationError("No configured dimensions found in genome to calculate aggregate intensity")
        
    return sum(intensities) / len(intensities)

def compare_genomes(genome_a: Dict[str, float], genome_b: Dict[str, float]) -> Dict[str, Any]:
    """
    Compares two genomes and returns fields added, removed, or changed.
    
    Both inputs are validated first.
    Returns:
    {
        "added": {field: new_val},
        "removed": {field: old_val},
        "changed": {field: {"old": old_val, "new": new_val, "delta": new_val - old_val}}
    }
    """
    validate_genome(genome_a)
    validate_genome(genome_b)
    
    keys_a = set(genome_a.keys())
    keys_b = set(genome_b.keys())
    
    added = {k: genome_b[k] for k in sorted(keys_b - keys_a)}
    removed = {k: genome_a[k] for k in sorted(keys_a - keys_b)}
    
    changed = {}
    for k in sorted(keys_a & keys_b):
        if genome_a[k] != genome_b[k]:
            changed[k] = {
                "old": genome_a[k],
                "new": genome_b[k],
                "delta": genome_b[k] - genome_a[k]
            }
            
    return {
        "added": added,
        "removed": removed,
        "changed": changed
    }

def serialize_genome(genome: Dict[str, float]) -> str:
    """
    Dumps genome to a deterministic JSON string with sorted keys.
    """
    validate_genome(genome)
    return json.dumps(genome, sort_keys=True)

def deserialize_genome(serialized: str) -> Dict[str, float]:
    """
    Loads JSON string back to a dictionary and validates it.
    """
    try:
        genome = json.loads(serialized)
    except json.JSONDecodeError as e:
        raise GenomeValidationError(f"Invalid JSON string: {e}")
        
    validate_genome(genome)
    return genome
