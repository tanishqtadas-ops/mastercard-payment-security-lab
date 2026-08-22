import math
import json
import pytest
from mutation.genome_engine import (
    validate_genome,
    normalize_genome,
    calculate_dimension_intensity,
    calculate_aggregate_intensity,
    compare_genomes,
    serialize_genome,
    deserialize_genome,
    GenomeValidationError,
    GenomeConfigurationError,
)

# Test 1: Valid genome validation
def test_valid_genome_validation():
    genome = {
        "amount_deviation": 0.5,
        "velocity_deviation": 2,
    }
    config = {
        "amount_deviation": {"min": 0.0, "max": 1.0},
        "velocity_deviation": {"min": 0, "max": 10},
    }
    # Should not raise any error
    validate_genome(genome, config)

# Test 2: Invalid non-mapping input
def test_invalid_non_mapping_input():
    for invalid_input in (None, "not a dict", [1, 2, 3], 4.2):
        with pytest.raises(GenomeValidationError, match="must be a mapping/dictionary"):
            validate_genome(invalid_input)

# Test 3: Invalid numeric values (including booleans)
def test_invalid_numeric_values():
    # String value
    with pytest.raises(GenomeValidationError, match="must be numeric"):
        validate_genome({"amount_deviation": "high"})
        
    # List value
    with pytest.raises(GenomeValidationError, match="must be numeric"):
        validate_genome({"amount_deviation": [0.5]})
        
    # Boolean value (should be rejected specifically as bool is subclass of int)
    with pytest.raises(GenomeValidationError, match="must be numeric, not boolean"):
        validate_genome({"amount_deviation": True})
        
    with pytest.raises(GenomeValidationError, match="must be numeric, not boolean"):
        validate_genome({"amount_deviation": False})

# Test 4: NaN rejection
def test_nan_rejection():
    with pytest.raises(GenomeValidationError, match="cannot be NaN or infinite"):
        validate_genome({"amount_deviation": float("nan")})

# Test 5: Infinity rejection
def test_infinity_rejection():
    with pytest.raises(GenomeValidationError, match="cannot be NaN or infinite"):
        validate_genome({"amount_deviation": float("inf")})
    with pytest.raises(GenomeValidationError, match="cannot be NaN or infinite"):
        validate_genome({"amount_deviation": float("-inf")})

# Test 6: Valid normalization
def test_valid_normalization():
    genome = {
        "amount_deviation": 0.3,
        "velocity_deviation": 5.0,
        "location_deviation": -1.0,  # below min
        "time_deviation": 12.0,       # above max
    }
    config = {
        "amount_deviation": {"min": 0.0, "max": 1.0},
        "velocity_deviation": {"min": 0.0, "max": 10.0},
        "location_deviation": {"min": 0.0, "max": 10.0},
        "time_deviation": {"min": 0.0, "max": 10.0},
    }
    normalized = normalize_genome(genome, config)
    assert normalized["amount_deviation"] == pytest.approx(0.3)
    assert normalized["velocity_deviation"] == pytest.approx(0.5)
    # verify clipping
    assert normalized["location_deviation"] == 0.0
    assert normalized["time_deviation"] == 1.0

# Test 7: Missing-range handling
def test_missing_range_handling():
    genome = {
        "amount_deviation": 0.5,
        "unconfigured_field": 10.0,
    }
    config = {
        "amount_deviation": {"min": 0.0, "max": 1.0},
    }
    
    # 1. With preserve_unconfigured=True (default), unconfigured field is preserved
    normalized = normalize_genome(genome, config, preserve_unconfigured=True)
    assert normalized["unconfigured_field"] == 10.0
    
    # 2. With preserve_unconfigured=False, raises GenomeConfigurationError
    with pytest.raises(GenomeConfigurationError, match="No range configuration provided"):
        normalize_genome(genome, config, preserve_unconfigured=False)

# Test 8: Intensity calculation (with directionality support)
def test_intensity_calculation():
    config = {
        "amount_deviation": {"min": 0.0, "max": 10.0, "suspicious_direction": "higher"},
        "profile_plausibility": {"min": 0.0, "max": 100.0, "suspicious_direction": "lower"},
        "no_direction": {"min": 0.0, "max": 1.0},  # missing direction
    }
    
    # 1. higher direction: intensity = normalized_val
    # val = 3.0 -> norm = 0.3 -> intensity = 0.3
    assert calculate_dimension_intensity("amount_deviation", 3.0, config) == pytest.approx(0.3)
    # val = 12.0 -> norm = 1.0 (clipped) -> intensity = 1.0
    assert calculate_dimension_intensity("amount_deviation", 12.0, config) == pytest.approx(1.0)
    
    # 2. lower direction: intensity = 1.0 - normalized_val
    # val = 80.0 -> norm = 0.8 -> intensity = 0.2
    assert calculate_dimension_intensity("profile_plausibility", 80.0, config) == pytest.approx(0.2)
    # val = -10.0 -> norm = 0.0 (clipped) -> intensity = 1.0
    assert calculate_dimension_intensity("profile_plausibility", -10.0, config) == pytest.approx(1.0)
    
    # 3. Missing direction: must raise GenomeConfigurationError
    with pytest.raises(GenomeConfigurationError, match="must specify 'suspicious_direction' as 'higher' or 'lower'"):
        calculate_dimension_intensity("no_direction", 0.5, config)
        
    # 4. Aggregate intensity calculation
    genome = {
        "amount_deviation": 3.0,       # intensity = 0.3
        "profile_plausibility": 80.0,  # intensity = 0.2
        "ignored_unconfigured": 99.9,
    }
    # Average of 0.3 and 0.2 is 0.25 (since aggregate uses dimensions present in genome & config)
    # field "amount_deviation" intensity = 0.3
    # field "profile_plausibility" intensity = 0.8 (since suspicious_direction is "lower", val=80.0 -> norm=0.8 -> intensity = 1.0 - 0.8 = 0.2)
    # Wait, let's verify aggregate:
    # aggregate intensity = (0.3 + 0.2) / 2 = 0.25
    agg = calculate_aggregate_intensity(genome, config)
    assert agg == pytest.approx(0.25)
    
    # 5. Aggregate fails when no configured fields exist in the genome
    with pytest.raises(GenomeConfigurationError, match="No configured dimensions found"):
        calculate_aggregate_intensity({"ignored_unconfigured": 99.9}, config)

# Test 9: Genome comparison
def test_genome_comparison():
    genome_a = {"field_1": 0.5, "field_2": 1.0}
    genome_b = {"field_1": 0.7, "field_2": 1.0}
    
    res = compare_genomes(genome_a, genome_b)
    assert "changed" in res
    assert "field_1" in res["changed"]
    assert res["changed"]["field_1"]["old"] == 0.5
    assert res["changed"]["field_1"]["new"] == 0.7
    assert res["changed"]["field_1"]["delta"] == pytest.approx(0.2)
    assert "field_2" not in res["changed"]

# Test 10: Added field detection
def test_added_field_detection():
    genome_a = {"field_1": 0.5}
    genome_b = {"field_1": 0.5, "field_2": 0.8}
    
    res = compare_genomes(genome_a, genome_b)
    assert res["added"] == {"field_2": 0.8}
    assert res["removed"] == {}

# Test 11: Removed field detection
def test_removed_field_detection():
    genome_a = {"field_1": 0.5, "field_2": 0.8}
    genome_b = {"field_1": 0.5}
    
    res = compare_genomes(genome_a, genome_b)
    assert res["removed"] == {"field_2": 0.8}
    assert res["added"] == {}

# Test 12: Changed value detection
def test_changed_value_detection():
    genome_a = {"field_1": 10.0}
    genome_b = {"field_1": 7.5}
    
    res = compare_genomes(genome_a, genome_b)
    assert "field_1" in res["changed"]
    assert res["changed"]["field_1"]["old"] == 10.0
    assert res["changed"]["field_1"]["new"] == 7.5
    assert res["changed"]["field_1"]["delta"] == pytest.approx(-2.5)

# Test 13: Serialization round-trip
def test_serialization_round_trip():
    genome = {
        "velocity_deviation": 4.2,
        "amount_deviation": 0.1,
    }
    
    # 1. Deterministic JSON serialization (keys sorted alphabetically)
    serialized = serialize_genome(genome)
    expected_json = '{"amount_deviation": 0.1, "velocity_deviation": 4.2}'
    assert serialized == expected_json
    
    # 2. Deserialization round-trip
    deserialized = deserialize_genome(serialized)
    assert deserialized == genome
    
    # 3. Invalid JSON raises validation error
    with pytest.raises(GenomeValidationError, match="Invalid JSON string"):
        deserialize_genome("invalid-json")

# Test 14: Deterministic repeated execution
def test_deterministic_repeated_execution():
    genome = {
        "amount_deviation": 5.0,
        "profile_plausibility": 20.0,
    }
    config = {
        "amount_deviation": {"min": 0.0, "max": 10.0, "suspicious_direction": "higher"},
        "profile_plausibility": {"min": 0.0, "max": 100.0, "suspicious_direction": "lower"},
    }
    
    # Repeated calls must return identical results
    for _ in range(50):
        # Validation
        validate_genome(genome, config)
        
        # Normalization
        norm = normalize_genome(genome, config)
        assert norm == {"amount_deviation": 0.5, "profile_plausibility": 0.2}
        
        # Intensity
        agg = calculate_aggregate_intensity(genome, config)
        assert agg == pytest.approx(0.65)  # 0.5 + (1.0 - 0.2) = 0.5 + 0.8 = 1.3 / 2 = 0.65
        
        # Serialization
        serialized = serialize_genome(genome)
        assert serialized == '{"amount_deviation": 5.0, "profile_plausibility": 20.0}'
