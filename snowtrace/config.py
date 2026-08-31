"""
SnowTrace Configuration Module
─────────────────────────────
Central configuration for SAR processing parameters, threat scoring
weights, geospatial thresholds, and system-wide constants.
"""

from dataclasses import dataclass, field
from typing import Tuple


@dataclass(frozen=True)
class SARConfig:
    """Sentinel-1 SAR acquisition and preprocessing parameters."""

    # Polarization channels to process
    polarization_channels: Tuple[str, str] = ("VV", "VH")

    # Speckle filter: Lee filter kernel size (odd integer)
    speckle_filter_size: int = 7

    # Radiometric calibration: sigma0 output calibration
    calibration_target: str = "sigma0"

    # Terrain correction: SRTM 30m DEM assumed
    dem_resolution: int = 30

    # Incidence angle range for Sentinel-1 GRD (degrees)
    incidence_angle_min: float = 29.1
    incidence_angle_max: float = 46.0

    # Coherence estimation window
    coherence_window_size: int = 15
    coherence_azimuth_window: int = 10
    coherence_range_window: int = 10


@dataclass(frozen=True)
class ChangeDetectionConfig:
    """Parameters controlling SAR change detection logic."""

    # Minimum dB difference to register as a change
    log_ratio_threshold_db: float = 3.0

    # Minimum coherence drop to flag change
    coherence_drop_threshold: float = 0.25

    # Intensity ratio threshold (linear scale)
    intensity_ratio_high: float = 2.0
    intensity_ratio_low: float = 0.5

    # Valid pixel percentage to consider a change reliable
    min_valid_pixel_pct: float = 0.1

    # Scale factor for log-ratio computation
    epsilon: float = 1e-10


@dataclass(frozen=True)
class FeatureClassificationConfig:
    """Spatial morphological thresholds for anomaly classification."""

    # ── Personnel / Footsteps ──
    personnel_area_min: float = 5.0       # m^2
    personnel_area_max: float = 200.0     # m^2
    personnel_aspect_ratio_min: float = 4.0  # elongated
    personnel_length_max: float = 80.0    # meters
    personnel_width_max: float = 2.0      # meters
    personnel_backscatter_min_db: float = -18.0

    # ── Vehicle Tracks ──
    vehicle_area_min: float = 50.0        # m^2
    vehicle_area_max: float = 5000.0      # m^2
    vehicle_aspect_ratio_min: float = 3.0
    vehicle_width_min: float = 2.5        # meters (track gauge)
    vehicle_width_max: float = 6.0        # meters
    vehicle_backscatter_min_db: float = -15.0

    # ── Encampments / Structures ──
    encampment_area_min: float = 80.0     # m^2
    encampment_circularity_min: float = 0.35
    encampment_backscatter_min_db: float = -12.0
    encampment_double_bounce_ratio: float = 1.5  # VH/VV ratio threshold

    # Morphological operations
    morph_kernel_open: int = 3
    morph_kernel_close: int = 5
    min_blob_area: int = 10  # pixels


@dataclass(frozen=True)
class ThreatScoringConfig:
    """Weights and thresholds for the threat scoring matrix."""

    # Weight components (must sum to 1.0)
    weight_classification: float = 0.40
    weight_proximity: float = 0.35
    weight_temporal: float = 0.25

    # Classification severity scores (0–10 scale)
    severity_personnel: float = 3.0
    severity_vehicle: float = 6.5
    severity_encampment: float = 9.0
    severity_unknown: float = 1.0

    # Proximity zones (meters from border line)
    proximity_critical_m: float = 500.0
    proximity_warning_m: float = 2000.0
    proximity_safe_m: float = 5000.0

    # Scoring within proximity zones
    proximity_score_critical: float = 10.0
    proximity_score_warning: float = 6.0
    proximity_score_safe: float = 2.0

    # Temporal / velocity scoring
    velocity_approach_bonus: float = 2.0   # extra points if heading toward border
    velocity_recede_penalty: float = 3.0   # deducted if moving away
    max_velocity_kmh: float = 60.0         # normalization ceiling


@dataclass(frozen=True)
class ExportConfig:
    """GeoJSON export and coordinate reference system settings."""

    output_crs: str = "EPSG:4326"
    internal_crs: str = "EPSG:32643"       # UTM Zone 43N (common for Himalayas)
    decimal_precision: int = 6
    min_dissolve_distance_m: float = 50.0  # merge nearby detections
    export_timestamp_format: str = "%Y-%dT%H:%M:%SZ"
