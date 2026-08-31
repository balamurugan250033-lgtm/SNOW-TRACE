"""
SnowTrace Data Models
────────────────────
Dataclass definitions for SAR frames, detected features, threat
assessments, and pipeline outputs.  Kept lean so they serialise
cleanly to JSON / GeoJSON.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


# ── Enums ────────────────────────────────────────────────────────────

class AnomalyType(str, Enum):
    PERSONNEL_LINE = "personnel_line"
    VEHICLE_TRACK = "vehicle_track"
    ENCAMPMENT_CLUSTER = "encampment_cluster"
    UNKNOWN = "unknown"


class ThreatLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class Polarization(str, Enum):
    VV = "VV"
    VH = "VH"


# ── SAR Frame ────────────────────────────────────────────────────────

@dataclass
class SARFrame:
    """A single Sentinel-1 SAR acquisition frame."""

    acquisition_id: str
    acquisition_time: datetime
    file_path: str
    polarization: Polarization
    orbit_direction: Optional[str] = None   # ASCENDING / DESCENDING
    relative_orbit: Optional[int] = None
    footprint_wkt: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.acquisition_id:
            self.acquisition_id = str(uuid.uuid4())[:12]


# ── Change Map ───────────────────────────────────────────────────────

@dataclass
class ChangeMap:
    """Result of SAR differencing between two frames."""

    change_map_id: str
    baseline_frame: SARFrame
    current_frame: SARFrame
    polarization: Polarization
    log_ratio_array: Optional[Any] = None       # np.ndarray
    coherence_array: Optional[Any] = None       # np.ndarray
    change_mask: Optional[Any] = None           # np.ndarray (bool)
    transform: Optional[Any] = None             # affine transform
    crs: Optional[str] = None
    pixel_resolution_m: float = 10.0

    def __post_init__(self) -> None:
        if not self.change_map_id:
            self.change_map_id = f"{self.baseline_frame.acquisition_id}_{self.current_frame.acquisition_id}"


# ── Detected Feature ─────────────────────────────────────────────────

@dataclass
class DetectedFeature:
    """A single spatial anomaly extracted from the change map."""

    feature_id: str
    anomaly_type: AnomalyType
    geometry_wkt: str                          # WKT Polygon or LineString
    centroid_lon: float = 0.0
    centroid_lat: float = 0.0
    area_m2: float = 0.0
    length_m: float = 0.0
    width_m: float = 0.0
    aspect_ratio: float = 0.0
    circularity: float = 0.0
    mean_backscatter_db: float = 0.0
    max_backscatter_db: float = 0.0
    vh_vv_ratio: float = 0.0
    pixel_count: int = 0
    confidence: float = 0.0
    properties: Dict[str, Any] = field(default_factory=dict)


# ── Heading Vector ───────────────────────────────────────────────────

@dataclass
class HeadingVector:
    """Temporal displacement vector for a tracked feature."""

    direction_deg: float = 0.0         # 0–360 compass bearing
    velocity_kmh: float = 0.0
    displacement_m: float = 0.0
    time_delta_hours: float = 0.0
    approaching_border: bool = False


# ── Threat Assessment ────────────────────────────────────────────────

@dataclass
class ThreatAssessment:
    """Final threat evaluation for one detected feature."""

    assessment_id: str
    feature_id: str
    anomaly_type: AnomalyType
    threat_score: float = 0.0         # 0.0 – 10.0
    threat_level: ThreatLevel = ThreatLevel.LOW
    distance_to_border_m: float = float("inf")
    proximity_component: float = 0.0
    classification_component: float = 0.0
    temporal_component: float = 0.0
    heading: Optional[HeadingVector] = None
    centroid_lon: float = 0.0
    centroid_lat: float = 0.0
    assessed_at: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)


# ── GeoJSON Feature ─────────────────────────────────────────────────

@dataclass
class GeoJSONFeature:
    """A single GeoJSON Feature ready for map rendering."""

    feature_type: str = "Feature"
    geometry: Dict[str, Any] = field(default_factory=dict)
    properties: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.feature_type,
            "geometry": self.geometry,
            "properties": self.properties,
        }
