"""
SnowTrace Module C: Threat Scoring Engine
──────────────────────────────────────────
Calculates a normalised Threat Score (0.0 – 10.0) for each detected
anomaly using a weighted multi-factor matrix:

    ThreatScore = w_cls·C + w_prox·P + w_temp·T

Where:
    C  = Classification Severity   (object type → base severity)
    P  = Proximity Score           (distance to border → risk zone)
    T  = Temporal / Velocity Score (heading & speed toward border)

Dynamic adjustments:
    - Approaching border → bonus added to T
    - Receding from border → penalty subtracted from T
    - Very close proximity (<500 m) overrides temporal to CRITICAL
"""

from __future__ import annotations

import logging
import math
from typing import List, Optional

import numpy as np

from snowtrace.config import ThreatScoringConfig
from snowtrace.models.data_models import (
    AnomalyType,
    DetectedFeature,
    HeadingVector,
    ThreatAssessment,
    ThreatLevel,
)
from snowtrace.utils.geo_utils import (
    EARTH_RADIUS_M,
    DEG_TO_RAD,
    haversine_distance_m,
    initial_bearing_deg,
    point_to_linestring_distance_m,
)

logger = logging.getLogger(__name__)


# ── Border Line Parsing ──────────────────────────────────────────────

def parse_border_line(border_wkt: str) -> list[tuple[float, float]]:
    """
    Parse a simplified WKT LINESTRING into coordinate pairs.

    Accepts formats:
      LINESTRING(lon1 lat1, lon2 lat2, ...)
      "lon1 lat1,lon2 lat2,..."   (plain CSV fallback)

    Returns
    -------
    list of (lon, lat) tuples.
    """
    coords = []
    wkt = border_wkt.strip()

    if wkt.upper().startswith("LINESTRING"):
        # Extract the coordinate string between parentheses
        start = wkt.index("(") + 1
        end = wkt.rindex(")")
        coord_str = wkt[start:end]
    else:
        coord_str = wkt

    for pair in coord_str.split(","):
        parts = pair.strip().split()
        if len(parts) >= 2:
            lon, lat = float(parts[0]), float(parts[1])
            coords.append((lon, lat))

    return coords


# ── Proximity Score ──────────────────────────────────────────────────

def compute_proximity_score(
    centroid_lon: float,
    centroid_lat: float,
    border_line_coords: list[tuple[float, float]],
    config: ThreatScoringConfig,
) -> Tuple[float, float]:
    """
    Compute the proximity-to-border score.

    Parameters
    ----------
    centroid_lon, centroid_lat : WGS-84 centroid of detected feature.
    border_line_coords         : list of (lon, lat) defining the border.
    config                     : scoring configuration.

    Returns
    -------
    (distance_m, score) where score ∈ [0, 10].
    """
    if not border_line_coords or len(border_line_coords) < 2:
        logger.warning("Border line has <2 coordinates; defaulting to safe score")
        return (config.proximity_safe_m, config.proximity_score_safe)

    dist_m = point_to_linestring_distance_m(
        centroid_lon, centroid_lat, border_line_coords
    )

    # Zone-based scoring with smooth interpolation
    if dist_m <= config.proximity_critical_m:
        score = config.proximity_score_critical
    elif dist_m <= config.proximity_warning_m:
        # Linear interpolation: critical → warning
        t = (dist_m - config.proximity_critical_m) / (
            config.proximity_warning_m - config.proximity_critical_m
        )
        score = config.proximity_score_critical + t * (
            config.proximity_score_warning - config.proximity_score_critical
        )
    elif dist_m <= config.proximity_safe_m:
        # Linear interpolation: warning → safe
        t = (dist_m - config.proximity_warning_m) / (
            config.proximity_safe_m - config.proximity_warning_m
        )
        score = config.proximity_score_warning + t * (
            config.proximity_score_safe - config.proximity_score_warning
        )
    else:
        score = config.proximity_score_safe

    return (dist_m, score)


# ── Classification Severity ──────────────────────────────────────────

def classification_severity(anomaly_type: AnomalyType, config: ThreatScoringConfig) -> float:
    """
    Map anomaly type to a severity score [0, 10].

    Hierarchy:  encampment > vehicle > personnel > unknown
    """
    severity_map = {
        AnomalyType.ENCAMPMENT_CLUSTER: config.severity_encampment,
        AnomalyType.VEHICLE_TRACK: config.severity_vehicle,
        AnomalyType.PERSONNEL_LINE: config.severity_personnel,
        AnomalyType.UNKNOWN: config.severity_unknown,
    }
    return severity_map.get(anomaly_type, config.severity_unknown)


# ── Temporal / Velocity Score ────────────────────────────────────────

def compute_temporal_score(
    heading: Optional[HeadingVector],
    border_line_coords: list[tuple[float, float]],
    centroid_lon: float,
    centroid_lat: float,
    config: ThreatScoringConfig,
) -> float:
    """
    Compute the temporal / velocity component of the threat score.

    Logic:
      - If no heading data available, return baseline (neutral).
      - Compute bearing from feature to nearest border point.
      - Compare with movement heading:
          * Same direction → approaching → add bonus
          * Opposite direction → receding → apply penalty
      - Scale by normalised velocity.

    Returns
    -------
    float ∈ [0, 10].
    """
    if heading is None or heading.velocity_kmh <= 0:
        return 5.0  # neutral baseline when no movement data

    # Find the nearest border point to compute target bearing
    if border_line_coords:
        min_dist = float("inf")
        nearest_border = border_line_coords[0]
        for bc in border_line_coords:
            d = haversine_distance_m(centroid_lon, centroid_lat, bc[0], bc[1])
            if d < min_dist:
                min_dist = d
                nearest_border = bc
        bearing_to_border = initial_bearing_deg(
            centroid_lon, centroid_lat,
            nearest_border[0], nearest_border[1]
        )
    else:
        bearing_to_border = 0.0

    # Angular difference (0–180°)
    angle_diff = abs(heading.direction_deg - bearing_to_border)
    if angle_diff > 180.0:
        angle_diff = 360.0 - angle_diff

    # Approaching if heading is within ±90° of bearing-to-border
    approaching = angle_diff < 90.0
    alignment_factor = math.cos(math.radians(angle_diff))  # 1.0 = perfect, -1.0 = opposite

    # Normalised velocity (0–1)
    vel_norm = min(heading.velocity_kmh / config.max_velocity_kmh, 1.0)

    # Base temporal score
    base_score = 5.0

    if approaching:
        # Boost: more aligned + faster → higher score
        base_score += config.velocity_approach_bonus * vel_norm * alignment_factor
    else:
        # Penalty: moving away reduces urgency
        base_score -= config.velocity_recede_penalty * vel_norm * abs(alignment_factor)

    return max(0.0, min(10.0, base_score))


# ── Composite Threat Score ───────────────────────────────────────────

def compute_threat_score(
    classification_score: float,
    proximity_score: float,
    temporal_score: float,
    distance_to_border_m: float,
    config: ThreatScoringConfig,
) -> Tuple[float, ThreatLevel]:
    """
    Combine weighted components into a single threat score.

    Special override: if distance < proximity_critical_m AND
    severity ≥ vehicle level, force CRITICAL.

    Returns
    -------
    (threat_score, ThreatLevel)
    """
    raw = (
        config.weight_classification * classification_score
        + config.weight_proximity * proximity_score
        + config.weight_temporal * temporal_score
    )

    # Clamp to [0, 10]
    threat_score = max(0.0, min(10.0, raw))

    # Override for critical proximity
    if (distance_to_border_m <= config.proximity_critical_m
            and classification_score >= config.severity_vehicle):
        threat_score = max(threat_score, 8.5)

    # Map to threat level
    if threat_score >= 8.0:
        level = ThreatLevel.CRITICAL
    elif threat_score >= 5.5:
        level = ThreatLevel.HIGH
    elif threat_score >= 3.0:
        level = ThreatLevel.MEDIUM
    else:
        level = ThreatLevel.LOW

    return (threat_score, level)


# ── Heading Estimation from Multi-Temporal Data ─────────────────────

def estimate_heading_from_positions(
    positions: list[tuple[float, float, float]],
) -> Optional[HeadingVector]:
    """
    Estimate heading vector from a time-ordered sequence of positions.

    Parameters
    ----------
    positions : list of (lon, lat, epoch_seconds) tuples,
                ordered chronologically (T0, T1, T2, ...).

    Returns
    -------
    HeadingVector or None if insufficient data.
    """
    if len(positions) < 2:
        return None

    # Use last two positions for most recent velocity
    lon1, lat1, t1 = positions[-2]
    lon2, lat2, t2 = positions[-1]

    dist_m = haversine_distance_m(lon1, lat1, lon2, lat2)
    time_h = abs(t2 - t1) / 3600.0

    if time_h < 1e-6:
        return None

    velocity_kmh = dist_m / time_h
    direction = initial_bearing_deg(lon1, lat1, lon2, lat2)

    return HeadingVector(
        direction_deg=direction,
        velocity_kmh=velocity_kmh,
        displacement_m=dist_m,
        time_delta_hours=time_h,
        approaching_border=False,  # caller can override
    )


# ── Main Threat Assessment Pipeline ─────────────────────────────────

def assess_threats(
    features: List[DetectedFeature],
    border_line_wkt: str,
    config: Optional[ThreatScoringConfig] = None,
    temporal_positions: Optional[dict[str, list[tuple[float, float, float]]]] = None,
) -> List[ThreatAssessment]:
    """
    Assess threat scores for all detected features.

    Parameters
    ----------
    features          : classified anomalies from Module B.
    border_line_wkt   : WKT LINESTRING or CSV of border coordinates.
    config            : threat scoring parameters.
    temporal_positions: optional dict mapping feature_id → [(lon, lat, epoch), ...]
                        used to compute heading vectors.

    Returns
    -------
    List[ThreatAssessment] : sorted by threat_score descending.
    """
    cfg = config or ThreatScoringConfig()
    border_coords = parse_border_line(border_line_wkt)

    logger.info("Assessing threats for %d features against border (%d vertices)",
                len(features), len(border_coords))

    assessments: List[ThreatAssessment] = []

    for feat in features:
        # 1. Classification severity
        cls_score = classification_severity(feat.anomaly_type, cfg)

        # 2. Proximity to border
        dist_m, prox_score = compute_proximity_score(
            feat.centroid_lon, feat.centroid_lat,
            border_coords, cfg,
        )

        # 3. Temporal / heading
        heading = None
        if temporal_positions and feat.feature_id in temporal_positions:
            heading = estimate_heading_from_positions(
                temporal_positions[feat.feature_id]
            )
            # Determine if approaching border
            if heading is not None:
                nearest = min(
                    border_coords,
                    key=lambda b: haversine_distance_m(
                        feat.centroid_lon, feat.centroid_lat, b[0], b[1]
                    ),
                )
                bearing = initial_bearing_deg(
                    feat.centroid_lon, feat.centroid_lat,
                    nearest[0], nearest[1],
                )
                angle = abs(heading.direction_deg - bearing)
                if angle > 180:
                    angle = 360 - angle
                heading.approaching_border = angle < 90.0

        temp_score = compute_temporal_score(
            heading, border_coords,
            feat.centroid_lon, feat.centroid_lat,
            cfg,
        )

        # 4. Composite score
        score, level = compute_threat_score(
            cls_score, prox_score, temp_score,
            dist_m, cfg,
        )

        assessment = ThreatAssessment(
            assessment_id=f"assess_{feat.feature_id}",
            feature_id=feat.feature_id,
            anomaly_type=feat.anomaly_type,
            threat_score=round(score, 2),
            threat_level=level,
            distance_to_border_m=round(dist_m, 1),
            proximity_component=round(prox_score, 2),
            classification_component=round(cls_score, 2),
            temporal_component=round(temp_score, 2),
            heading=heading,
            centroid_lon=feat.centroid_lon,
            centroid_lat=feat.centroid_lat,
            metadata={
                "area_m2": feat.area_m2,
                "aspect_ratio": feat.aspect_ratio,
                "backscatter_db": feat.mean_backscatter_db,
            },
        )
        assessments.append(assessment)

    assessments.sort(key=lambda a: a.threat_score, reverse=True)

    level_counts = {lv.value: 0 for lv in ThreatLevel}
    for a in assessments:
        level_counts[a.threat_level.value] += 1
    logger.info("Threat distribution: %s", level_counts)

    return assessments
