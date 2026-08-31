"""
SnowTrace Module D: GeoJSON Exporter
─────────────────────────────────────
Converts threat assessments and detected features into structured
GeoJSON payloads suitable for Leaflet / Mapbox / Kepler.gl rendering.

Output schema:
  {
    "type": "FeatureCollection",
    "metadata": { ... },
    "features": [
      {
        "type": "Feature",
        "geometry": { "type": "Point" | "Polygon", ... },
        "properties": {
          "feature_id": "...",
          "anomaly_type": "...",
          "threat_score": 0-10,
          "threat_level": "low|medium|high|critical",
          "distance_to_border_m": ...,
          "confidence": ...,
          ...
        }
      }
    ]
  }
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from snowtrace.config import ExportConfig, ThreatScoringConfig
from snowtrace.models.data_models import (
    DetectedFeature,
    GeoJSONFeature,
    ThreatAssessment,
    ThreatLevel,
)

logger = logging.getLogger(__name__)


# ── WKT → GeoJSON Geometry ──────────────────────────────────────────

def wkt_to_geojson_geometry(wkt: str) -> Dict[str, Any]:
    """
    Convert a WKT geometry string to GeoJSON geometry dict.

    Supports: POINT, LINESTRING, POLYGON.
    """
    wkt_upper = wkt.strip().upper()

    if wkt_upper.startswith("POINT"):
        # POINT(lon lat)
        coord_str = wkt[wkt.index("(") + 1:wkt.rindex(")")]
        lon, lat = coord_str.split()[:2]
        return {"type": "Point", "coordinates": [float(lon), float(lat)]}

    if wkt_upper.startswith("LINESTRING"):
        coord_str = wkt[wkt.index("(") + 1:wkt.rindex(")")]
        coords = []
        for pair in coord_str.split(","):
            parts = pair.strip().split()
            coords.append([float(parts[0]), float(parts[1])])
        return {"type": "LineString", "coordinates": coords}

    if wkt_upper.startswith("POLYGON"):
        coord_str = wkt[wkt.index("(") + 1:wkt.rindex(")")]
        # For POLYGON, strip one level of parens
        if coord_str.startswith("("):
            coord_str = coord_str[1:coord_str.rindex(")")]
        ring = []
        for pair in coord_str.split(","):
            parts = pair.strip().split()
            ring.append([float(parts[0]), float(parts[1])])
        return {"type": "Polygon", "coordinates": [ring]}

    # Fallback: return empty geometry
    logger.warning("Unrecognised WKT type: %s", wkt[:20])
    return {"type": "Point", "coordinates": [0.0, 0.0]}


# ── GeoJSON Feature Builder ─────────────────────────────────────────

def build_feature_properties(
    assessment: ThreatAssessment,
    feature: Optional[DetectedFeature] = None,
) -> Dict[str, Any]:
    """
    Merge assessment + original feature data into a flat properties
    dictionary for the GeoJSON feature.
    """
    props: Dict[str, Any] = {
        "feature_id": assessment.feature_id,
        "anomaly_type": assessment.anomaly_type.value,
        "threat_score": assessment.threat_score,
        "threat_level": assessment.threat_level.value,
        "distance_to_border_m": assessment.distance_to_border_m,
        "assessment_time": assessment.assessed_at.isoformat(),
        # Component scores for debugging / UI tooltips
        "score_breakdown": {
            "classification": assessment.classification_component,
            "proximity": assessment.proximity_component,
            "temporal": assessment.temporal_component,
        },
    }

    if feature is not None:
        props.update({
            "area_m2": round(feature.area_m2, 1),
            "length_m": round(feature.length_m, 1),
            "width_m": round(feature.width_m, 1),
            "aspect_ratio": round(feature.aspect_ratio, 2),
            "circularity": round(feature.circularity, 3),
            "backscatter_db": round(feature.mean_backscatter_db, 2),
            "max_backscatter_db": round(feature.max_backscatter_db, 2),
            "vh_vv_ratio": round(feature.vh_vv_ratio, 3),
            "pixel_count": feature.pixel_count,
            "confidence": round(feature.confidence, 3),
        })

    if assessment.heading is not None:
        props["heading"] = {
            "direction_deg": round(assessment.heading.direction_deg, 1),
            "velocity_kmh": round(assessment.heading.velocity_kmh, 2),
            "approaching_border": assessment.heading.approaching_border,
        }

    # Map threat level to a color class for frontend styling
    color_map = {
        ThreatLevel.CRITICAL: "#dc2626",  # red
        ThreatLevel.HIGH: "#ea580c",      # orange
        ThreatLevel.MEDIUM: "#eab308",    # yellow
        ThreatLevel.LOW: "#22c55e",       # green
    }
    props["color"] = color_map.get(assessment.threat_level, "#6b7280")

    # Marker radius proportional to threat score (for circle markers)
    props["marker_radius"] = max(4, int(assessment.threat_score * 2))

    return props


# ── GeoJSON Collection Builder ───────────────────────────────────────

def build_geojson_collection(
    assessments: List[ThreatAssessment],
    features: Optional[List[DetectedFeature]] = None,
    export_config: Optional[ExportConfig] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Build a complete GeoJSON FeatureCollection from threat assessments.

    Parameters
    ----------
    assessments   : list of ThreatAssessment from Module C.
    features      : optional list of DetectedFeature from Module B.
                    Used to enrich properties with geometry metrics.
    export_config : export settings.
    metadata      : optional metadata dict (scene IDs, timestamps, etc.)

    Returns
    -------
    dict : GeoJSON FeatureCollection.
    """
    cfg = export_config or ExportConfig()

    # Index features by ID for quick lookup
    feat_index = {}
    if features:
        feat_index = {f.feature_id: f for f in features}

    geojson_features = []

    for assessment in assessments:
        feat = feat_index.get(assessment.feature_id)

        # Determine geometry source
        if feat is not None and feat.geometry_wkt:
            geometry = wkt_to_geojson_geometry(feat.geometry_wkt)
        else:
            # Fallback: Point geometry from centroid
            geometry = {
                "type": "Point",
                "coordinates": [
                    round(assessment.centroid_lon, cfg.decimal_precision),
                    round(assessment.centroid_lat, cfg.decimal_precision),
                ],
            }

        properties = build_feature_properties(assessment, feat)

        geojson_features.append(GeoJSONFeature(
            feature_type="Feature",
            geometry=geometry,
            properties=properties,
        ))

    # Assemble FeatureCollection
    collection = {
        "type": "FeatureCollection",
        "metadata": {
            "generated_at": datetime.utcnow().strftime(cfg.export_timestamp_format),
            "total_features": len(geojson_features),
            "crs": cfg.output_crs,
            "threat_summary": _summarise_threats(assessments),
            **(metadata or {}),
        },
        "features": [f.to_dict() for f in geojson_features],
    }

    logger.info("Exported GeoJSON FeatureCollection: %d features",
                len(geojson_features))

    return collection


# ── Threat Summary ───────────────────────────────────────────────────

def _summarise_threats(assessments: List[ThreatAssessment]) -> Dict[str, Any]:
    """Compute aggregate statistics for the threat summary."""
    if not assessments:
        return {"total": 0}

    scores = [a.threat_score for a in assessments]
    level_counts = {lv.value: 0 for lv in ThreatLevel}
    for a in assessments:
        level_counts[a.threat_level.value] += 1

    return {
        "total": len(assessments),
        "mean_score": round(sum(scores) / len(scores), 2),
        "max_score": round(max(scores), 2),
        "min_score": round(min(scores), 2),
        "by_level": level_counts,
        "by_type": _count_by_type(assessments),
    }


def _count_by_type(assessments: List[ThreatAssessment]) -> Dict[str, int]:
    """Count features per anomaly type."""
    counts: Dict[str, int] = {}
    for a in assessments:
        key = a.anomaly_type.value
        counts[key] = counts.get(key, 0) + 1
    return counts


# ── Serialisation Helpers ────────────────────────────────────────────

def to_json_string(geojson: Dict[str, Any], indent: int = 2) -> str:
    """Serialize GeoJSON dict to a JSON string."""
    return json.dumps(geojson, indent=indent, default=str)


def save_geojson(geojson: Dict[str, Any], output_path: str) -> str:
    """
    Write GeoJSON FeatureCollection to a file.

    Parameters
    ----------
    geojson      : FeatureCollection dict.
    output_path  : file path (e.g. "threats_2026-08-28.geojson").

    Returns
    -------
    str : absolute path of written file.
    """
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(geojson, f, indent=2, default=str)

    logger.info("GeoJSON written to: %s", output_path)
    return output_path


# ── Mapbox-Specific Styling Layer ────────────────────────────────────

def generate_mapbox_style(geojson: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate a minimal Mapbox GL style spec that renders the
    GeoJSON threat layer with color-coded circle markers.
    """
    return {
        "version": 8,
        "name": "SnowTrace Threats",
        "sources": {
            "snowtrace-threats": {
                "type": "geojson",
                "data": geojson,
            }
        },
        "layers": [
            {
                "id": "threat-points",
                "type": "circle",
                "source": "snowtrace-threats",
                "layout": {
                    "circle-sort-key": ["get", "threat_score"],
                },
                "paint": {
                    "circle-radius": ["get", "marker_radius"],
                    "circle-color": ["get", "color"],
                    "circle-stroke-width": 1.5,
                    "circle-stroke-color": "#1f2937",
                    "circle-opacity": 0.85,
                },
            },
            {
                "id": "threat-labels",
                "type": "symbol",
                "source": "snowtrace-threats",
                "layout": {
                    "text-field": ["to-string", ["get", "threat_score"]],
                    "text-size": 10,
                    "text-offset": [0, -1.5],
                },
                "paint": {
                    "text-color": "#ffffff",
                    "text-halo-color": "#000000",
                    "text-halo-width": 0.5,
                },
            },
        ],
    }
