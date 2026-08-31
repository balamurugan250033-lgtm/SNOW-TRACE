"""
SnowTrace Module B: Feature Classifier
───────────────────────────────────────
Extracts spatial features from SAR change regions and classifies them
into threat categories based on morphological analysis:

  - personnel_line   → linear, narrow, low–moderate backscatter
  - vehicle_track    → parallel wide tracks, moderate backscatter
  - encampment_cluster → compact blobs, high backscatter / double-bounce

Classification uses connected-component labelling, bounding-box
geometry, aspect ratio, circularity, and backscatter statistics.
"""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import ndimage

from snowtrace.config import FeatureClassificationConfig
from snowtrace.models.data_models import (
    AnomalyType,
    ChangeMap,
    DetectedFeature,
    Polarization,
)
from snowtrace.utils.geo_utils import (
    bounding_box,
    haversine_distance_m,
    linear_to_db,
    pixel_to_geo,
    polygon_centroid,
)

logger = logging.getLogger(__name__)


# ── Connected-Component Labelling ────────────────────────────────────

def label_regions(change_mask: np.ndarray, min_area: int = 10) -> Tuple[np.ndarray, int]:
    """
    Label connected components in the binary change mask.

    Parameters
    ----------
    change_mask : boolean array.
    min_area    : minimum pixel count to keep a label.

    Returns
    -------
    (labeled_array, num_labels)
    """
    struct = ndimage.generate_binary_structure(2, 1)  # 4-connectivity
    labeled, num_features = ndimage.label(change_mask, structure=struct)

    # Remove small components
    for i in range(1, num_features + 1):
        if np.sum(labeled == i) < min_area:
            labeled[labeled == i] = 0

    # Re-label after removal
    labeled[labeled > 0] = ndimage.label(labeled > 0)[0][labeled > 0]
    final_count = labeled.max()
    return labeled, final_count


# ── Region Geometry Extraction ───────────────────────────────────────

def extract_region_geometry(
    labeled: np.ndarray,
    region_id: int,
    transform: Optional[object],
    pixel_resolution_m: float,
) -> Dict:
    """
    Extract morphological metrics for a single labelled region.

    Metrics computed:
      - area_m2            : region area in square metres
      - length_m           : major axis length in metres
      - width_m            : minor axis length in metres
      - aspect_ratio       : length / width
      - circularity        : 4π·area / perimeter²  (1.0 = circle)
      - mean_backscatter_db: mean intensity in dB within region
      - max_backscatter_db : peak intensity in dB within region
      - centroid_pixel     : (row, col) pixel centroid
      - centroid_geo       : (lon, lat) if transform available
      - bounding_box       : (min_lon, min_lat, max_lon, max_lat)
      - orientation_deg    : angle of major axis from horizontal
      - perimeter_m        : approximate perimeter in metres
    """
    mask = labeled == region_id
    coords = np.argwhere(mask)  # (row, col) pairs

    area_px = coords.shape[0]
    area_m2 = area_px * (pixel_resolution_m ** 2)

    # Bounding box in pixel space
    row_min, col_min = coords.min(axis=0)
    row_max, col_max = coords.max(axis=0)
    bbox_rows = row_max - row_min
    bbox_cols = col_max - col_min

    # Approximate perimeter using boundary detection
    from scipy.ndimage import binary_dilation
    dilated = binary_dilation(mask)
    boundary = dilated & ~mask
    perimeter_px = np.sum(boundary)
    perimeter_m = perimeter_px * pixel_resolution_m

    # Centroid
    centroid_row = coords[:, 0].mean()
    centroid_col = coords[:, 1].mean()

    # Orientation via second-order central moments (image moments)
    r = coords[:, 0].astype(np.float64)
    c = coords[:, 1].astype(np.float64)
    r_mean, c_mean = r.mean(), c.mean()

    mu_rr = np.mean((r - r_mean) ** 2)
    mu_cc = np.mean((c - c_mean) ** 2)
    mu_rc = np.mean((r - r_mean) * (c - c_mean))

    # Major and minor axis lengths (eigenvalues of covariance)
    cov = np.array([[mu_rr, mu_rc], [mu_rc, mu_cc]])
    eigvals = np.linalg.eigvalsh(cov)
    eigvals = np.maximum(eigvals, 0.0)

    length_px = 2.0 * math.sqrt(6.0 * eigvals.max())   # 2√6·σ approximation
    width_px = 2.0 * math.sqrt(6.0 * eigvals.min()) if eigvals.min() > 0 else 1.0

    length_m = length_px * pixel_resolution_m
    width_m = width_px * pixel_resolution_m
    aspect_ratio = length_m / max(width_m, 0.1)

    # Circularity:  4π·A / P²
    circularity = (4.0 * math.pi * area_m2 / (perimeter_m ** 2)) if perimeter_m > 0 else 0.0
    circularity = min(circularity, 1.0)

    # Orientation angle (from horizontal, clockwise)
    if mu_cc - mu_rr != 0:
        orientation_deg = 0.5 * math.atan2(2.0 * mu_rc, mu_cc - mu_rr) * (180.0 / math.pi)
    else:
        orientation_deg = 0.0

    # Geo-coordinate centroid
    if transform is not None:
        centroid_lon, centroid_lat = pixel_to_geo(
            int(round(centroid_row)), int(round(centroid_col)), transform
        )
        # Bounding box corners
        geo_coords = []
        for r_px, c_px in [(row_min, col_min), (row_max, col_max)]:
            geo_coords.append(pixel_to_geo(r_px, c_px, transform))
        bbox = bounding_box(geo_coords)
    else:
        centroid_lon, centroid_lat = 0.0, 0.0
        bbox = (0.0, 0.0, 0.0, 0.0)

    return {
        "area_m2": area_m2,
        "length_m": length_m,
        "width_m": width_m,
        "aspect_ratio": aspect_ratio,
        "circularity": circularity,
        "perimeter_m": perimeter_m,
        "orientation_deg": orientation_deg,
        "centroid_row": centroid_row,
        "centroid_col": centroid_col,
        "centroid_lon": centroid_lon,
        "centroid_lat": centroid_lat,
        "bbox": bbox,
        "pixel_count": area_px,
    }


# ── Backscatter Statistics ───────────────────────────────────────────

def compute_backscatter_stats(
    change_map: ChangeMap,
    labeled: np.ndarray,
    region_id: int,
) -> Tuple[float, float, float]:
    """
    Compute backscatter statistics for a labelled region.

    Returns
    -------
    (mean_db, max_db, vh_vv_ratio)
    """
    mask = labeled == region_id

    # Use log-ratio values (already in dB)
    if change_map.log_ratio_array is not None:
        log_vals = change_map.log_ratio_array[mask]
        mean_db = float(np.mean(log_vals))
        max_db = float(np.max(log_vals))
    else:
        mean_db = 0.0
        max_db = 0.0

    # VH/VV ratio proxy: use coherence as a stand-in when
    # only single-pol data is available
    vh_vv_ratio = 1.0
    if change_map.coherence_array is not None:
        coh_vals = change_map.coherence_array[mask]
        vh_vv_ratio = float(np.mean(coh_vals))

    return mean_db, max_db, vh_vv_ratio


# ── Classification Logic ────────────────────────────────────────────

def classify_region(
    geometry: Dict,
    mean_backscatter_db: float,
    max_backscatter_db: float,
    vh_vv_ratio: float,
    config: FeatureClassificationConfig,
) -> Tuple[AnomalyType, float]:
    """
    Classify a detected region into an anomaly type based on its
    morphometric and radiometric properties.

    Decision tree (priority order):

    1. ENCAMPMENT_CLUSTER
       - High circularity (>0.35)
       - Large area (>80 m²)
       - High backscatter (>-12 dB) → double-bounce from metal
       - Aspect ratio relatively low (compact)

    2. VEHICLE_TRACK
       - Elongated (aspect ratio > 3)
       - Width in 2.5–6.0 m range (track gauge)
       - Moderate backscatter (>-15 dB)
       - Large area (>50 m²)

    3. PERSONNEL_LINE
       - Very elongated (aspect ratio > 4)
       - Narrow (<2 m wide)
       - Low–moderate backscatter (>-18 dB)
       - Moderate area

    4. UNKNOWN
       - Fallback for ambiguous shapes.

    Returns
    -------
    (AnomalyType, confidence_float)
    """
    ar = geometry["aspect_ratio"]
    area = geometry["area_m2"]
    circ = geometry["circularity"]
    width = geometry["width_m"]

    scores: Dict[AnomalyType, float] = {
        AnomalyType.ENCAMPMENT_CLUSTER: 0.0,
        AnomalyType.VEHICLE_TRACK: 0.0,
        AnomalyType.PERSONNEL_LINE: 0.0,
        AnomalyType.UNKNOWN: 0.0,
    }

    # ── Encampment scoring ──────────────────────────────────────
    if (area >= config.encampment_area_min
            and circ >= config.encampment_circularity_min
            and mean_backscatter_db >= config.encampment_backscatter_min_db):
        scores[AnomalyType.ENCAMPMENT_CLUSTER] += 3.0
        # Double-bounce bonus
        if vh_vv_ratio >= config.encampment_double_bounce_ratio:
            scores[AnomalyType.ENCAMPMENT_CLUSTER] += 2.0
        # Penalise very elongated shapes (not encampments)
        if ar > 3.0:
            scores[AnomalyType.ENCAMPMENT_CLUSTER] -= 1.0

    # ── Vehicle track scoring ───────────────────────────────────
    if (config.vehicle_area_min <= area <= config.vehicle_area_max
            and ar >= config.vehicle_aspect_ratio_min
            and config.vehicle_width_min <= width <= config.vehicle_width_max
            and mean_backscatter_db >= config.vehicle_backscatter_min_db):
        scores[AnomalyType.VEHICLE_TRACK] += 3.0
        # Wider tracks more confident
        if 3.5 <= width <= 5.0:
            scores[AnomalyType.VEHICLE_TRACK] += 1.0

    # ── Personnel line scoring ──────────────────────────────────
    if (config.personnel_area_min <= area <= config.personnel_area_max
            and ar >= config.personnel_aspect_ratio_min
            and width <= config.personnel_width_max
            and mean_backscatter_db >= config.personnel_backscatter_min_db):
        scores[AnomalyType.PERSONNEL_LINE] += 3.0
        # Narrower = more confident (single file)
        if width <= 1.5:
            scores[AnomalyType.PERSONNEL_LINE] += 1.5
        # Very long trails boost confidence
        if geometry["length_m"] > 40.0:
            scores[AnomalyType.PERSONNEL_LINE] += 1.0

    # ── Fallback ────────────────────────────────────────────────
    max_score = max(scores.values())
    if max_score < 1.0:
        scores[AnomalyType.UNKNOWN] = 1.0

    # Select winner
    best_type = max(scores, key=lambda k: scores[k])
    confidence = min(scores[best_type] / 6.0, 1.0)  # normalise to [0, 1]

    return best_type, confidence


# ── Manual Geo-Coordinate Assignment ─────────────────────────────────

def assign_geo_coords(
    geometry: Dict,
    region_mask: np.ndarray,
    transform: Optional[object],
    pixel_resolution_m: float,
) -> Tuple[float, float]:
    """
    Compute the centroid geo-coordinate for a region.
    If transform is available, uses pixel→geo conversion;
    otherwise returns 0,0.
    """
    if transform is not None:
        return geometry["centroid_lon"], geometry["centroid_lat"]

    # Fallback: use pixel centroid and assume a reference origin
    return 0.0, 0.0


# ── Main Classification Pipeline ────────────────────────────────────

def classify_anomalies(
    change_map: ChangeMap,
    config: Optional[FeatureClassificationConfig] = None,
) -> List[DetectedFeature]:
    """
    Extract and classify all anomalous regions from a change map.

    Parameters
    ----------
    change_map : ChangeMap from Module A (sar_difference).
    config     : classification thresholds.

    Returns
    -------
    List[DetectedFeature] : classified anomalies sorted by area (desc).
    """
    cfg = config or FeatureClassificationConfig()

    if change_map.change_mask is None or not np.any(change_map.change_mask):
        logger.warning("No change pixels found — returning empty feature list")
        return []

    logger.info("Labelling connected regions in change mask")
    labeled, num_regions = label_regions(
        change_map.change_mask,
        min_area=cfg.min_blob_area,
    )
    logger.info("Found %d candidate regions", num_regions)

    features: List[DetectedFeature] = []

    for region_id in range(1, num_regions + 1):
        # Extract geometry
        geom = extract_region_geometry(
            labeled, region_id,
            change_map.transform,
            change_map.pixel_resolution_m,
        )

        # Backscatter stats
        mean_db, max_db, vh_vv_ratio = compute_backscatter_stats(
            change_map, labeled, region_id
        )

        # Classify
        anomaly_type, confidence = classify_region(
            geom, mean_db, max_db, vh_vv_ratio, cfg
        )

        # Build WKT geometry (approximate rectangle from bounding box)
        bb = geom["bbox"]
        wkt = _bbox_to_wkt(bb)

        feature = DetectedFeature(
            feature_id=f"feat_{region_id:04d}",
            anomaly_type=anomaly_type,
            geometry_wkt=wkt,
            centroid_lon=geom["centroid_lon"],
            centroid_lat=geom["centroid_lat"],
            area_m2=geom["area_m2"],
            length_m=geom["length_m"],
            width_m=geom["width_m"],
            aspect_ratio=geom["aspect_ratio"],
            circularity=geom["circularity"],
            mean_backscatter_db=mean_db,
            max_backscatter_db=max_db,
            vh_vv_ratio=vh_vv_ratio,
            pixel_count=geom["pixel_count"],
            confidence=confidence,
            properties={
                "orientation_deg": geom["orientation_deg"],
                "perimeter_m": geom["perimeter_m"],
                "bbox_pixel": (geom["centroid_row"], geom["centroid_col"]),
            },
        )
        features.append(feature)

    # Sort by area descending (most significant first)
    features.sort(key=lambda f: f.area_m2, reverse=True)
    logger.info("Classified %d features: %s",
                len(features),
                {t.value: sum(1 for f in features if f.anomaly_type == t)
                 for t in AnomalyType})

    return features


# ── Geometry Helpers ─────────────────────────────────────────────────

def _bbox_to_wkt(bbox: Tuple[float, float, float, float]) -> str:
    """Convert (min_lon, min_lat, max_lon, max_lat) to WKT POLYGON."""
    min_lon, min_lat, max_lon, max_lat = bbox
    if min_lon == max_lon == min_lat == max_lat == 0.0:
        return "POLYGON((0 0, 0 0, 0 0, 0 0, 0 0))"
    return (
        f"POLYGON(({min_lon} {min_lat}, {max_lon} {min_lat}, "
        f"{max_lon} {max_lat}, {min_lon} {max_lat}, {min_lon} {min_lat}))"
    )
