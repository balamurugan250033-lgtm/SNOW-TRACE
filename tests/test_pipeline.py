"""
SnowTrace Integration Test
──────────────────────────
Synthetic test that exercises the full pipeline with
artificial SAR data, verifying each module produces valid output.
"""

import sys
import os
import json
import logging
from datetime import datetime

import numpy as np

# Ensure the project root is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from snowtrace.config import (
    SARConfig,
    ChangeDetectionConfig,
    FeatureClassificationConfig,
    ThreatScoringConfig,
    ExportConfig,
)
from snowtrace.models.data_models import (
    AnomalyType,
    ChangeMap,
    DetectedFeature,
    HeadingVector,
    Polarization,
    SARFrame,
    ThreatLevel,
)
from snowtrace.core.sar_difference import (
    lee_filter,
    compute_log_ratio,
    estimate_coherence,
    threshold_change_mask,
    morphological_cleanup,
    detect_changes,
)
from snowtrace.core.feature_classifier import (
    label_regions,
    extract_region_geometry,
    classify_region,
    classify_anomalies,
)
from snowtrace.core.threat_engine import (
    parse_border_line,
    compute_proximity_score,
    classification_severity,
    compute_temporal_score,
    compute_threat_score,
    assess_threats,
)
from snowtrace.core.geo_export import (
    wkt_to_geojson_geometry,
    build_geojson_collection,
    save_geojson,
    to_json_string,
)
from snowtrace.pipeline import SnowTracePipeline, run_snowtrace

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def make_test_frames() -> tuple[SARFrame, SARFrame]:
    """Create two synthetic SAR frame metadata objects."""
    t0 = SARFrame(
        acquisition_id="S1_GRD_20260801",
        acquisition_time=datetime(2026, 8, 1, 5, 30, 0),
        file_path="/data/s1_t0.tif",
        polarization=Polarization.VV,
        orbit_direction="ASCENDING",
        relative_orbit=42,
        footprint_wkt="POLYGON((78.0 30.0, 78.5 30.0, 78.5 30.5, 78.0 30.5, 78.0 30.0))",
    )
    t1 = SARFrame(
        acquisition_id="S1_GRD_20260813",
        acquisition_time=datetime(2026, 8, 13, 5, 30, 0),
        file_path="/data/s1_t1.tif",
        polarization=Polarization.VV,
        orbit_direction="ASCENDING",
        relative_orbit=42,
        footprint_wkt="POLYGON((78.0 30.0, 78.5 30.0, 78.5 30.5, 78.0 30.5, 78.0 30.0))",
    )
    return t0, t1


def make_synthetic_sar_data(
    rows: int = 512, cols: int = 512, seed: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic SAR intensity arrays with embedded anomalies.

    Baseline: uniform snow-covered scene with speckle.
    Current: same scene + three injected anomalies:
      - Narrow linear track (personnel_line)  at (50-120, 100)
      - Wide parallel track (vehicle_track)    at (200-280, 300)
      - Bright blob (encampment_cluster)       at (350-400, 350-400)
    """
    rng = np.random.RandomState(seed)

    # Baseline: snow-covered surface (low-to-moderate backscatter)
    baseline = 0.05 + rng.exponential(0.02, (rows, cols))

    # Current: mostly identical + anomalies
    current = baseline.copy()

    # Personnel track: narrow line of increased backscatter
    current[50:120, 100:102] += 0.15  # ~3 dB rise

    # Vehicle track: wider parallel strips
    current[200:280, 300:305] += 0.20  # ~4 dB rise
    current[200:280, 310:315] += 0.20  # second track
    current[200:205, 300:315] += 0.15  # cross-link

    # Encampment: high-backscatter cluster
    rr, cc = np.ogrid[350:400, 350:400]
    blob_mask = ((rr - 375) ** 2 + (cc - 375) ** 2) < 20 ** 2
    current[350:400, 350:400][blob_mask] += 0.35  # ~8 dB rise (metal)

    return baseline, current


# ── Test: Module A ──────────────────────────────────────────────────

def test_module_a_sar_difference():
    logger.info("=" * 50)
    logger.info("TEST: Module A — SAR Difference Engine")
    baseline, current = make_synthetic_sar_data()

    # Lee filter
    filtered = lee_filter(baseline, kernel_size=7)
    assert filtered.shape == baseline.shape, "Lee filter shape mismatch"
    logger.info("  Lee filter: OK")

    # Log-ratio
    log_ratio = compute_log_ratio(baseline, current)
    assert log_ratio.shape == baseline.shape
    assert np.any(np.abs(log_ratio) > 1.0), "Expected significant changes"
    logger.info("  Log-ratio: OK (max dB = %.2f)", np.max(np.abs(log_ratio)))

    # Coherence
    coh = estimate_coherence(baseline, current, window_size=7, azimuth_window=5, range_window=5)
    assert coh.shape == baseline.shape
    assert np.all((coh >= 0) & (coh <= 1)), "Coherence out of range"
    logger.info("  Coherence: OK (mean = %.3f)", np.mean(coh))

    # Change mask
    cfg = ChangeDetectionConfig()
    mask = threshold_change_mask(log_ratio, coh, cfg)
    assert mask.dtype == bool
    assert np.sum(mask) > 0, "No changes detected"
    logger.info("  Threshold mask: OK (%d px changed)", np.sum(mask))

    # Morphological cleanup
    cleaned = morphological_cleanup(mask)
    assert cleaned.dtype == bool
    logger.info("  Morphology: OK (%d px after cleanup)", np.sum(cleaned))

    logger.info("TEST Module A: PASSED\n")
    return True


# ── Test: Module B ──────────────────────────────────────────────────

def test_module_b_feature_classifier():
    logger.info("=" * 50)
    logger.info("TEST: Module B — Feature Classifier")
    baseline, current = make_synthetic_sar_data()
    frame0, frame1 = make_test_frames()

    change_map = detect_changes(
        frame0, frame1, baseline, current,
        Polarization.VV,
        pixel_resolution_m=10.0,
    )

    features = classify_anomalies(change_map)
    logger.info("  Features detected: %d", len(features))

    for f in features:
        logger.info("    %s | area=%.0f m² | AR=%.1f | conf=%.2f",
                     f.anomaly_type.value, f.area_m2, f.aspect_ratio, f.confidence)

    assert len(features) > 0, "No features detected"
    types_found = {f.anomaly_type for f in features}
    logger.info("  Types found: %s", [t.value for t in types_found])
    logger.info("TEST Module B: PASSED\n")
    return True


# ── Test: Module C ──────────────────────────────────────────────────

def test_module_c_threat_engine():
    logger.info("=" * 50)
    logger.info("TEST: Module C — Threat Scoring Engine")

    # Synthetic border line (a diagonal across a region)
    border_wkt = "LINESTRING(78.0 30.2, 78.5 30.2)"

    # Synthetic features near and far from border
    features = [
        DetectedFeature(
            feature_id="feat_0001",
            anomaly_type=AnomalyType.VEHICLE_TRACK,
            geometry_wkt="POLYGON((78.15 30.21, 78.16 30.21, 78.16 30.22, 78.15 30.22, 78.15 30.21))",
            centroid_lon=78.155, centroid_lat=30.215,
            area_m2=500, length_m=80, width_m=4.5, aspect_ratio=18.0,
            mean_backscatter_db=-13.0, confidence=0.75,
        ),
        DetectedFeature(
            feature_id="feat_0002",
            anomaly_type=AnomalyType.PERSONNEL_LINE,
            geometry_wkt="POLYGON((78.45 30.4, 78.46 30.4, 78.46 30.41, 78.45 30.41, 78.45 30.4))",
            centroid_lon=78.455, centroid_lat=30.405,
            area_m2=30, length_m=60, width_m=1.0, aspect_ratio=60.0,
            mean_backscatter_db=-16.0, confidence=0.80,
        ),
        DetectedFeature(
            feature_id="feat_0003",
            anomaly_type=AnomalyType.ENCAMPMENT_CLUSTER,
            geometry_wkt="POLYGON((78.25 30.22, 78.26 30.22, 78.26 30.23, 78.25 30.23, 78.25 30.22))",
            centroid_lon=78.255, centroid_lat=30.225,
            area_m2=400, length_m=20, width_m=18.0, aspect_ratio=1.1,
            mean_backscatter_db=-10.0, confidence=0.85,
        ),
    ]

    # Temporal positions for vehicle track
    temporal_positions = {
        "feat_0001": [
            (78.15, 30.20, 1000000),
            (78.155, 30.215, 100003600),  # 10 hours later, moving toward border
        ],
    }

    assessments = assess_threats(features, border_wkt, temporal_positions=temporal_positions)
    assert len(assessments) == 3

    for a in assessments:
        logger.info("  %s | score=%.2f | level=%s | dist=%.0fm",
                     a.anomaly_type.value, a.threat_score, a.threat_level.value, a.distance_to_border_m)

    # Vehicle near border should have highest score
    scores = {a.feature_id: a.threat_score for a in assessments}
    assert scores["feat_0001"] > scores["feat_0002"], "Near-border vehicle should outrank far personnel"

    # Encampment should be CRITICAL or HIGH
    encampment = [a for a in assessments if a.anomaly_type == AnomalyType.ENCAMPMENT_CLUSTER][0]
    assert encampment.threat_level in (ThreatLevel.CRITICAL, ThreatLevel.HIGH)

    logger.info("TEST Module C: PASSED\n")
    return True


# ── Test: Module D ──────────────────────────────────────────────────

def test_module_d_geo_export():
    logger.info("=" * 50)
    logger.info("TEST: Module D — GeoJSON Export")

    border_wkt = "LINESTRING(78.0 30.2, 78.5 30.2)"

    features = [
        DetectedFeature(
            feature_id="feat_0001",
            anomaly_type=AnomalyType.VEHICLE_TRACK,
            geometry_wkt="POLYGON((78.15 30.21, 78.16 30.21, 78.16 30.22, 78.15 30.22, 78.15 30.21))",
            centroid_lon=78.155, centroid_lat=30.215,
            area_m2=500, length_m=80, width_m=4.5, aspect_ratio=18.0,
            mean_backscatter_db=-13.0, confidence=0.75,
        ),
    ]

    assessments = assess_threats(features, border_wkt)
    geojson = build_geojson_collection(assessments, features)

    # Validate structure
    assert geojson["type"] == "FeatureCollection"
    assert len(geojson["features"]) == 1
    assert "metadata" in geojson
    assert "threat_summary" in geojson["metadata"]

    feat = geojson["features"][0]
    assert feat["type"] == "Feature"
    assert feat["geometry"]["type"] == "Polygon"
    assert "threat_score" in feat["properties"]
    assert "color" in feat["properties"]

    # Serialise to JSON string
    json_str = to_json_string(geojson)
    parsed = json.loads(json_str)
    assert parsed["type"] == "FeatureCollection"

    # Save to file
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_output.geojson")
    save_geojson(geojson, out_path)
    assert os.path.exists(out_path)

    # Cleanup
    os.remove(out_path)

    logger.info("  GeoJSON structure: valid")
    logger.info("  Serialization: OK")
    logger.info("  File I/O: OK")
    logger.info("TEST Module D: PASSED\n")
    return True


# ── Test: Full Pipeline ─────────────────────────────────────────────

def test_full_pipeline():
    logger.info("=" * 50)
    logger.info("TEST: Full Pipeline Integration")

    baseline, current = make_synthetic_sar_data()
    frame0, frame1 = make_test_frames()
    border_wkt = "LINESTRING(78.0 30.2, 78.5 30.2)"

    result = run_snowtrace(
        baseline_intensity=baseline,
        current_intensity=current,
        baseline_frame=frame0,
        current_frame=frame1,
        border_wkt=border_wkt,
        pixel_resolution_m=10.0,
    )

    assert len(result.features) > 0, "No features in pipeline result"
    assert len(result.assessments) > 0, "No assessments in pipeline result"
    assert result.geojson["type"] == "FeatureCollection"
    assert result.elapsed_seconds > 0

    logger.info("  Features: %d", len(result.features))
    logger.info("  Assessments: %d", len(result.assessments))
    logger.info("  Elapsed: %.3f seconds", result.elapsed_seconds)
    logger.info("TEST Full Pipeline: PASSED\n")
    return True


# ── Runner ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        test_module_a_sar_difference,
        test_module_b_feature_classifier,
        test_module_c_threat_engine,
        test_module_d_geo_export,
        test_full_pipeline,
    ]
    passed = 0
    failed = 0
    for test_fn in tests:
        try:
            if test_fn():
                passed += 1
        except Exception as e:
            logger.error("FAILED: %s — %s", test_fn.__name__, e)
            failed += 1

    logger.info("=" * 50)
    logger.info("RESULTS: %d passed, %d failed out of %d", passed, failed, len(tests))
    if failed:
        sys.exit(1)
