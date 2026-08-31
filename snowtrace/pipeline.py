"""
SnowTrace Pipeline Orchestrator
───────────────────────────────
Ties together all four modules into a single end-to-end processing
pipeline:

    SAR Frames → Change Detection → Feature Classification
        → Threat Assessment → GeoJSON Export

Usage:
    from snowtrace.pipeline import SnowTracePipeline

    pipeline = SnowTracePipeline(border_wkt="LINESTRING(...)")
    result = pipeline.run(
        baseline_intensity=arr_t0,
        current_intensity=arr_t1,
        baseline_frame=frame_t0,
        current_frame=frame_t1,
    )
    # result is a GeoJSON FeatureCollection dict
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from snowtrace.config import (
    ChangeDetectionConfig,
    ExportConfig,
    FeatureClassificationConfig,
    SARConfig,
    ThreatScoringConfig,
)
from snowtrace.core.feature_classifier import classify_anomalies
from snowtrace.core.geo_export import build_geojson_collection, save_geojson
from snowtrace.core.sar_difference import detect_changes
from snowtrace.core.threat_engine import assess_threats
from snowtrace.models.data_models import (
    DetectedFeature,
    Polarization,
    SARFrame,
    ThreatAssessment,
)

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """Container for the full pipeline output."""

    geojson: Dict[str, Any]
    assessments: List[ThreatAssessment]
    features: List[DetectedFeature]
    elapsed_seconds: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class SnowTracePipeline:
    """
    End-to-end SnowTrace processing pipeline.

    Parameters
    ----------
    border_wkt           : WKT LINESTRING defining the border to monitor.
    sar_config           : SAR preprocessing parameters.
    change_config        : change detection thresholds.
    classification_config: anomaly classification parameters.
    threat_config        : threat scoring weights.
    export_config        : GeoJSON export settings.
    """

    def __init__(
        self,
        border_wkt: str,
        sar_config: Optional[SARConfig] = None,
        change_config: Optional[ChangeDetectionConfig] = None,
        classification_config: Optional[FeatureClassificationConfig] = None,
        threat_config: Optional[ThreatScoringConfig] = None,
        export_config: Optional[ExportConfig] = None,
    ) -> None:
        self.border_wkt = border_wkt
        self.sar_config = sar_config or SARConfig()
        self.change_config = change_config or ChangeDetectionConfig()
        self.classification_config = classification_config or FeatureClassificationConfig()
        self.threat_config = threat_config or ThreatScoringConfig()
        self.export_config = export_config or ExportConfig()

    def run(
        self,
        baseline_intensity: np.ndarray,
        current_intensity: np.ndarray,
        baseline_frame: SARFrame,
        current_frame: SARFrame,
        polarization: Polarization = Polarization.VV,
        pixel_resolution_m: float = 10.0,
        transform: Optional[object] = None,
        crs: Optional[str] = None,
        temporal_positions: Optional[Dict[str, list]] = None,
        output_path: Optional[str] = None,
    ) -> PipelineResult:
        """
        Execute the full SnowTrace pipeline.

        Parameters
        ----------
        baseline_intensity  : 2-D numpy array of baseline SAR intensity.
        current_intensity   : 2-D numpy array of current SAR intensity.
        baseline_frame      : metadata for baseline frame.
        current_frame       : metadata for current frame.
        polarization        : SAR polarization channel.
        pixel_resolution_m  : ground pixel spacing.
        transform           : affine geotransform.
        crs                 : coordinate reference system.
        temporal_positions  : historical positions per feature for heading.
        output_path         : optional file path to write GeoJSON.

        Returns
        -------
        PipelineResult
        """
        t0 = time.time()
        logger.info("=" * 60)
        logger.info("SnowTrace Pipeline START")
        logger.info("  Baseline: %s | Current: %s", baseline_frame.acquisition_id, current_frame.acquisition_id)
        logger.info("  Polarization: %s | Pixel: %.1f m", polarization.value, pixel_resolution_m)
        logger.info("=" * 60)

        # ── Step 1: SAR Change Detection ────────────────────────
        logger.info("[Step 1/4] SAR Change Detection")
        change_map = detect_changes(
            baseline_frame=baseline_frame,
            current_frame=current_frame,
            baseline_intensity=baseline_intensity,
            current_intensity=current_intensity,
            polarization=polarization,
            sar_config=self.sar_config,
            change_config=self.change_config,
            pixel_resolution_m=pixel_resolution_m,
            transform=transform,
            crs=crs,
        )

        # ── Step 2: Feature Classification ──────────────────────
        logger.info("[Step 2/4] Feature Classification")
        features = classify_anomalies(change_map, self.classification_config)

        # ── Step 3: Threat Assessment ───────────────────────────
        logger.info("[Step 3/4] Threat Assessment")
        assessments = assess_threats(
            features=features,
            border_line_wkt=self.border_wkt,
            config=self.threat_config,
            temporal_positions=temporal_positions,
        )

        # ── Step 4: GeoJSON Export ──────────────────────────────
        logger.info("[Step 4/4] GeoJSON Export")
        geojson = build_geojson_collection(
            assessments=assessments,
            features=features,
            export_config=self.export_config,
            metadata={
                "baseline_frame": baseline_frame.acquisition_id,
                "current_frame": current_frame.acquisition_id,
                "processing_polarization": polarization.value,
                "pixel_resolution_m": pixel_resolution_m,
            },
        )

        # Optionally persist to disk
        if output_path:
            save_geojson(geojson, output_path)

        elapsed = time.time() - t0
        logger.info("=" * 60)
        logger.info("SnowTrace Pipeline COMPLETE (%.2f seconds)", elapsed)
        logger.info("  Features detected: %d", len(features))
        logger.info("  Assessments generated: %d", len(assessments))
        if assessments:
            logger.info("  Max threat score: %.2f", max(a.threat_score for a in assessments))
        logger.info("=" * 60)

        return PipelineResult(
            geojson=geojson,
            assessments=assessments,
            features=features,
            elapsed_seconds=elapsed,
            metadata={
                "baseline_id": baseline_frame.acquisition_id,
                "current_id": current_frame.acquisition_id,
            },
        )


# ── Convenience Function ────────────────────────────────────────────

def run_snowtrace(
    baseline_intensity: np.ndarray,
    current_intensity: np.ndarray,
    baseline_frame: SARFrame,
    current_frame: SARFrame,
    border_wkt: str,
    polarization: Polarization = Polarization.VV,
    pixel_resolution_m: float = 10.0,
    transform: Optional[object] = None,
    crs: Optional[str] = None,
    output_path: Optional[str] = None,
) -> PipelineResult:
    """
    One-call convenience wrapper around SnowTracePipeline.
    """
    pipeline = SnowTracePipeline(border_wkt=border_wkt)
    return pipeline.run(
        baseline_intensity=baseline_intensity,
        current_intensity=current_intensity,
        baseline_frame=baseline_frame,
        current_frame=current_frame,
        polarization=polarization,
        pixel_resolution_m=pixel_resolution_m,
        transform=transform,
        crs=crs,
        output_path=output_path,
    )
