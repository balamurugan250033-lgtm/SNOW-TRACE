"""
SnowTrace Local Server
──────────────────────
Fetches real Sentinel-1 SAR data, runs the pipeline, and serves
a Leaflet map on localhost.

Usage:
    python server/serve.py
"""

import json
import os
import sys
import threading
import webbrowser
from datetime import datetime
from http.server import HTTPServer, SimpleHTTPRequestHandler

import numpy as np
from affine import Affine

# Project root
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from snowtrace.config import SARConfig, ChangeDetectionConfig, FeatureClassificationConfig, ThreatScoringConfig
from snowtrace.models.data_models import Polarization, SARFrame
from snowtrace.pipeline import run_snowtrace

import ee
import geemap

PORT = 8765
SERVER_DIR = os.path.dirname(os.path.abspath(__file__))


def fetch_sentinel_data():
    """Fetch two real Sentinel-1 acquisitions and convert to numpy arrays."""
    ee.Initialize(project='academic-oath-507004-i5')
    aoi = ee.Geometry.Rectangle([76.8, 35.0, 77.4, 35.4])
    collection = ee.ImageCollection('COPERNICUS/S1_GRD') \
        .filterBounds(aoi) \
        .filter(ee.Filter.eq('instrumentMode', 'IW')) \
        .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV')) \
        .select('VV')
    img_t0 = collection.filterDate('2026-08-01', '2026-08-06').mosaic().clip(aoi)
    img_t1 = collection.filterDate('2026-08-13', '2026-08-18').mosaic().clip(aoi)

    def to_linear(img):
        return ee.Image(10).pow(img.divide(10))

    baseline_arr = geemap.ee_to_numpy(to_linear(img_t0), region=aoi, scale=30)
    current_arr = geemap.ee_to_numpy(to_linear(img_t1), region=aoi, scale=30)
    return baseline_arr[:, :, 0], current_arr[:, :, 0]


def generate_synthetic_data():
    """Fetch real SAR scene from Sentinel-1 and run the pipeline."""
    baseline, current = fetch_sentinel_data()
    rows, cols = baseline.shape

    # Frame metadata
    t0 = SARFrame(
        acquisition_id="S1_GRD_20260801",
        acquisition_time=datetime(2026, 8, 1, 5, 30),
        file_path="sentinel1_t0",
        polarization=Polarization.VV,
        orbit_direction="ASCENDING",
    )
    t1 = SARFrame(
        acquisition_id="S1_GRD_20260813",
        acquisition_time=datetime(2026, 8, 13, 5, 30),
        file_path="sentinel1_t1",
        polarization=Polarization.VV,
        orbit_direction="ASCENDING",
    )

    # Border line across the full width
    border_wkt = "LINESTRING(76.8 35.2, 77.4 35.2)"

    # Affine transform: map pixel (row, col) → (lon, lat)
    lon_per_px = (77.4 - 76.8) / cols
    lat_per_px = (35.4 - 35.0) / rows
    transform = Affine(
        lon_per_px, 0, 76.8,
        0, -lat_per_px, 35.4,
    )

    result = run_snowtrace(
        baseline_intensity=baseline,
        current_intensity=current,
        baseline_frame=t0,
        current_frame=t1,
        border_wkt=border_wkt,
        pixel_resolution_m=30.0,
        transform=transform,
        output_path=os.path.join(SERVER_DIR, "threats.geojson"),
    )

    return result


class SnowTraceHandler(SimpleHTTPRequestHandler):
    """Serve files from the server directory."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=SERVER_DIR, **kwargs)

    def log_message(self, fmt, *args):
        print(f"  {args[0]}")


def main():
    print("=" * 55)
    print("  SnowTrace — Local Threat Monitor")
    print("=" * 55)

    # Step 1: Generate data
    print("\n[1/2] Fetching real Sentinel-1 SAR scene...")
    result = generate_synthetic_data()
    print(f"      Detected {len(result.features)} features, "
          f"{len(result.assessments)} assessments")
    print(f"      Max threat score: {max(a.threat_score for a in result.assessments):.2f}")
    print(f"      Output: {os.path.join(SERVER_DIR, 'threats.geojson')}")

    # Step 2: Start server
    print(f"\n[2/2] Starting server on http://localhost:{PORT}")
    server = HTTPServer(("127.0.0.1", PORT), SnowTraceHandler)

    threading.Timer(0.5, lambda: webbrowser.open(f"http://localhost:{PORT}")).start()

    print(f"      Serving: {SERVER_DIR}")
    print("      Press Ctrl+C to stop\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")
        server.server_close()


if __name__ == "__main__":
    main()