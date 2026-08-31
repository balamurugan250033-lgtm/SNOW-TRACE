"""
SnowTrace Geospatial Utilities
───────────────────────────────
Coordinate transforms, distance calculations, bearing computation,
and geometry helpers used across all SnowTrace modules.
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np


# ── Constants ────────────────────────────────────────────────────────

EARTH_RADIUS_M = 6_371_000.0  # mean Earth radius in metres
DEG_TO_RAD = math.pi / 180.0
RAD_TO_DEG = 180.0 / math.pi


# ── Haversine Distance ──────────────────────────────────────────────

def haversine_distance_m(
    lon1: float, lat1: float,
    lon2: float, lat2: float,
) -> float:
    """
    Compute great-circle distance between two WGS-84 points.

    Parameters
    ----------
    lon1, lat1 : float   – longitude / latitude of point A (degrees).
    lon2, lat2 : float   – longitude / latitude of point B (degrees).

    Returns
    -------
    float : distance in metres.
    """
    φ1 = lat1 * DEG_TO_RAD
    φ2 = lat2 * DEG_TO_RAD
    Δφ = (lat2 - lat1) * DEG_TO_RAD
    Δλ = (lon2 - lon1) * DEG_TO_RAD

    a = math.sin(Δφ / 2) ** 2 + math.cos(φ1) * math.cos(φ2) * math.sin(Δλ / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return EARTH_RADIUS_M * c


# ── Bearing ──────────────────────────────────────────────────────────

def initial_bearing_deg(
    lon1: float, lat1: float,
    lon2: float, lat2: float,
) -> float:
    """
    Compute initial compass bearing (0–360°) from A → B.

    Uses the forward azimuth formula on a sphere.
    """
    φ1 = lat1 * DEG_TO_RAD
    φ2 = lat2 * DEG_TO_RAD
    Δλ = (lon2 - lon1) * DEG_TO_RAD

    y = math.sin(Δλ) * math.cos(φ2)
    x = math.cos(φ1) * math.sin(φ2) - math.sin(φ1) * math.cos(φ2) * math.cos(Δλ)
    θ = math.atan2(y, x) * RAD_TO_DEG
    return (θ + 360) % 360


# ── Point-to-Line Distance ──────────────────────────────────────────

def point_to_linestring_distance_m(
    px: float, py: float,
    line_coords: list[Tuple[float, float]],
) -> float:
    """
    Minimum distance from a point (px, py) to a polyline in WGS-84.

    Projects to a local tangent plane (UTM-like approximation) for each
    segment, then returns the minimum segment distance.

    Parameters
    ----------
    px, py         : point longitude, latitude.
    line_coords    : list of (lon, lat) tuples defining the polyline.

    Returns
    -------
    float : minimum distance in metres.
    """
    if len(line_coords) < 2:
        return float("inf")

    min_dist = float("inf")
    for i in range(len(line_coords) - 1):
        x1, y1 = line_coords[i]
        x2, y2 = line_coords[i + 1]
        # Project to metres using simple equirectangular approximation
        mid_lat = (y1 + y2) / 2.0 * DEG_TO_RAD
        # Convert to approximate metres
        ax, ay = (x1 - px) * math.cos(mid_lat) * EARTH_RADIUS_M * DEG_TO_RAD, (y1 - py) * EARTH_RADIUS_M * DEG_TO_RAD
        bx, by = (x2 - px) * math.cos(mid_lat) * EARTH_RADIUS_M * DEG_TO_RAD, (y2 - py) * EARTH_RADIUS_M * DEG_TO_RAD

        seg_len_sq = bx ** 2 + by ** 2
        if seg_len_sq < 1e-12:
            dist = math.sqrt(ax ** 2 + ay ** 2)
        else:
            t = max(0.0, min(1.0, (ax * bx + ay * by) / seg_len_sq))
            proj_x = ax - t * bx
            proj_y = ay - t * by
            dist = math.sqrt(proj_x ** 2 + proj_y ** 2)

        min_dist = min(min_dist, dist)

    return min_dist


# ── Pixel → Geo-coordinate ──────────────────────────────────────────

def pixel_to_geo(
    row: int, col: int,
    transform: object,
) -> Tuple[float, float]:
    """
    Convert raster row/col to geographic (lon, lat) using an affine transform.

    Parameters
    ----------
    row, col   : integer pixel indices.
    transform  : affine.Affine or 6-tuple (a, b, c, d, e, f).

    Returns
    -------
    (lon, lat) in the CRS of the raster.
    """
    # Handle both Affine objects and raw tuples
    try:
        x = transform.c + col * transform.a + row * transform.b
        y = transform.f + col * transform.d + row * transform.e
    except AttributeError:
        a, b, c, d, e, f = transform[:6]
        x = c + col * a + row * b
        y = f + col * d + row * e
    return (x, y)


# ── Geo-coordinate → Pixel ──────────────────────────────────────────

def geo_to_pixel(
    x: float, y: float,
    transform: object,
) -> Tuple[int, int]:
    """
    Inverse of pixel_to_geo.  Returns (row, col).
    """
    try:
        inv = ~transform
        col_f = inv.c + x * inv.a + y * inv.b
        row_f = inv.f + x * inv.d + y * inv.e
    except AttributeError:
        a, b, c, d, e, f = transform[:6]
        det = a * e - b * d
        inv_a, inv_b, inv_d, inv_e = e / det, -b / det, -d / det, a / det
        col_f = inv_a * (x - c) + inv_b * (y - f)
        row_f = inv_d * (x - c) + inv_e * (y - f)
    return int(round(row_f)), int(round(col_f))


# ── dB Conversion ────────────────────────────────────────────────────

def linear_to_db(value: float, epsilon: float = 1e-10) -> float:
    """Convert linear power ratio to decibels."""
    return 10.0 * math.log10(max(value, epsilon))


def db_to_linear(db_value: float) -> float:
    """Convert decibels to linear power ratio."""
    return 10.0 ** (db_value / 10.0)


# ── Geodesic Midpoint ───────────────────────────────────────────────

def geodesic_midpoint(
    lon1: float, lat1: float,
    lon2: float, lat2: float,
) -> Tuple[float, float]:
    """Compute the geographic midpoint between two WGS-84 points."""
    φ1, λ1 = lat1 * DEG_TO_RAD, lon1 * DEG_TO_RAD
    φ2, λ2 = lat2 * DEG_TO_RAD, lon2 * DEG_TO_RAD

    Bx = math.cos(φ2) * math.cos(λ2 - λ1)
    By = math.cos(φ2) * math.sin(λ2 - λ1)
    φm = math.atan2(math.sin(φ1) + math.sin(φ2),
                     math.sqrt((math.cos(φ1) + Bx) ** 2 + By ** 2))
    λm = λ1 + math.atan2(By, math.cos(φ1) + Bx)

    return (λm * RAD_TO_DEG, φm * RAD_TO_DEG)


# ── Polyline Centroid ───────────────────────────────────────────────

def polyline_centroid(coords: list[Tuple[float, float]]) -> Tuple[float, float]:
    """Simple centroid of a list of (lon, lat) coordinates."""
    if not coords:
        return (0.0, 0.0)
    lons = [c[0] for c in coords]
    lats = [c[1] for c in coords]
    return (sum(lons) / len(lons), sum(lats) / len(lats))


# ── Polygon Centroid (Shapely-like without dependency) ───────────────

def polygon_centroid(coords: list[Tuple[float, float]]) -> Tuple[float, float]:
    """
    Compute the centroid of a simple polygon defined by (lon, lat) vertices.

    Uses the shoelace-based centroid formula.
    """
    n = len(coords)
    if n < 3:
        return polyline_centroid(coords)

    area = 0.0
    cx = 0.0
    cy = 0.0
    for i in range(n):
        x0, y0 = coords[i]
        x1, y1 = coords[(i + 1) % n]
        cross = x0 * y1 - x1 * y0
        area += cross
        cx += (x0 + x1) * cross
        cy += (y0 + y1) * cross

    area *= 0.5
    if abs(area) < 1e-14:
        return polyline_centroid(coords)

    cx /= (6.0 * area)
    cy /= (6.0 * area)
    return (cx, cy)


# ── Bounding Box ─────────────────────────────────────────────────────

def bounding_box(coords: list[Tuple[float, float]]) -> Tuple[float, float, float, float]:
    """
    Return (min_lon, min_lat, max_lon, max_lat) for a set of coordinates.
    """
    lons = [c[0] for c in coords]
    lats = [c[1] for c in coords]
    return (min(lons), min(lats), max(lons), max(lats))
