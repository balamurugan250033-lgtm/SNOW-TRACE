"""
SnowTrace Module A: SAR Difference Engine
──────────────────────────────────────────
Implements multi-temporal SAR change detection for Sentinel-1 GRD/SLC
data.  Computes log-ratio intensity differencing and coherence
estimation to isolate snow-surface disturbances.

Processing chain:
  1. Load co-registered SAR frames (VV / VH).
  2. Apply Lee speckle filter.
  3. Compute log-ratio:  Δ = 10·log10(σ_T1 / σ_T0).
  4. Estimate interferometric coherence (if SLC pair available).
  5. Threshold → binary change mask.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np
from scipy import ndimage

from snowtrace.config import ChangeDetectionConfig, SARConfig
from snowtrace.models.data_models import ChangeMap, Polarization, SARFrame
from snowtrace.utils.geo_utils import linear_to_db

logger = logging.getLogger(__name__)


# ── Lee Speckle Filter ──────────────────────────────────────────────

def lee_filter(
    image: np.ndarray,
    kernel_size: int = 7,
    noise_variance: float = 0.04,
) -> np.ndarray:
    """
    Apply Lee's enhanced speckle filter to a SAR intensity image.

    The filter preserves edges while smoothing homogeneous regions.
    It uses local statistics (mean, variance) to compute an
    adaptive weighting factor.

    Parameters
    ----------
    image          : 2-D float array (linear intensity, not dB).
    kernel_size    : filter window dimension (must be odd).
    noise_variance : expected speckle noise variance (configurable).

    Returns
    -------
    np.ndarray : filtered image, same shape and dtype.
    """
    if kernel_size % 2 == 0:
        kernel_size += 1

    pad = kernel_size // 2
    rows, cols = image.shape
    output = np.zeros_like(image, dtype=np.float64)

    # Pre-compute local statistics using uniform filter for speed
    # This avoids double loops and is ~10x faster per pixel
    weights = np.ones((kernel_size, kernel_size), dtype=np.float64)
    local_sum = ndimage.uniform_filter(image.astype(np.float64), size=kernel_size, mode="reflect")
    local_sq_sum = ndimage.uniform_filter(image.astype(np.float64) ** 2, size=kernel_size, mode="reflect")

    n = kernel_size * kernel_size
    local_mean = local_sum
    local_var = np.maximum(local_sq_sum - local_mean ** 2, 0.0)

    # Lee filter weight:  W = max(0, 1 - (noise_var / local_var))
    W = np.where(local_var > noise_variance,
                 1.0 - noise_variance / local_var,
                 0.0)
    W = np.clip(W, 0.0, 1.0)

    output = local_mean + W * (image - local_mean)
    return output.astype(image.dtype)


# ── Log-Ratio Change Detection ──────────────────────────────────────

def compute_log_ratio(
    baseline: np.ndarray,
    current: np.ndarray,
    epsilon: float = 1e-10,
) -> np.ndarray:
    """
    Compute the log-ratio change map between two SAR intensity images.

        ΔdB = 10 · log10( (σ_current + ε) / (σ_baseline + ε) )

    Positive values indicate increased backscatter (e.g. surface
    roughening from footsteps/vehicles).  Negative values indicate
    decreased backscatter (e.g. snow smoothing or shadowing).

    Parameters
    ----------
    baseline  : 2-D array of baseline SAR intensity (linear scale).
    current   : 2-D array of current SAR intensity (linear scale).
    epsilon   : small constant to avoid log(0).

    Returns
    -------
    np.ndarray : log-ratio in dB, same spatial dimensions.
    """
    baseline_f = baseline.astype(np.float64)
    current_f = current.astype(np.float64)

    ratio = (current_f + epsilon) / (baseline_f + epsilon)
    log_ratio = 10.0 * np.log10(np.maximum(ratio, epsilon))

    return log_ratio


# ── Coherence Estimation ────────────────────────────────────────────

def estimate_coherence(
    reference: np.ndarray,
    secondary: np.ndarray,
    window_size: int = 15,
    azimuth_window: int = 10,
    range_window: int = 10,
) -> np.ndarray:
    """
    Estimate interferometric coherence between two co-registered
    complex SAR images using a sliding window estimator.

    For GRD (amplitude-only) data this is a simplified proxy that
    computes amplitude correlation.  For SLC data, replace with
    complex cross-correlation.

    Coherence γ ∈ [0, 1]:
        γ = |Σ s1·s2*| / √(Σ|s1|² · Σ|s2|²)

    A drop in coherence indicates surface change (e.g. snow
    disturbance breaking phase stability).

    Parameters
    ----------
    reference      : 2-D array (complex if SLC, float if GRD).
    secondary      : 2-D array (same shape and type as reference).
    window_size    : integration window in range direction.
    azimuth_window : integration window in azimuth direction.

    Returns
    -------
    np.ndarray : coherence map, values in [0, 1].
    """
    ref = reference.astype(np.complex128) if np.iscomplexobj(reference) else reference.astype(np.float64)
    sec = secondary.astype(np.complex128) if np.iscomplexobj(secondary) else secondary.astype(np.float64)

    # Cross-correlation
    cross = ref * np.conj(sec)
    power_ref = np.abs(ref) ** 2
    power_sec = np.abs(sec) ** 2

    # Boxcar averaging (windowed mean)
    kernel = np.ones((azimuth_window, range_window), dtype=np.float64) / (azimuth_window * range_window)

    # Complex cross-correlation magnitude
    cross_avg = ndimage.convolve(cross, kernel, mode="reflect")
    ref_avg = ndimage.convolve(power_ref, kernel, mode="reflect")
    sec_avg = ndimage.convolve(power_sec, kernel, mode="reflect")

    # Coherence with stability epsilon
    denom = np.sqrt(ref_avg * sec_avg)
    denom = np.where(denom < 1e-12, 1e-12, denom)

    coh = np.abs(cross_avg) / denom
    return np.clip(coh, 0.0, 1.0)


# ── Binary Change Mask ──────────────────────────────────────────────

def threshold_change_mask(
    log_ratio_db: np.ndarray,
    coherence: Optional[np.ndarray],
    config: ChangeDetectionConfig,
) -> np.ndarray:
    """
    Generate a binary change mask from log-ratio and/or coherence.

    A pixel is flagged as "changed" if:
      - |log_ratio| > log_ratio_threshold_db,  OR
      - coherence drops below coherence_drop_threshold.

    Parameters
    ----------
    log_ratio_db  : log-ratio change map (dB).
    coherence     : optional coherence map.
    config        : change detection configuration.

    Returns
    -------
    np.ndarray : boolean mask, True = changed pixel.
    """
    mask = np.abs(log_ratio_db) > config.log_ratio_threshold_db

    if coherence is not None:
        coh_drop = coherence < (1.0 - config.coherence_drop_threshold)
        mask = mask | coh_drop

    return mask


# ── Morphological Cleanup ───────────────────────────────────────────

def morphological_cleanup(
    change_mask: np.ndarray,
    open_kernel: int = 3,
    close_kernel: int = 5,
    min_area_px: int = 10,
) -> np.ndarray:
    """
    Apply morphological opening → closing → area filtering to
    remove salt-and-pepper noise and tiny isolated clusters.

    Parameters
    ----------
    change_mask   : boolean array.
    open_kernel   : structuring element size for opening.
    close_kernel  : structuring element size for closing.
    min_area_px   : minimum connected-component size in pixels.

    Returns
    -------
    np.ndarray : cleaned boolean change mask.
    """
    struct_open = ndimage.generate_binary_structure(2, 1)
    struct_close = ndimage.generate_binary_structure(2, 1)

    # Opening: remove small bright spots
    opened = ndimage.binary_opening(change_mask, structure=struct_open, iterations=open_kernel // 2)

    # Closing: fill small dark holes
    closed = ndimage.binary_closing(opened, structure=struct_close, iterations=close_kernel // 2)

    # Remove small connected components
    labeled, num_features = ndimage.label(closed)
    cleaned = np.zeros_like(closed)
    for i in range(1, num_features + 1):
        component = labeled == i
        if np.sum(component) >= min_area_px:
            cleaned |= component

    return cleaned


# ── Main Change Detection Pipeline ──────────────────────────────────

def detect_changes(
    baseline_frame: SARFrame,
    current_frame: SARFrame,
    baseline_intensity: np.ndarray,
    current_intensity: np.ndarray,
    polarization: Polarization,
    sar_config: Optional[SARConfig] = None,
    change_config: Optional[ChangeDetectionConfig] = None,
    coherence: Optional[np.ndarray] = None,
    pixel_resolution_m: float = 10.0,
    transform: Optional[object] = None,
    crs: Optional[str] = None,
) -> ChangeMap:
    """
    End-to-end change detection between two SAR frames.

    Parameters
    ----------
    baseline_frame      : metadata of the baseline SAR frame.
    current_frame       : metadata of the current SAR frame.
    baseline_intensity  : 2-D numpy array of baseline intensity.
    current_intensity   : 2-D numpy array of current intensity.
    polarization        : which polarization channel is being processed.
    sar_config          : SAR processing parameters (defaults applied).
    change_config       : change detection thresholds.
    coherence           : optional pre-computed coherence map.
    pixel_resolution_m  : ground pixel spacing in metres.
    transform           : affine transform for geolocating pixels.
    crs                 : coordinate reference system string.

    Returns
    -------
    ChangeMap : populated change map object.
    """
    cfg_sar = sar_config or SARConfig()
    cfg_chg = change_config or ChangeDetectionConfig()

    logger.info("Applying Lee speckle filter (kernel=%d)", cfg_sar.speckle_filter_size)
    baseline_filtered = lee_filter(baseline_intensity, cfg_sar.speckle_filter_size)
    current_filtered = lee_filter(current_intensity, cfg_sar.speckle_filter_size)

    logger.info("Computing log-ratio change map")
    log_ratio = compute_log_ratio(baseline_filtered, current_filtered, cfg_chg.epsilon)

    # Coherence estimation if SLC data or pre-computed coherence provided
    if coherence is None:
        logger.info("Estimating amplitude coherence (proxy)")
        coherence = estimate_coherence(
            baseline_filtered, current_filtered,
            window_size=cfg_sar.coherence_window_size,
            azimuth_window=cfg_sar.coherence_azimuth_window,
            range_window=cfg_sar.coherence_range_window,
        )
    else:
        logger.info("Using provided coherence map")

    logger.info("Thresholding change mask (±%.1f dB)", cfg_chg.log_ratio_threshold_db)
    raw_mask = threshold_change_mask(log_ratio, coherence, cfg_chg)

    logger.info("Morphological cleanup (min area=%d px)", cfg_chg.min_valid_pixel_pct * 100)
    cleaned_mask = morphological_cleanup(
        raw_mask,
        open_kernel=3,
        close_kernel=5,
        min_area_px=10,
    )

    change_pct = 100.0 * np.sum(cleaned_mask) / cleaned_mask.size
    logger.info("Change coverage: %.2f%% of scene", change_pct)

    return ChangeMap(
        change_map_id="",
        baseline_frame=baseline_frame,
        current_frame=current_frame,
        polarization=polarization,
        log_ratio_array=log_ratio,
        coherence_array=coherence,
        change_mask=cleaned_mask,
        transform=transform,
        crs=crs or "EPSG:4326",
        pixel_resolution_m=pixel_resolution_m,
    )
