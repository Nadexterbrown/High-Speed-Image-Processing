"""
Process Photron high-speed video files using the photron library.

This script demonstrates loading MRAW/CIHX video files and processing them
with support for timing, calibration, and MPI parallelization.

Usage:
    Serial:   python scripts/process_videos.py
    Parallel: mpiexec -n 4 python scripts/process_videos.py
"""

import os
import sys
import csv
import re
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src import (
    open_video,
    open_collection,
    PhotonVideo,
    VideoCollection,
    SpatialCalibration,
    MPIVideoProcessor,
)

# Optional MPI import
try:
    from mpi4py import MPI
    HAS_MPI = True
except ImportError:
    HAS_MPI = False
    MPI = None


########################################################################################################################
# Configuration
########################################################################################################################

@dataclass
class FileCalibration:
    """
    Calibration settings for specific files or file ranges.

    Use 'files' to specify which files this calibration applies to.
    Supports:
    - Exact filenames: "Run-001.cihx"
    - Partial matches: "Run-001" (matches any file containing this string)
    - Range patterns: "Run-001:Run-005" (matches Run-001 through Run-005)

    Example:
        # Runs 1-3 with calibration A
        FileCalibration(calibration=0.00074, position_offset=0.0, files=["Run-001:Run-003"])

        # Runs 4-10 with calibration B and offset
        FileCalibration(calibration=0.00080, position_offset=0.05, files=["Run-004:Run-010"])
    """
    calibration: float  # meters per pixel
    position_offset: float = 0.0  # position offset in meters (added to detected position)
    files: List[str] = field(default_factory=list)  # file patterns, names, or ranges

    def matches(self, filename: str) -> bool:
        """Check if this calibration applies to the given filename."""
        for pattern in self.files:
            # Check for range pattern (e.g., "Run-001:Run-005")
            if ':' in pattern:
                start, end = pattern.split(':', 1)
                if self._matches_range(filename, start.strip(), end.strip()):
                    return True
            # Check for exact or partial match
            elif pattern in filename:
                return True
        return False

    def _matches_range(self, filename: str, start: str, end: str) -> bool:
        """Check if filename falls within a range pattern."""
        # Extract numeric portions for comparison
        start_nums = re.findall(r'\d+', start)
        end_nums = re.findall(r'\d+', end)
        file_nums = re.findall(r'\d+', filename)

        if not start_nums or not end_nums or not file_nums:
            return False

        # Use the last number found (typically the run number)
        try:
            start_num = int(start_nums[-1])
            end_num = int(end_nums[-1])
            file_num = int(file_nums[-1])
            return start_num <= file_num <= end_num
        except ValueError:
            return False


@dataclass
class VideoSourceConfig:
    """Configuration for a video source."""
    name: str
    enabled: bool = False
    calibration: float = 1.0  # meters per pixel (default, used if no file-specific calibration)
    position_offset: float = 0.0  # position offset in meters (default)
    trigger_frame: Optional[int] = None  # Frame index where trigger occurred. None = use metadata/absolute time
    use_frame_diff: bool = True  # Use prior frame subtraction for flame isolation
    detection_method: str = "half_maximum"  # "threshold", "gradient", or "half_maximum"
    use_absolute_time: bool = True  # Use absolute time from recording start (not trigger-relative)
    skip_frames: List[int] = field(default_factory=list)  # Frames to skip (no centerline detection)
    file_calibrations: List[FileCalibration] = field(default_factory=list)  # Per-file calibration settings

    _video_path: Optional[str] = field(default=None, init=False, repr=False)
    _output_dir: Optional[str] = field(default=None, init=False, repr=False)

    @property
    def video_path(self) -> Optional[str]:
        return self._video_path

    @video_path.setter
    def video_path(self, path: Optional[str]):
        self._video_path = self._resolve_path(path)

    @property
    def output_dir(self) -> Optional[str]:
        return self._output_dir

    @output_dir.setter
    def output_dir(self, path: Optional[str]):
        self._output_dir = self._resolve_path(path)

    def _resolve_path(self, path: Optional[str]) -> Optional[str]:
        if path is None:
            return None
        elif os.path.isabs(path):
            return path
        else:
            base_path = Path(__file__).parent.parent
            return str((base_path / path).resolve())

    def get_calibration_for_file(self, filename: str) -> Tuple[float, float]:
        """
        Get calibration and position offset for a specific file.

        Checks file_calibrations list for matching patterns/ranges.
        Falls back to default calibration and position_offset if no match.

        Args:
            filename: Name of the video file (e.g., "Run-005.cihx")

        Returns:
            Tuple of (calibration, position_offset) in meters
        """
        for fc in self.file_calibrations:
            if fc.matches(filename):
                return (fc.calibration, fc.position_offset)
        return (self.calibration, self.position_offset)


########################################################################################################################
# Processing Functions
########################################################################################################################

def subtract_scalar_background(image: np.ndarray, background_scalar: float) -> np.ndarray:
    """Subtract scalar background; set negative values to zero."""
    subtracted = image.astype(np.float64) - background_scalar
    subtracted[subtracted < 0] = 0
    return subtracted


def subtract_prior_frame(
    current_frame: np.ndarray,
    prior_frame: np.ndarray,
    threshold: float = 0.0
) -> np.ndarray:
    """
    Subtract prior frame from current frame to isolate moving flame front.

    This technique highlights the flame front motion between frames,
    which is useful for tracking the leading edge of a detonation wave.

    Args:
        current_frame: Current frame data
        prior_frame: Previous frame data
        threshold: Minimum difference to keep (noise reduction)

    Returns:
        Difference image showing motion between frames
    """
    current = current_frame.astype(np.float64)
    prior = prior_frame.astype(np.float64)

    diff = current - prior
    diff[diff < threshold] = 0
    return diff


def three_frame_difference(
    frame_prev: np.ndarray,
    frame_curr: np.ndarray,
    frame_next: np.ndarray,
    threshold: float = 0.0
) -> np.ndarray:
    """
    Three-frame differencing for robust motion isolation.

    This technique isolates pixels that changed in BOTH transitions:
    - From frame_prev to frame_curr
    - From frame_curr to frame_next

    This reduces noise and isolates the actual moving object (flame front).

    Args:
        frame_prev: Frame at time t-1
        frame_curr: Frame at time t (the one being analyzed)
        frame_next: Frame at time t+1
        threshold: Minimum difference to keep

    Returns:
        Motion-isolated image for frame_curr
    """
    prev = frame_prev.astype(np.float64)
    curr = frame_curr.astype(np.float64)
    next_f = frame_next.astype(np.float64)

    # Compute differences
    diff1 = np.abs(curr - prev)
    diff2 = np.abs(next_f - curr)

    # Take minimum (pixel must have changed in both transitions)
    motion = np.minimum(diff1, diff2)
    motion[motion < threshold] = 0

    return motion


def is_empty_frame(
    frame: np.ndarray,
    noise_threshold: float = 50.0,
    min_signal_fraction: float = 0.001
) -> bool:
    """
    Check if a frame is empty (only contains noise).

    Args:
        frame: Frame data as numpy array
        noise_threshold: Pixel values below this are considered noise
        min_signal_fraction: Minimum fraction of pixels that must be above threshold

    Returns:
        True if frame is considered empty/noise-only
    """
    above_noise = np.sum(frame > noise_threshold)
    total_pixels = frame.size
    signal_fraction = above_noise / total_pixels

    return signal_fraction < min_signal_fraction


def detect_rightmost_flame_edge(
    row: np.ndarray,
    threshold: float = 5.0,
    min_region_size: int = 3
) -> Optional[int]:
    """
    Detect rightmost flame edge by finding the rightmost edge of the main flame region.

    Uses adaptive thresholding and contiguous region detection to avoid
    detecting noise at image edges.

    Args:
        row: 1D array of pixel intensities along centerline
        threshold: Minimum absolute intensity threshold for detection
        min_region_size: Minimum number of consecutive pixels to be considered a valid region

    Returns:
        Pixel index of rightmost flame edge, or None if not detected
    """
    # Use adaptive threshold: max of fixed threshold and percentage of row max
    row_max = np.max(row)
    adaptive_threshold = max(threshold, row_max * 0.05)  # At least 5% of max intensity

    above_thresh = row > adaptive_threshold

    if not np.any(above_thresh):
        return None

    # Find contiguous regions above threshold
    # Detect transitions: 0->1 (start) and 1->0 (end)
    padded = np.concatenate([[False], above_thresh, [False]])
    starts = np.where(padded[:-1] < padded[1:])[0]
    ends = np.where(padded[:-1] > padded[1:])[0]

    if len(starts) == 0:
        return None

    # Find the largest contiguous region (most likely the actual flame)
    region_sizes = ends - starts
    valid_regions = region_sizes >= min_region_size

    if not np.any(valid_regions):
        # No region large enough, fall back to largest region
        largest_idx = np.argmax(region_sizes)
        return int(ends[largest_idx] - 1)

    # Among valid regions, find the one with highest total intensity (the real flame)
    best_region_idx = None
    best_intensity = 0

    for i in range(len(starts)):
        if valid_regions[i]:
            region_intensity = np.sum(row[starts[i]:ends[i]])
            if region_intensity > best_intensity:
                best_intensity = region_intensity
                best_region_idx = i

    if best_region_idx is None:
        return None

    # Return rightmost edge of the best region
    return int(ends[best_region_idx] - 1)


def detect_flame_edge_gradient(
    row: np.ndarray,
    smooth_window: int = 5,
    min_gradient: float = 1.0
) -> Optional[int]:
    """
    Detect flame edge using maximum gradient (derivative) method.

    Finds the steepest intensity DROP on the RIGHT side of the peak,
    which corresponds to the leading edge of a flame front propagating rightward.

    Args:
        row: 1D array of pixel intensities along centerline
        smooth_window: Window size for smoothing before gradient calculation
        min_gradient: Minimum gradient magnitude to consider valid

    Returns:
        Pixel index of flame edge (steepest drop location), or None if not detected
    """
    if len(row) < smooth_window + 2:
        return None

    # Smooth the signal to reduce noise
    if smooth_window > 1:
        kernel = np.ones(smooth_window) / smooth_window
        smoothed = np.convolve(row, kernel, mode='same')
    else:
        smoothed = row.astype(np.float64)

    # Find the peak location first
    peak_idx = np.argmax(smoothed)
    peak_val = smoothed[peak_idx]

    if peak_val < min_gradient * 10:
        return None

    # Compute gradient (forward difference)
    gradient = np.diff(smoothed)

    # Only look at gradient to the RIGHT of the peak (falling edge / leading edge of flame)
    # We want the steepest negative gradient (biggest drop) after the peak
    if peak_idx >= len(gradient):
        return None

    right_gradient = gradient[peak_idx:]

    if len(right_gradient) == 0:
        return None

    # Find the steepest descent (most negative gradient) on the right side
    min_grad_local_idx = np.argmin(right_gradient)
    min_grad_val = right_gradient[min_grad_local_idx]

    # Check if the gradient is significant enough
    if abs(min_grad_val) < min_gradient:
        return None

    # Convert back to global index
    edge_idx = peak_idx + min_grad_local_idx

    return int(edge_idx)


def detect_flame_edge_half_maximum(
    row: np.ndarray,
    smooth_window: int = 5
) -> Optional[int]:
    """
    Detect flame edge using half-maximum method.

    Finds where intensity drops to 50% of peak value on the trailing edge
    (right side) of the flame. This is a standard physics method for edge detection.

    Args:
        row: 1D array of pixel intensities along centerline
        smooth_window: Window size for smoothing

    Returns:
        Pixel index where intensity crosses half-maximum, or None if not detected
    """
    if len(row) < smooth_window + 2:
        return None

    # Smooth the signal
    if smooth_window > 1:
        kernel = np.ones(smooth_window) / smooth_window
        smoothed = np.convolve(row, kernel, mode='same')
    else:
        smoothed = row.astype(np.float64)

    # Find the peak
    peak_idx = np.argmax(smoothed)
    peak_val = smoothed[peak_idx]

    if peak_val < 10:  # No significant signal
        return None

    # Calculate half-maximum value
    # Use local baseline (minimum near the right side) for better accuracy
    right_region = smoothed[peak_idx:]
    if len(right_region) < 5:
        return None

    baseline = np.min(right_region[-min(50, len(right_region)):])  # Baseline from far right
    half_max = baseline + (peak_val - baseline) * 0.5

    # Find where intensity crosses half-maximum on the RIGHT side of peak
    right_side = smoothed[peak_idx:]

    # Find first crossing below half-max
    below_half = right_side < half_max

    if not np.any(below_half):
        return None

    # Find the first index where we go below half-max
    first_below = np.argmax(below_half)

    if first_below == 0:
        # Already below at peak, something's wrong
        return None

    # Interpolate for sub-pixel accuracy
    idx_above = peak_idx + first_below - 1
    idx_below = peak_idx + first_below

    if idx_below >= len(smoothed):
        return int(idx_above)

    # Linear interpolation
    val_above = smoothed[idx_above]
    val_below = smoothed[idx_below]

    if val_above == val_below:
        return int(idx_above)

    fraction = (val_above - half_max) / (val_above - val_below)
    edge_position = idx_above + fraction

    return int(round(edge_position))


def detect_flame_edge(
    row: np.ndarray,
    method: str = "half_maximum",
    **kwargs
) -> Optional[int]:
    """
    Unified flame edge detection with selectable method.

    Args:
        row: 1D array of pixel intensities along centerline
        method: Detection method - "threshold", "gradient", or "half_maximum"
        **kwargs: Additional arguments passed to the specific method

    Returns:
        Pixel index of flame edge, or None if not detected
    """
    if method == "threshold":
        return detect_rightmost_flame_edge(row, **kwargs)
    elif method == "gradient":
        return detect_flame_edge_gradient(row, **kwargs)
    elif method == "half_maximum":
        return detect_flame_edge_half_maximum(row, **kwargs)
    else:
        raise ValueError(f"Unknown detection method: {method}")


def process_frame(
    frame: np.ndarray,
    background_scalar: float,
    calibration: float,
    prior_frame: Optional[np.ndarray] = None,
    use_frame_diff: bool = False
) -> Tuple[Optional[int], Optional[float], np.ndarray]:
    """
    Process a single frame to detect flame position.

    Args:
        frame: Frame data as numpy array
        background_scalar: Background value to subtract
        calibration: Meters per pixel
        prior_frame: Optional previous frame for frame differencing
        use_frame_diff: If True, combine background subtraction with frame differencing

    Returns:
        Tuple of (pixel_position, position_in_meters, processed_image)
    """
    # Background subtraction
    img_subtracted = subtract_scalar_background(frame, background_scalar)

    # Optionally combine with prior frame differencing
    if use_frame_diff and prior_frame is not None:
        frame_diff = subtract_prior_frame(frame, prior_frame, threshold=10.0)
        # Combine: use frame diff to enhance the moving flame front
        img_processed = img_subtracted * (frame_diff > 0).astype(float) + frame_diff * 0.5
        img_processed = np.clip(img_processed, 0, None)
    else:
        img_processed = img_subtracted

    # Detect flame edge at center row
    center_row = img_processed.shape[0] // 2
    pixel_idx = detect_rightmost_flame_edge(img_processed[center_row, :])

    if pixel_idx is not None:
        position_m = pixel_idx * calibration
        return pixel_idx, position_m, img_processed

    return None, None, img_processed


def write_results(output_dict: dict, path: str) -> str:
    """Write results to space-delimited text file."""
    csv.register_dialect('gnuplot_spaces', delimiter=' ', skipinitialspace=True)

    with open(path, 'w', newline='') as f:
        fieldnames = list(output_dict.keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames, dialect='gnuplot_spaces')
        writer.writeheader()

        n_rows = len(list(output_dict.values())[0])
        for i in range(n_rows):
            row = {key: output_dict[key][i] for key in fieldnames}
            writer.writerow(row)

    return path


def save_frame_image(
    frame: np.ndarray,
    frame_subtracted: np.ndarray,
    frame_diff: Optional[np.ndarray],
    flame_position: Optional[int],
    frame_idx: int,
    time_s: float,
    output_path: Path,
    source_name: str
) -> None:
    """
    Save frame image with flame location marker and centerline intensity plot.

    Args:
        frame: Original frame data
        frame_subtracted: Background-subtracted frame
        frame_diff: Frame difference image (can be None)
        flame_position: Detected flame position in pixels (x-coordinate)
        frame_idx: Frame index
        time_s: Time in seconds
        output_path: Output directory path
        source_name: Name of video source for filename
    """
    center_row = frame.shape[0] // 2
    height, width = frame.shape[:2]

    # Create figure with 4 rows: original, BG subtracted, frame diff, centerline intensity
    n_image_rows = 3 if frame_diff is not None else 2

    fig = plt.figure(figsize=(12, 10))

    # Use GridSpec for flexible layout
    if frame_diff is not None:
        gs = fig.add_gridspec(4, 1, height_ratios=[1, 1, 1, 0.8], hspace=0.3)
    else:
        gs = fig.add_gridspec(3, 1, height_ratios=[1, 1, 0.8], hspace=0.3)

    # Original frame
    ax0 = fig.add_subplot(gs[0])
    ax0.imshow(frame, cmap='gray', aspect='equal')
    ax0.axhline(y=center_row, color='cyan', linestyle='--', linewidth=0.5, alpha=0.5)
    ax0.set_title(f'Original - Frame {frame_idx}')
    ax0.set_xlabel('X (pixels)')
    ax0.set_ylabel('Y (pixels)')

    # Background-subtracted frame with flame marker
    ax1 = fig.add_subplot(gs[1])
    ax1.imshow(frame_subtracted, cmap='gray', aspect='equal')
    ax1.axhline(y=center_row, color='cyan', linestyle='--', linewidth=0.5, alpha=0.5)

    if flame_position is not None:
        ax1.plot(flame_position, center_row, 'ro', markersize=8, markeredgecolor='yellow', markeredgewidth=1)
        ax1.set_title(f'BG Subtracted | t={time_s*1e6:.1f} µs | x={flame_position} px')
    else:
        ax1.set_title(f'BG Subtracted | t={time_s*1e6:.1f} µs | No detection')

    ax1.set_xlabel('X (pixels)')
    ax1.set_ylabel('Y (pixels)')

    # Frame difference (if available)
    if frame_diff is not None:
        ax2 = fig.add_subplot(gs[2])
        ax2.imshow(frame_diff, cmap='hot', aspect='equal')
        ax2.axhline(y=center_row, color='cyan', linestyle='--', linewidth=0.5, alpha=0.5)

        if flame_position is not None:
            ax2.plot(flame_position, center_row, 'go', markersize=8, markeredgecolor='white', markeredgewidth=1)

        ax2.set_title(f'Frame Difference (motion)')
        ax2.set_xlabel('X (pixels)')
        ax2.set_ylabel('Y (pixels)')

        intensity_ax_idx = 3
    else:
        intensity_ax_idx = 2

    # Centerline intensity plot
    ax_intensity = fig.add_subplot(gs[intensity_ax_idx])

    # Extract centerline profiles
    x_pixels = np.arange(width)
    centerline_original = frame[center_row, :]
    centerline_subtracted = frame_subtracted[center_row, :]

    ax_intensity.plot(x_pixels, centerline_original, 'b-', label='Original', alpha=0.5, linewidth=1)
    ax_intensity.plot(x_pixels, centerline_subtracted, 'k-', label='BG Subtracted', linewidth=1.5)

    if frame_diff is not None:
        centerline_diff = frame_diff[center_row, :]
        ax_intensity.plot(x_pixels, centerline_diff, 'r-', label='Frame Diff', linewidth=1.5)

    # Show all detection methods for comparison
    if frame_diff is not None:
        # Run all detection methods and show results
        pos_threshold = detect_rightmost_flame_edge(centerline_diff)
        pos_gradient = detect_flame_edge_gradient(centerline_diff)
        pos_half_max = detect_flame_edge_half_maximum(centerline_diff)

        # Mark each method with different colors
        if pos_threshold is not None:
            ax_intensity.axvline(x=pos_threshold, color='red', linestyle=':', linewidth=1.5,
                               label=f'Threshold: {pos_threshold} px', alpha=0.7)
        if pos_gradient is not None:
            ax_intensity.axvline(x=pos_gradient, color='blue', linestyle='-.', linewidth=1.5,
                               label=f'Gradient: {pos_gradient} px', alpha=0.7)
        if pos_half_max is not None:
            ax_intensity.axvline(x=pos_half_max, color='green', linestyle='--', linewidth=2,
                               label=f'Half-max: {pos_half_max} px')

        # Highlight the selected method (flame_position)
        if flame_position is not None:
            ax_intensity.axvline(x=flame_position, color='magenta', linestyle='-', linewidth=2.5,
                               label=f'Selected: {flame_position} px', alpha=0.5)
    elif flame_position is not None:
        ax_intensity.axvline(x=flame_position, color='green', linestyle='--', linewidth=2,
                           label=f'Flame @ {flame_position} px')

    ax_intensity.set_xlabel('X (pixels)')
    ax_intensity.set_ylabel('Intensity')
    ax_intensity.set_title('Centerline Intensity Profile (Frame Diff)')
    ax_intensity.legend(loc='upper right', fontsize=8)
    ax_intensity.set_xlim(0, width)
    ax_intensity.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save figure
    output_file = output_path / f"{source_name}-Frame-{frame_idx:06d}.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close(fig)


def generate_stacked_sequence(
    video,
    frame_indices: List[int],
    background_scalar: float,
    output_path: Path,
    title: str = "",
    show_frame_diff: bool = True,
    show_tracking: bool = True,
    detection_threshold: float = 5.0,
    figsize_width: float = 10.0
) -> List[Tuple[int, Optional[int]]]:
    """
    Generate stacked frame sequence image (paper-style visualization).

    Creates a figure with frames stacked vertically, numbered on the left,
    similar to DDT progression figures in combustion papers.

    Args:
        video: PhotonVideo instance
        frame_indices: List of frame indices to include
        background_scalar: Background value for subtraction
        output_path: Path to save the output image
        title: Optional title for the figure
        show_frame_diff: If True, show two columns (original + processed)
        show_tracking: If True, show flame position markers
        detection_threshold: Threshold for flame detection on frame difference
        figsize_width: Width of the figure in inches

    Returns:
        List of (frame_idx, flame_position) tuples
    """
    n_frames = len(frame_indices)
    height, width = video.frame_shape
    center_row = height // 2

    # Determine number of columns
    n_cols = 2 if show_frame_diff else 1

    # Calculate figure dimensions
    aspect_ratio = width / height
    panel_height = (figsize_width / n_cols) / aspect_ratio
    fig_height = panel_height * n_frames

    fig, axes = plt.subplots(n_frames, n_cols, figsize=(figsize_width, fig_height))

    # Handle single column case
    if n_cols == 1:
        axes = axes.reshape(-1, 1)

    prior_frame = None
    tracking_results = []

    for i, frame_idx in enumerate(frame_indices):
        frame = video[frame_idx]
        frame_subtracted = subtract_scalar_background(frame, background_scalar)

        # Compute frame difference
        if prior_frame is not None:
            frame_diff = subtract_prior_frame(frame, prior_frame, threshold=0.0)
        else:
            frame_diff = np.zeros_like(frame)

        # Detect flame position on FRAME DIFFERENCE
        if prior_frame is not None:
            flame_pos = detect_rightmost_flame_edge(frame_diff[center_row, :], threshold=detection_threshold)
        else:
            flame_pos = None

        tracking_results.append((frame_idx, flame_pos))

        # Column 1: Background subtracted
        axes[i, 0].imshow(frame_subtracted, cmap='gray', aspect='equal', vmin=0)
        axes[i, 0].set_ylabel(f'{i+1}', rotation=0, labelpad=20, fontsize=10, fontweight='bold', color='white')
        axes[i, 0].set_xticks([])
        axes[i, 0].set_yticks([])

        # Show tracking marker on BG subtracted
        if show_tracking and flame_pos is not None:
            axes[i, 0].plot(flame_pos, center_row, 'ro', markersize=4, markeredgecolor='yellow', markeredgewidth=0.5)
            axes[i, 0].axhline(y=center_row, color='cyan', linewidth=0.3, alpha=0.3)

        # Column 2: Frame difference (if enabled)
        if n_cols > 1:
            axes[i, 1].imshow(frame_diff, cmap='gray', aspect='equal', vmin=0)
            axes[i, 1].set_xticks([])
            axes[i, 1].set_yticks([])

            # Show tracking marker on frame diff
            if show_tracking and flame_pos is not None:
                axes[i, 1].plot(flame_pos, center_row, 'ro', markersize=4, markeredgecolor='yellow', markeredgewidth=0.5)
                axes[i, 1].axhline(y=center_row, color='cyan', linewidth=0.3, alpha=0.3)

        prior_frame = frame.copy()

    # Remove spacing between subplots
    plt.subplots_adjust(wspace=0.02, hspace=0)

    if title:
        fig.suptitle(title, fontsize=12, fontweight='bold', color='white')

    # Save figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='black', edgecolor='none')
    plt.close(fig)
    print(f"Saved stacked sequence: {output_path}")

    return tracking_results


def generate_stacked_sequence_single_column(
    video,
    frame_indices: List[int],
    background_scalar: float,
    output_path: Path,
    use_frame_diff: bool = False,
    show_tracking: bool = True,
    detection_threshold: float = 5.0,
    title: str = "",
    figsize_width: float = 6.0
) -> List[Tuple[int, Optional[int]]]:
    """
    Generate single-column stacked frame sequence (compact paper-style).

    Args:
        video: PhotonVideo instance
        frame_indices: List of frame indices to include
        background_scalar: Background value for subtraction
        output_path: Path to save the output image
        use_frame_diff: If True, show frame difference instead of BG subtracted
        show_tracking: If True, show flame position markers (detected on frame diff)
        detection_threshold: Threshold for flame detection
        title: Optional title
        figsize_width: Width of figure

    Returns:
        List of (frame_idx, flame_position) tuples
    """
    n_frames = len(frame_indices)
    height, width = video.frame_shape
    center_row = height // 2

    # Stack all frames into single image
    stacked_height = height * n_frames
    stacked_image = np.zeros((stacked_height, width), dtype=np.float64)

    prior_frame = None
    tracking_results = []

    for i, frame_idx in enumerate(frame_indices):
        frame = video[frame_idx]
        frame_subtracted = subtract_scalar_background(frame, background_scalar)

        # Compute frame difference for tracking
        if prior_frame is not None:
            frame_diff = subtract_prior_frame(frame, prior_frame, threshold=0.0)
            # Detect flame on frame difference
            flame_pos = detect_rightmost_flame_edge(frame_diff[center_row, :], threshold=detection_threshold)
        else:
            frame_diff = np.zeros_like(frame)
            flame_pos = None

        tracking_results.append((frame_idx, flame_pos))

        # Choose display frame
        if use_frame_diff:
            display_frame = frame_diff
        else:
            display_frame = frame_subtracted

        # Place frame in stack
        y_start = i * height
        y_end = (i + 1) * height
        stacked_image[y_start:y_end, :] = display_frame

        prior_frame = frame.copy()

    # Create figure
    aspect_ratio = width / stacked_height
    fig_height = figsize_width / aspect_ratio

    fig, ax = plt.subplots(figsize=(figsize_width, fig_height))
    ax.imshow(stacked_image, cmap='gray', aspect='equal', vmin=0)

    # Add frame numbers, separator lines, and tracking markers
    for i, (frame_idx, flame_pos) in enumerate(zip(frame_indices, [r[1] for r in tracking_results])):
        y_center = i * height + center_row
        ax.text(-width * 0.02, y_center, f'{i+1}', color='white', fontsize=8,
                fontweight='bold', ha='right', va='center')

        # Horizontal separator line
        if i > 0:
            ax.axhline(y=i * height - 0.5, color='white', linewidth=0.5, alpha=0.5)

        # Flame tracking marker
        if show_tracking and flame_pos is not None:
            ax.plot(flame_pos, y_center, 'ro', markersize=3, markeredgecolor='yellow', markeredgewidth=0.5)

    ax.set_xlim(-width * 0.05, width)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor('black')

    if title:
        ax.set_title(title, color='white', fontsize=10, fontweight='bold')

    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='black', edgecolor='none')
    plt.close(fig)
    print(f"Saved stacked sequence: {output_path}")

    return tracking_results


########################################################################################################################
# Main Processing
########################################################################################################################

def process_video_source(config: VideoSourceConfig, processor: Optional[MPIVideoProcessor] = None):
    """
    Process all video files for a given source configuration.

    Args:
        config: Video source configuration
        processor: Optional MPI processor for parallel execution
    """
    is_root = processor is None or processor.is_root
    rank = 0 if processor is None else processor.rank

    if is_root:
        print(f"\n{'='*60}")
        print(f"Processing: {config.name}")
        print(f"Video path: {config.video_path}")
        print(f"Default calibration: {config.calibration} m/pixel")
        print(f"Default position offset: {config.position_offset} m")
        if config.file_calibrations:
            print(f"File-specific calibrations: {len(config.file_calibrations)} rules defined")
        print(f"{'='*60}")

    # Find all CIHX files in the video path
    video_path = Path(config.video_path)
    cihx_files = list(video_path.rglob("*.cihx"))

    if not cihx_files:
        if is_root:
            print(f"No CIHX files found in {config.video_path}")
        return

    # Process each video file
    for cihx_file in sorted(cihx_files):
        # Get per-file calibration and position offset
        file_calibration, file_position_offset = config.get_calibration_for_file(cihx_file.name)

        if is_root:
            print(f"\nLoading: {cihx_file.name}")
            print(f"  Using calibration: {file_calibration} m/pixel, offset: {file_position_offset} m")

        # Load video with file-specific calibration
        video = open_video(
            str(cihx_file),
            trigger_frame=config.trigger_frame,
            calibration=SpatialCalibration(
                scale=file_calibration,
                units='m'
            )
        )

        if is_root:
            print(f"  Frames: {len(video)}")
            print(f"  Frame rate: {video.frame_rate} fps")
            print(f"  Frame shape: {video.frame_shape}")
            print(f"  Duration: {video.duration:.6f} s")

            # Print CIHX timing metadata (parsed from XML)
            if video.has_absolute_timing:
                print(f"  CIHX Timing (parsed from XML):")
                cihx = video.cihx_metadata
                print(f"    Recording datetime: {cihx.get('recording_datetime')}")
                print(f"    Record rate: {cihx.get('record_rate')} fps")
                print(f"    Start frame: {cihx.get('start_frame')}")
                print(f"    Recorded frame (at trigger): {cihx.get('recorded_frame')}")
                print(f"    Skip frame: {cihx.get('skip_frame')}")
                print(f"    IRIG enabled: {cihx.get('irig_enabled')}")
                print(f"    Shutter speed: {cihx.get('shutter_speed_ns')} ns")
                # Show sample timing for first and last frame
                print(f"    Frame 0 absolute time: {video.get_absolute_time(0):.9f} s")
                print(f"    Frame 0 datetime: {video.get_datetime(0)}")
                print(f"    Frame {len(video)-1} absolute time: {video.get_absolute_time(len(video)-1):.9f} s")
                print(f"    Frame {len(video)-1} datetime: {video.get_datetime(len(video)-1)}")
            else:
                print(f"  CIHX Timing: Not available (using pyMRAW timing)")

            # Debug: Print all raw metadata to see available timing fields
            print(f"  Raw metadata fields (from pyMRAW):")
            for key, value in sorted(video.raw_metadata.items()):
                print(f"    {key}: {value}")

        # Get background from first frame
        background_frame = video[0]
        background_scalar = float(np.max(background_frame))

        # Extract centerline noise baseline from first frame (assumed empty/no flame)
        center_row = background_frame.shape[0] // 2
        centerline_noise = background_frame[center_row, :].astype(np.float64)
        centerline_noise_mean = np.mean(centerline_noise)
        centerline_noise_std = np.std(centerline_noise)
        centerline_noise_max = np.max(centerline_noise)
        # Threshold: signal must exceed noise by significant margin (e.g., 5 sigma above mean, or 2x max noise)
        centerline_flame_threshold = max(
            centerline_noise_mean + 5 * centerline_noise_std,
            centerline_noise_max * 2.0
        )

        # Create output directories
        output_dir = Path(config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        frames_output_dir = output_dir / f"{cihx_file.stem}-frames"
        frames_output_dir.mkdir(parents=True, exist_ok=True)

        if is_root:
            print(f"  Background scalar: {background_scalar}")
            print(f"  Centerline noise (from frame 0): mean={centerline_noise_mean:.1f}, std={centerline_noise_std:.1f}, max={centerline_noise_max:.1f}")
            print(f"  Centerline flame threshold: {centerline_flame_threshold:.1f}")

            # Generate paper-style stacked sequence visualization
            # Select frames to display (e.g., every 10th frame, or specific range)
            total_frames = len(video)
            n_display = min(15, total_frames)  # Show up to 15 frames
            step = max(1, total_frames // n_display)
            display_frames = list(range(0, total_frames, step))[:n_display]

            print(f"  Generating stacked sequence with frames: {display_frames}")

            # Two-column version (BG subtracted + frame diff) - save in frames folder
            generate_stacked_sequence(
                video=video,
                frame_indices=display_frames,
                background_scalar=background_scalar,
                output_path=frames_output_dir / f"{cihx_file.stem}-stacked-sequence.png",
                title=f"{cihx_file.stem}",
                show_frame_diff=True,
                figsize_width=12.0
            )

            # Single column version (just BG subtracted) - save in frames folder
            generate_stacked_sequence_single_column(
                video=video,
                frame_indices=display_frames,
                background_scalar=background_scalar,
                output_path=frames_output_dir / f"{cihx_file.stem}-stacked-single.png",
                use_frame_diff=False,
                title=f"{cihx_file.stem}",
                figsize_width=8.0
            )

        # Process frames
        results = {
            '#Frame': [],
            'Time_s': [],
            'Position_px': [],
            'Position_m': [],
        }

        # Determine frame indices to process
        if processor is not None:
            frame_indices = processor.distribute_indices(len(video))
        else:
            frame_indices = list(range(len(video)))

        local_results = []
        flame_exited = False

        prior_frame = None
        prior_flame_pos = None  # Track previous flame position for velocity check
        empty_frame_count = 0

        for frame_idx in frame_indices:
            # Skip explicitly configured frames (e.g., no centerline flame detection)
            if frame_idx in config.skip_frames:
                print(f"[Rank {rank}] Skipping frame {frame_idx} (in skip_frames list)")
                continue

            frame = video[frame_idx]
            # Calculate time: absolute (from CIHX start_frame) or relative (to trigger)
            if config.use_absolute_time:
                time_s = video.get_absolute_time(frame_idx)
            else:
                time_s = video.get_time(frame_idx)

            # Subtract background for processing and visualization
            frame_subtracted = subtract_scalar_background(frame, background_scalar)

            # Skip empty frames (no significant signal)
            noise_thresh = max(10.0, background_scalar * 0.5)
            if is_empty_frame(frame_subtracted, noise_threshold=noise_thresh, min_signal_fraction=0.0005):
                empty_frame_count += 1
                prior_frame = frame.copy()
                continue

            # Compute frame difference (prior frame subtraction) for motion detection
            if prior_frame is not None:
                frame_diff = subtract_prior_frame(frame, prior_frame, threshold=5.0)
            else:
                frame_diff = None

            # Check if centerline has flame signal above noise threshold (from frame 0)
            # This determines if there's meaningful flame at the center before attempting detection
            current_centerline = frame_subtracted[center_row, :]
            centerline_max_intensity = np.max(current_centerline)
            has_centerline_flame = centerline_max_intensity > centerline_flame_threshold

            # Automatic skip: if no flame along centerline and tracking hasn't started yet,
            # skip this frame entirely without updating prior_frame
            if not has_centerline_flame and prior_flame_pos is None:
                print(f"[Rank {rank}] Skipping frame {frame_idx} (centerline max={centerline_max_intensity:.1f} < threshold={centerline_flame_threshold:.1f})")
                continue

            # Detect flame edge on the DIFFERENCE IMAGE (motion-based detection)
            # This isolates the moving flame front from static background
            if frame_diff is not None:
                pixel_pos = detect_flame_edge(
                    frame_diff[center_row, :],
                    method=config.detection_method
                )
            else:
                # First frame - no prior available, skip detection
                pixel_pos = None

            # Enforce non-negative velocity: flame can only move forward (increasing x)
            if pixel_pos is not None and prior_flame_pos is not None:
                if pixel_pos < prior_flame_pos:
                    # Reject detection - flame cannot move backwards
                    # Use prior position as current (flame is at least where it was)
                    pixel_pos = None

            # Check for flame exiting domain using frame difference peak location
            # This is more robust than relying on detection failure
            flame_exited = False
            if frame_diff is not None and prior_flame_pos is not None:
                diff_centerline = frame_diff[center_row, :]
                diff_peak_idx = int(np.argmax(diff_centerline))
                diff_peak_val = float(diff_centerline[diff_peak_idx])

                # Flame is exiting if:
                # 1. Peak is near right edge (within 5% of domain width)
                # 2. There's significant signal (above noise threshold)
                right_edge_threshold = video.width * 0.95
                if diff_peak_idx >= right_edge_threshold and diff_peak_val > centerline_flame_threshold:
                    print(f"[Rank {rank}] Flame exiting domain at frame {frame_idx}")
                    print(f"  Frame diff peak at {diff_peak_idx}/{video.width} pixels (intensity={diff_peak_val:.1f})")

                    # Use peak position as final flame position
                    pixel_pos = diff_peak_idx
                    pos_m = pixel_pos * file_calibration + file_position_offset
                    local_results.append((frame_idx, time_s, pixel_pos, pos_m))
                    flame_exited = True

                    # Save this final frame
                    save_frame_image(
                        frame=frame,
                        frame_subtracted=frame_subtracted,
                        frame_diff=frame_diff,
                        flame_position=pixel_pos,
                        frame_idx=frame_idx,
                        time_s=time_s,
                        output_path=frames_output_dir,
                        source_name=config.name
                    )
                    break

            if pixel_pos is not None:
                pos_m = pixel_pos * file_calibration + file_position_offset
                local_results.append((frame_idx, time_s, pixel_pos, pos_m))
                prior_flame_pos = pixel_pos  # Update for next iteration

            # Save frame image
            save_frame_image(
                frame=frame,
                frame_subtracted=frame_subtracted,
                frame_diff=frame_diff,
                flame_position=pixel_pos,
                frame_idx=frame_idx,
                time_s=time_s,
                output_path=frames_output_dir,
                source_name=config.name
            )

            # Store current frame as prior for next iteration
            prior_frame = frame.copy()

            if frame_idx % 50 == 0:
                print(f"[Rank {rank}] Processed frame {frame_idx}/{len(video)} (skipped {empty_frame_count} empty)")

        if is_root:
            print(f"  Skipped {empty_frame_count} empty/noise-only frames")

        # Gather results if using MPI
        if processor is not None:
            all_results = processor.gather(local_results)
            if is_root:
                # Flatten and sort
                flat_results = [item for sublist in all_results for item in sublist]
                flat_results.sort(key=lambda x: x[0])
            else:
                flat_results = []
        else:
            flat_results = local_results

        # Write results (root only)
        if is_root and flat_results:
            for frame_idx, time_s, pixel_pos, pos_m in flat_results:
                results['#Frame'].append(frame_idx)
                results['Time_s'].append(f"{time_s:.9f}")
                results['Position_px'].append(pixel_pos)
                results['Position_m'].append(f"{pos_m:.9f}")

            # Write results file
            output_file = output_dir / f"{cihx_file.stem}-flame-position.txt"
            write_results(results, str(output_file))
            print(f"\nResults saved to: {output_file}")
            print(f"  Total frames processed: {len(flat_results)}")

        video.close()


def main():
    """Main entry point."""
    # Initialize MPI if available
    if HAS_MPI:
        comm = MPI.COMM_WORLD
        processor = MPIVideoProcessor(comm)
        if processor.is_root:
            print(f"Running with MPI: {processor.size} ranks")
    else:
        processor = None
        print("Running in serial mode (mpi4py not available)")

    # Configure video sources
    nova_config = VideoSourceConfig(name="Nova")
    nova_config.enabled = True
    nova_config.calibration = 0.00074074  # meters per pixel (default)
    nova_config.position_offset = 0.0  # meters (default)
    nova_config.use_frame_diff = True  # Enable prior frame subtraction for flame isolation
    nova_config.detection_method = "half_maximum"  # Options: "threshold", "gradient", "half_maximum"
    nova_config.use_absolute_time = True  # Use absolute time from recording start
    nova_config.video_path = "./Video-Files"
    nova_config.output_dir = "./Processed-Photos/Nova-Output"

    # Per-file calibration and position offset
    nova_config.file_calibrations = [
        FileCalibration(
            calibration=0.000833333,
            position_offset=1.0159,
            files=["run-1-"]
        ),
        FileCalibration(
            calibration=0.000833333,
            position_offset=1.197565,
            files=["run-2-"]
        ),
        FileCalibration(
            calibration=0.000833333,
            position_offset=1.347567,
            files=["run-3-:run-10-"]
        ),
    ]

    # Process enabled sources
    if nova_config.enabled:
        process_video_source(nova_config, processor)

    # Synchronize before exit
    if processor is not None:
        processor.barrier()

    if processor is None or processor.is_root:
        print("\nProcessing complete!")


if __name__ == "__main__":
    main()
