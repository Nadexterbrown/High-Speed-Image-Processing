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
from scipy.ndimage import gaussian_filter, grey_opening, sobel
from scipy.interpolate import UnivariateSpline

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
    use_absolute_time: bool = True  # Use absolute time from recording start (not trigger-relative)
    skip_frames: List[int] = field(default_factory=list)  # Frames to skip
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


@dataclass
class FlameDetectorConfig:
    """Configuration for flame front detection algorithm."""

    # Preprocessing parameters (applied in order: frame_diff -> noise_removal -> blur)
    frame_diff_threshold: float = 5.0        # Threshold for frame differencing
    morphology_kernel_size: int = 3          # Kernel size for morphological opening (noise removal)
    gaussian_sigma: float = 1.5              # Gaussian blur sigma (1-2 recommended)

    # Detection parameters
    min_gradient_strength: float = 10.0      # Minimum gradient magnitude to consider valid
    edge_margin_px: int = 10                 # Margin from image edges to ignore
    sobel_threshold_fraction: float = 0.1    # Fraction of max Sobel value for "rightmost" detection

    # Tracking constraints
    max_velocity_change_m_s: float = 200.0   # Maximum velocity change between frames (m/s)

    # DDT detection
    ddt_velocity_jump_m_s: float = 1250.0    # Velocity jump threshold to detect DDT (m/s)

    # Spline estimator parameters
    use_spline_estimator: bool = True        # Use spline for position prediction
    spline_smoothing: float = 0.5            # Spline smoothing factor (0=interpolate, higher=smoother)
    min_points_for_spline: int = 5           # Minimum points before using spline

    # Search window
    search_window_px: int = 100              # Search window half-width around predicted position

    # Domain exit
    exit_margin_px: int = 15                 # Stop when position >= width - exit_margin_px


@dataclass
class FlameDetectionResult:
    """Results from a single frame's flame detection."""
    frame_idx: int
    time_s: float

    # Processing step outputs (for visualization)
    frame_subtracted: np.ndarray          # Background-subtracted frame
    frame_diff: Optional[np.ndarray]      # Step 1: current - prior
    noise_removed: Optional[np.ndarray]   # Step 2: morphological opening on diff
    blurred: Optional[np.ndarray]         # Step 3: Gaussian blur
    sobel_output: Optional[np.ndarray]    # Step 4a: Sobel filter
    gradient_output: Optional[np.ndarray] # Step 4b: np.gradient filter

    # Detection results
    pos_min_gradient: Optional[int]       # Position from minimum gradient
    pos_rightmost_sobel: Optional[int]    # Position from rightmost Sobel
    pos_spline_predicted: Optional[int]   # Position from spline estimator
    search_bounds: Optional[Tuple[int, int]]  # Velocity-constrained search bounds

    # Final selected position
    final_position: Optional[int]


class FlameDetector:
    """
    Flame front detection with velocity-constrained tracking.

    Processing pipeline (in order):
    1. Subtract prior frame from current frame (isolate new flame region)
    2. Remove isolated pixels (morphological opening on diff)
    3. Apply Gaussian blur to smooth
    4. Apply Sobel filter AND gradient filter (separately)
    5. Find position by: (a) min gradient, (b) rightmost Sobel
    6. Compare with spline estimator prediction bounded by velocity
    """

    def __init__(
        self,
        config: FlameDetectorConfig,
        frame_rate: float,
        calibration_m_per_px: float
    ):
        """
        Initialize flame detector.

        Args:
            config: Detection configuration parameters
            frame_rate: Video frame rate in fps
            calibration_m_per_px: Spatial calibration (meters per pixel)
        """
        self.config = config
        self.frame_rate = frame_rate
        self.calibration = calibration_m_per_px

        # Tracking state
        self._position_history: List[Tuple[int, Optional[int]]] = []
        # Velocity history: (frame_idx, v_backward1, v_backward2, v_central)
        # v_backward1: first-order backward difference (current method)
        # v_backward2: second-order backward difference
        # v_central: second-order central difference (for prior time point)
        self._velocity_history: List[Tuple[int, float, Optional[float], Optional[float]]] = []
        self._prior_frame: Optional[np.ndarray] = None  # BG-subtracted prior frame
        self._spline: Optional[UnivariateSpline] = None

        # DDT detection
        self._ddt_frame_idx: Optional[int] = None  # Frame where DDT was detected

        # Store all detection results for plotting
        self._detection_results: List[FlameDetectionResult] = []

        # Precompute max pixel displacement per frame
        self._max_displacement_px = self._compute_max_displacement()

    def _compute_max_displacement(self) -> int:
        """Compute maximum allowed pixel displacement between frames."""
        if self.frame_rate <= 0 or self.calibration <= 0:
            return 1000  # No constraint if parameters unknown
        dt = 1.0 / self.frame_rate
        max_displacement_m = self.config.max_velocity_change_m_s * dt
        return int(np.ceil(max_displacement_m / self.calibration)) + 1

    def reset(self) -> None:
        """Reset tracking state for a new video."""
        self._position_history.clear()
        self._velocity_history.clear()
        self._detection_results.clear()
        self._prior_frame = None
        self._spline = None
        self._ddt_frame_idx = None

    def _update_spline(self) -> None:
        """Update spline estimator with current position history."""
        valid_points = [(f, p) for f, p in self._position_history if p is not None]
        if len(valid_points) < self.config.min_points_for_spline:
            self._spline = None
            return

        frames = np.array([f for f, p in valid_points])
        positions = np.array([p for f, p in valid_points])

        try:
            # Use smoothing spline - higher s = smoother
            self._spline = UnivariateSpline(
                frames, positions,
                s=self.config.spline_smoothing * len(frames),
                k=min(3, len(frames) - 1)  # Cubic or lower if not enough points
            )
        except Exception:
            self._spline = None

    def predict_with_spline(self, frame_idx: int) -> Optional[int]:
        """Predict position using spline estimator."""
        if self._spline is None:
            return None
        try:
            predicted = int(self._spline(frame_idx))
            return max(0, predicted)
        except Exception:
            return None

    def get_search_bounds(self, frame_idx: int, width: int) -> Tuple[int, int]:
        """
        Get velocity-constrained search bounds for this frame.

        Returns:
            (search_start, search_end) pixel positions
        """
        margin = self.config.edge_margin_px

        # Get last known position
        last_position = None
        last_frame_idx = None
        for f_idx, pos in reversed(self._position_history):
            if pos is not None:
                last_position = pos
                last_frame_idx = f_idx
                break

        if last_position is None:
            # No history - search full width
            return (margin, width - margin)

        # Compute bounds based on max velocity
        frames_elapsed = frame_idx - last_frame_idx
        max_displacement = self._max_displacement_px * max(1, frames_elapsed)

        # Flame only moves right, so search_start = last_position
        # search_end = last_position + max_displacement
        search_start = last_position
        search_end = min(width - margin, last_position + max_displacement + self.config.search_window_px)

        return (search_start, search_end)

    def detect(
        self,
        frame: np.ndarray,
        frame_idx: int,
        background_scalar: float
    ) -> FlameDetectionResult:
        """
        Main detection entry point with full pipeline.

        Processing steps:
        1. Subtract prior frame from current (isolate new flame region)
        2. Remove isolated pixels (morphological opening)
        3. Gaussian blur to smooth
        4. Apply Sobel filter AND gradient filter
        5. Find position by min gradient AND rightmost Sobel
        6. Compare with spline prediction

        Args:
            frame: Raw frame data
            frame_idx: Frame index
            background_scalar: Background value for subtraction

        Returns:
            FlameDetectionResult with all intermediate outputs
        """
        height, width = frame.shape[:2]
        center_row = height // 2
        time_s = frame_idx / self.frame_rate if self.frame_rate > 0 else 0

        # Background subtraction (always done first)
        frame_subtracted = subtract_scalar_background(frame, background_scalar)

        # Get search bounds from velocity constraint
        search_start, search_end = self.get_search_bounds(frame_idx, width)

        # Initialize outputs
        frame_diff = None
        noise_removed = None
        blurred = None
        sobel_output = None
        gradient_output = None
        pos_min_gradient = None
        pos_rightmost_sobel = None
        pos_spline_predicted = None

        # ========== STEP 1: Frame Differencing ==========
        # Subtract prior frame from current to isolate new flame region
        if self._prior_frame is not None:
            frame_diff = frame_subtracted.astype(np.float64) - self._prior_frame.astype(np.float64)
            frame_diff[frame_diff < self.config.frame_diff_threshold] = 0

            # ========== STEP 2: Remove Isolated Pixels ==========
            # Morphological opening on the frame difference to remove noise
            kernel_size = self.config.morphology_kernel_size
            noise_removed = grey_opening(frame_diff, size=(kernel_size, kernel_size))

            # ========== STEP 3: Gaussian Blur ==========
            blurred = gaussian_filter(noise_removed, sigma=self.config.gaussian_sigma)

            # ========== STEP 4a: Sobel Filter ==========
            sobel_output = sobel(blurred, axis=1)  # Horizontal gradient

            # ========== STEP 4b: Gradient Filter (np.gradient) ==========
            gradient_output = np.gradient(blurred, axis=1)

            # ========== STEP 5: Find Flame Position ==========
            # Extract centerline profiles in search region
            sobel_line = sobel_output[center_row, :]
            gradient_line = gradient_output[center_row, :]

            # Constrain to search bounds
            search_sobel = sobel_line[search_start:search_end]
            search_gradient = gradient_line[search_start:search_end]

            if len(search_sobel) > 0 and len(search_gradient) > 0:
                # Method A: Minimum gradient location (most negative = leading edge)
                # The leading edge has a DROP in intensity (negative gradient)
                min_grad_val = np.min(search_gradient)
                if min_grad_val < -self.config.min_gradient_strength:
                    min_grad_idx = np.argmin(search_gradient)
                    pos_min_gradient = search_start + min_grad_idx

                # Method B: Rightmost position in Sobel above threshold
                # Find the rightmost significant edge (positive or negative Sobel)
                sobel_max = np.max(np.abs(search_sobel))
                if sobel_max > self.config.min_gradient_strength:
                    threshold = sobel_max * self.config.sobel_threshold_fraction
                    above_thresh = np.abs(search_sobel) > threshold
                    if np.any(above_thresh):
                        rightmost_idx = np.max(np.where(above_thresh)[0])
                        pos_rightmost_sobel = search_start + rightmost_idx

        # ========== STEP 6: Spline Estimator Prediction ==========
        if self.config.use_spline_estimator:
            pos_spline_predicted = self.predict_with_spline(frame_idx)

        # ========== SELECT FINAL POSITION ==========
        # Use the rightmost detected position (this is the leading edge for left-to-right propagation)
        # Trust the Sobel/gradient detection - don't override with velocity constraints
        final_position = None

        # Collect all valid candidates from detection methods
        candidates = []
        if pos_min_gradient is not None:
            candidates.append(('min_gradient', pos_min_gradient))
        if pos_rightmost_sobel is not None:
            candidates.append(('rightmost_sobel', pos_rightmost_sobel))

        # Select the RIGHTMOST candidate (flame front is at the right edge)
        if candidates:
            # Sort by position (descending) and pick the rightmost
            candidates.sort(key=lambda x: x[1], reverse=True)
            best_method, candidate = candidates[0]

            # USE THE DETECTED POSITION DIRECTLY - trust Sobel/gradient over velocity prediction
            final_position = candidate

        # ========== UPDATE STATE ==========
        self._position_history.append((frame_idx, final_position))
        self._prior_frame = frame_subtracted.copy()

        # Update spline
        self._update_spline()

        # Update velocity history and detect DDT
        # Calculate three velocity methods:
        # 1) First-order backward: v_n = (x_n - x_{n-1}) / dt
        # 2) Second-order backward: v_n = (3*x_n - 4*x_{n-1} + x_{n-2}) / (2*dt)
        # 3) Second-order central: v_{n-1} = (x_n - x_{n-2}) / (2*dt)
        if final_position is not None and len(self._position_history) >= 2:
            curr_frame, curr_pos = self._position_history[-1]  # Current (just added)
            prev_frame, prev_pos = self._position_history[-2]  # Previous

            if prev_pos is not None and self.frame_rate > 0:
                dt = (curr_frame - prev_frame) / self.frame_rate
                if dt > 0:
                    # 1) First-order backward difference (current method)
                    v_backward1 = (curr_pos - prev_pos) * self.calibration / dt

                    # 2) Second-order backward difference (needs 3 points)
                    v_backward2: Optional[float] = None
                    if len(self._position_history) >= 3:
                        prev2_frame, prev2_pos = self._position_history[-3]
                        if prev2_pos is not None:
                            # Assuming uniform time steps
                            v_backward2 = (3*curr_pos - 4*prev_pos + prev2_pos) * self.calibration / (2*dt)

                    # 3) Second-order central difference (for prior time point)
                    # v_{n-1} = (x_n - x_{n-2}) / (2*dt)
                    v_central: Optional[float] = None
                    if len(self._position_history) >= 3:
                        prev2_frame, prev2_pos = self._position_history[-3]
                        if prev2_pos is not None:
                            v_central = (curr_pos - prev2_pos) * self.calibration / (2*dt)
                            # Update the previous velocity entry with central difference
                            if len(self._velocity_history) >= 1:
                                old_entry = self._velocity_history[-1]
                                self._velocity_history[-1] = (old_entry[0], old_entry[1], old_entry[2], v_central)

                    self._velocity_history.append((frame_idx, v_backward1, v_backward2, None))

                    # Detect DDT: velocity jump > threshold (using first-order backward)
                    if self._ddt_frame_idx is None and len(self._velocity_history) >= 2:
                        prev_vel = self._velocity_history[-2][1]  # First-order backward from prev
                        velocity_jump = v_backward1 - prev_vel
                        if velocity_jump > self.config.ddt_velocity_jump_m_s:
                            self._ddt_frame_idx = frame_idx

        # Build result
        result = FlameDetectionResult(
            frame_idx=frame_idx,
            time_s=time_s,
            frame_subtracted=frame_subtracted,
            frame_diff=frame_diff,
            noise_removed=noise_removed,
            blurred=blurred,
            sobel_output=sobel_output,
            gradient_output=gradient_output,
            pos_min_gradient=pos_min_gradient,
            pos_rightmost_sobel=pos_rightmost_sobel,
            pos_spline_predicted=pos_spline_predicted,
            search_bounds=(search_start, search_end),
            final_position=final_position
        )

        self._detection_results.append(result)
        return result

    def _validate_position(
        self,
        candidate_position: int,
        frame_idx: int
    ) -> Optional[int]:
        """Validate position against tracking constraints."""
        # Get last known valid position
        last_position = None
        last_frame_idx = None
        for f_idx, pos in reversed(self._position_history):
            if pos is not None:
                last_position = pos
                last_frame_idx = f_idx
                break

        if last_position is None:
            return candidate_position

        # Constraint 1: Non-negative velocity (position can only increase)
        if candidate_position < last_position:
            return None

        # Constraint 2: Maximum velocity
        frames_elapsed = frame_idx - last_frame_idx
        if frames_elapsed > 0:
            max_displacement = self._max_displacement_px * frames_elapsed
            actual_displacement = candidate_position - last_position
            if actual_displacement > max_displacement:
                return last_position + max_displacement

        return candidate_position

    def get_spline_curve(self, frame_range: Optional[Tuple[int, int]] = None) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Get spline curve for plotting.

        Args:
            frame_range: Optional (start, end) frame range

        Returns:
            (frames, positions) arrays for plotting, or None if no spline
        """
        if self._spline is None:
            return None

        valid_points = [(f, p) for f, p in self._position_history if p is not None]
        if not valid_points:
            return None

        if frame_range is None:
            f_min = min(f for f, _ in valid_points)
            f_max = max(f for f, _ in valid_points)
        else:
            f_min, f_max = frame_range

        frames = np.linspace(f_min, f_max, 100)
        try:
            positions = self._spline(frames)
            return frames, positions
        except Exception:
            return None

    @property
    def position_history(self) -> List[Tuple[int, Optional[int]]]:
        """Get position history."""
        return self._position_history

    @property
    def last_position(self) -> Optional[int]:
        """Get last detected position."""
        for _, pos in reversed(self._position_history):
            if pos is not None:
                return pos
        return None

    @property
    def last_velocity(self) -> Optional[float]:
        """Get last computed velocity (first-order backward) in m/s."""
        if self._velocity_history:
            return self._velocity_history[-1][1]  # First-order backward
        return None

    @property
    def last_velocities(self) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """Get last computed velocities (v_backward1, v_backward2, v_central) in m/s."""
        if self._velocity_history:
            entry = self._velocity_history[-1]
            return (entry[1], entry[2], entry[3])
        return (None, None, None)

    @property
    def ddt_frame(self) -> Optional[int]:
        """Get frame index where DDT was detected, or None if not detected."""
        return self._ddt_frame_idx

    @property
    def ddt_detected(self) -> bool:
        """Check if DDT has been detected."""
        return self._ddt_frame_idx is not None

    def get_velocity_history(self) -> List[Tuple[int, float, Optional[float], Optional[float]]]:
        """Get full velocity history with all three methods."""
        return list(self._velocity_history)

    def get_pre_ddt_velocities(self) -> List[Tuple[int, float, Optional[float], Optional[float]]]:
        """Get velocity history before DDT."""
        if self._ddt_frame_idx is None:
            return list(self._velocity_history)
        return [entry for entry in self._velocity_history if entry[0] < self._ddt_frame_idx]

    def get_post_ddt_velocities(self) -> List[Tuple[int, float, Optional[float], Optional[float]]]:
        """Get velocity history after DDT (including DDT frame)."""
        if self._ddt_frame_idx is None:
            return []
        return [entry for entry in self._velocity_history if entry[0] >= self._ddt_frame_idx]

    def clear_last_central_difference(self) -> None:
        """Clear the central difference from the second-to-last velocity entry.

        Called when flame exits domain - the central difference for frame n-1
        was computed using x_n which is invalid (at edge), so we must clear it.
        """
        if len(self._velocity_history) >= 2:
            entry = self._velocity_history[-2]
            # Set v_central (index 3) to None
            self._velocity_history[-2] = (entry[0], entry[1], entry[2], None)


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
    result: FlameDetectionResult,
    output_path: Path,
    source_name: str,
    detector: Optional['FlameDetector'] = None
) -> None:
    """
    Save frame image showing all processing steps VERTICALLY stacked.

    Layout (11 rows x 1 column):
    1. BG-Subtracted frame
    2. Frame Diff (current - prior)
    3. Noise Removed (morphological opening)
    4. Gaussian Blur
    5. Sobel Filter
    6. Gradient Filter (np.gradient)
    7. Frame Diff centerline profile
    8. Sobel centerline profile
    9. Gradient centerline profile
    10. Result overlay with all candidates
    11. Position history + spline estimator

    Args:
        frame: Original frame data
        result: FlameDetectionResult with all processing outputs
        output_path: Output directory path
        source_name: Name of video source for filename
        detector: Optional FlameDetector for spline curve
    """
    height, width = frame.shape[:2]
    center_row = height // 2
    x_pixels = np.arange(width)

    # Calculate aspect ratio for images (they're typically wide and short)
    img_aspect = width / height
    img_height = 1.5  # Height in inches for image subplots
    plot_height = 2.5  # Height in inches for line plots

    # Total figure height: 6 images + 6 plots
    total_height = 6 * img_height + 6 * plot_height
    fig_width = 14

    # Create figure with 12 rows, using GridSpec for variable heights
    fig = plt.figure(figsize=(fig_width, total_height))

    # Define height ratios: images get less height, plots get more
    height_ratios = [
        img_height,   # 1. BG-Subtracted
        img_height,   # 2. Frame Diff
        img_height,   # 3. Noise Removed
        img_height,   # 4. Gaussian Blur
        img_height,   # 5. Sobel Filter
        img_height,   # 6. Gradient Filter
        plot_height,  # 7. Frame Diff centerline
        plot_height,  # 8. Sobel centerline
        plot_height,  # 9. Gradient centerline
        img_height,   # 10. Result overlay
        plot_height,  # 11. Position history
        plot_height,  # 12. Velocity history
    ]
    gs = fig.add_gridspec(12, 1, height_ratios=height_ratios, hspace=0.3)
    axes = [fig.add_subplot(gs[i, 0]) for i in range(12)]

    # Helper function to add detection markers on images
    def add_position_markers(ax, show_final=True):
        if result.search_bounds:
            ax.axvline(x=result.search_bounds[0], color='lime', linestyle='--', linewidth=1.5, alpha=0.8)
            ax.axvline(x=result.search_bounds[1], color='lime', linestyle=':', linewidth=1.5, alpha=0.8)
        if result.pos_min_gradient is not None:
            ax.axvline(x=result.pos_min_gradient, color='purple', linestyle='-', linewidth=2, alpha=0.7)
        if result.pos_rightmost_sobel is not None:
            ax.axvline(x=result.pos_rightmost_sobel, color='orange', linestyle='-', linewidth=2, alpha=0.7)
        if show_final and result.final_position is not None:
            ax.axvline(x=result.final_position, color='red', linestyle='-', linewidth=3, alpha=0.9)

    # Get velocity from detector
    velocity_str = ""
    if detector is not None and detector.last_velocity is not None:
        velocity_str = f" | v={detector.last_velocity:.1f} m/s"

    # ========== 1. BG-Subtracted frame ==========
    ax = axes[0]
    ax.imshow(result.frame_subtracted, cmap='gray', aspect='auto')
    ax.axhline(y=center_row, color='cyan', linestyle='--', linewidth=0.5, alpha=0.5)
    add_position_markers(ax)
    ax.set_title(f'1. BG Subtracted - Frame {result.frame_idx} | t={result.time_s*1e6:.1f} µs{velocity_str}', fontsize=10)
    ax.set_ylabel('Y')

    # ========== 2. Frame Diff (current - prior) ==========
    ax = axes[1]
    if result.frame_diff is not None:
        vmax = np.percentile(result.frame_diff, 99) if np.any(result.frame_diff > 0) else 1
        ax.imshow(result.frame_diff, cmap='hot', aspect='auto', vmin=0, vmax=vmax)
        ax.axhline(y=center_row, color='cyan', linestyle='--', linewidth=0.5, alpha=0.5)
        add_position_markers(ax)
    else:
        ax.text(0.5, 0.5, 'No prior frame', ha='center', va='center', transform=ax.transAxes, fontsize=12)
        ax.set_facecolor('lightgray')
    ax.set_title('2. Frame Diff (current - prior)', fontsize=10)
    ax.set_ylabel('Y')

    # ========== 3. Noise Removed (morphological opening) ==========
    ax = axes[2]
    if result.noise_removed is not None:
        vmax = np.percentile(result.noise_removed, 99) if np.any(result.noise_removed > 0) else 1
        ax.imshow(result.noise_removed, cmap='hot', aspect='auto', vmin=0, vmax=vmax)
        ax.axhline(y=center_row, color='cyan', linestyle='--', linewidth=0.5, alpha=0.5)
        add_position_markers(ax)
    else:
        ax.text(0.5, 0.5, 'N/A', ha='center', va='center', transform=ax.transAxes, fontsize=12)
        ax.set_facecolor('lightgray')
    ax.set_title('3. Noise Removed (morphological opening)', fontsize=10)
    ax.set_ylabel('Y')

    # ========== 4. Gaussian Blur ==========
    ax = axes[3]
    if result.blurred is not None:
        vmax = np.percentile(result.blurred, 99) if np.any(result.blurred > 0) else 1
        ax.imshow(result.blurred, cmap='hot', aspect='auto', vmin=0, vmax=vmax)
        ax.axhline(y=center_row, color='cyan', linestyle='--', linewidth=0.5, alpha=0.5)
        add_position_markers(ax)
    else:
        ax.text(0.5, 0.5, 'N/A', ha='center', va='center', transform=ax.transAxes, fontsize=12)
        ax.set_facecolor('lightgray')
    ax.set_title('4. Gaussian Blur', fontsize=10)
    ax.set_ylabel('Y')

    # ========== 5. Sobel Filter ==========
    ax = axes[4]
    if result.sobel_output is not None:
        vmax = np.percentile(np.abs(result.sobel_output), 99) if np.any(result.sobel_output != 0) else 1
        ax.imshow(result.sobel_output, cmap='RdBu', aspect='auto', vmin=-vmax, vmax=vmax)
        ax.axhline(y=center_row, color='black', linestyle='--', linewidth=0.5, alpha=0.5)
        add_position_markers(ax)
    else:
        ax.text(0.5, 0.5, 'N/A', ha='center', va='center', transform=ax.transAxes, fontsize=12)
        ax.set_facecolor('lightgray')
    ax.set_title('5. Sobel Filter (horizontal)', fontsize=10)
    ax.set_ylabel('Y')

    # ========== 6. Gradient Filter ==========
    ax = axes[5]
    if result.gradient_output is not None:
        vmax = np.percentile(np.abs(result.gradient_output), 99) if np.any(result.gradient_output != 0) else 1
        ax.imshow(result.gradient_output, cmap='RdBu', aspect='auto', vmin=-vmax, vmax=vmax)
        ax.axhline(y=center_row, color='black', linestyle='--', linewidth=0.5, alpha=0.5)
        add_position_markers(ax)
    else:
        ax.text(0.5, 0.5, 'N/A', ha='center', va='center', transform=ax.transAxes, fontsize=12)
        ax.set_facecolor('lightgray')
    ax.set_title('6. Gradient Filter (np.gradient)', fontsize=10)
    ax.set_ylabel('Y')

    # ========== 7. Frame Diff Centerline Profile ==========
    ax = axes[6]
    if result.frame_diff is not None:
        diff_line = result.frame_diff[center_row, :]
        ax.plot(x_pixels, diff_line, 'r-', linewidth=1.5, label='Frame Diff')
        ax.fill_between(x_pixels, 0, diff_line, alpha=0.3, color='red')
    # Add detection markers
    if result.search_bounds:
        ax.axvline(x=result.search_bounds[0], color='lime', linestyle='--', linewidth=2, label=f'Search: {result.search_bounds[0]}-{result.search_bounds[1]}')
        ax.axvline(x=result.search_bounds[1], color='lime', linestyle=':', linewidth=2)
    if result.pos_min_gradient is not None:
        ax.axvline(x=result.pos_min_gradient, color='purple', linestyle='-', linewidth=2, label=f'Min Grad: {result.pos_min_gradient}')
    if result.pos_rightmost_sobel is not None:
        ax.axvline(x=result.pos_rightmost_sobel, color='orange', linestyle='-', linewidth=2, label=f'R-Sobel: {result.pos_rightmost_sobel}')
    if result.final_position is not None:
        ax.axvline(x=result.final_position, color='red', linestyle='-', linewidth=3, label=f'FINAL: {result.final_position}')
    ax.set_xlim(0, width)
    ax.set_ylabel('Intensity')
    ax.set_title('7. Frame Diff Centerline', fontsize=10)
    ax.legend(loc='upper right', fontsize=8, ncol=3)
    ax.grid(True, alpha=0.3)

    # ========== 8. Sobel Centerline Profile ==========
    ax = axes[7]
    if result.sobel_output is not None:
        sobel_line = result.sobel_output[center_row, :]
        ax.plot(x_pixels, sobel_line, 'b-', linewidth=1)
        ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
    if result.search_bounds:
        ax.axvline(x=result.search_bounds[0], color='lime', linestyle='--', linewidth=2)
        ax.axvline(x=result.search_bounds[1], color='lime', linestyle=':', linewidth=2)
    if result.pos_rightmost_sobel is not None:
        ax.axvline(x=result.pos_rightmost_sobel, color='orange', linestyle='-', linewidth=2, label=f'Rightmost Sobel: {result.pos_rightmost_sobel}')
    if result.final_position is not None:
        ax.axvline(x=result.final_position, color='red', linestyle='-', linewidth=3, label=f'FINAL: {result.final_position}')
    ax.set_xlim(0, width)
    ax.set_ylabel('Sobel Value')
    ax.set_title('8. Sobel Centerline', fontsize=10)
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)

    # ========== 9. Gradient Centerline Profile ==========
    ax = axes[8]
    if result.gradient_output is not None:
        gradient_line = result.gradient_output[center_row, :]
        ax.plot(x_pixels, gradient_line, 'purple', linewidth=1)
        ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
    if result.search_bounds:
        ax.axvline(x=result.search_bounds[0], color='lime', linestyle='--', linewidth=2)
        ax.axvline(x=result.search_bounds[1], color='lime', linestyle=':', linewidth=2)
    if result.pos_min_gradient is not None:
        ax.axvline(x=result.pos_min_gradient, color='purple', linestyle='-', linewidth=2, label=f'Min Gradient: {result.pos_min_gradient}')
    if result.final_position is not None:
        ax.axvline(x=result.final_position, color='red', linestyle='-', linewidth=3, label=f'FINAL: {result.final_position}')
    ax.set_xlim(0, width)
    ax.set_ylabel('Gradient Value')
    ax.set_title('9. Gradient Centerline (min = leading edge)', fontsize=10)
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)

    # ========== 10. Result Overlay ==========
    ax = axes[9]
    ax.imshow(result.frame_subtracted, cmap='gray', aspect='auto')
    ax.axhline(y=center_row, color='cyan', linestyle='--', linewidth=0.5, alpha=0.5)
    # Show search bounds
    if result.search_bounds:
        ax.axvline(x=result.search_bounds[0], color='lime', linestyle='--', linewidth=2, alpha=0.8)
        ax.axvline(x=result.search_bounds[1], color='lime', linestyle=':', linewidth=2, alpha=0.8)
    # Show all candidate positions as markers
    if result.pos_min_gradient is not None:
        ax.plot(result.pos_min_gradient, center_row, 'p', color='purple', markersize=6, label=f'Min Grad: {result.pos_min_gradient}')
    if result.pos_rightmost_sobel is not None:
        ax.plot(result.pos_rightmost_sobel, center_row, 's', color='orange', markersize=6, label=f'R-Sobel: {result.pos_rightmost_sobel}')
    if result.pos_spline_predicted is not None:
        ax.plot(result.pos_spline_predicted, center_row, '^', color='cyan', markersize=6, label=f'Spline: {result.pos_spline_predicted}')
    if result.final_position is not None:
        ax.plot(result.final_position, center_row, 'o', color='red', markersize=8,
                markeredgecolor='yellow', markeredgewidth=1, label=f'FINAL: {result.final_position}')
    ax.legend(loc='upper right', fontsize=8, ncol=2)
    title_str = f'FINAL: x={result.final_position} px' if result.final_position else 'No detection'
    ax.set_title(f'10. Result: {title_str}{velocity_str}', fontsize=10)
    ax.set_ylabel('Y')

    # ========== 11. Position History + Spline ==========
    ax = axes[10]
    if detector is not None and len(detector.position_history) > 0:
        frames_hist = []
        pos_hist = []
        for f, p in detector.position_history:
            if p is not None:
                frames_hist.append(f)
                pos_hist.append(p)

        if frames_hist:
            ax.scatter(frames_hist, pos_hist, c='blue', s=20, alpha=0.7, label='Detected positions', zorder=3)

            # Plot spline curve
            spline_data = detector.get_spline_curve()
            if spline_data is not None:
                spline_frames, spline_pos = spline_data
                ax.plot(spline_frames, spline_pos, 'g-', linewidth=2, label='Spline estimator', zorder=2)

            # Mark current frame
            ax.axvline(x=result.frame_idx, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
            if result.final_position is not None:
                ax.scatter([result.frame_idx], [result.final_position], c='red', s=60,
                          marker='*', zorder=5, label=f'Current: {result.final_position}')
            if result.pos_spline_predicted is not None:
                ax.scatter([result.frame_idx], [result.pos_spline_predicted], c='cyan', s=40,
                          marker='^', zorder=4, label=f'Spline pred: {result.pos_spline_predicted}')

            ax.legend(loc='upper left', fontsize=8)
    else:
        ax.text(0.5, 0.5, 'No history yet', ha='center', va='center', transform=ax.transAxes, fontsize=12)
    ax.set_ylabel('Position (pixels)')
    ax.set_title('11. Position History + Spline Estimator', fontsize=10)
    ax.grid(True, alpha=0.3)

    # ========== 12. Velocity History (3 methods) ==========
    ax = axes[11]
    if detector is not None and len(detector._velocity_history) > 0:
        # Velocity history: (frame_idx, v_backward1, v_backward2, v_central)
        vel_hist = detector._velocity_history

        # Extract data for each method
        frames = [entry[0] for entry in vel_hist]
        v_backward1 = [entry[1] for entry in vel_hist]  # First-order backward
        v_backward2 = [entry[2] for entry in vel_hist if entry[2] is not None]  # Second-order backward
        frames_b2 = [entry[0] for entry in vel_hist if entry[2] is not None]
        v_central = [entry[3] for entry in vel_hist if entry[3] is not None]  # Central difference
        frames_central = [entry[0] for entry in vel_hist if entry[3] is not None]

        if frames:
            # Plot all three velocity methods with different colors/styles
            ax.plot(frames, v_backward1, 'b-', linewidth=1.5, alpha=0.8,
                   label='1st-order backward')
            if frames_b2 and v_backward2:
                ax.plot(frames_b2, v_backward2, 'g--', linewidth=1.5, alpha=0.8,
                       label='2nd-order backward')
            if frames_central and v_central:
                ax.plot(frames_central, v_central, 'r:', linewidth=2, alpha=0.8,
                       label='2nd-order central')

            ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)

            # Mark DDT if detected
            if detector.ddt_detected:
                ax.axvline(x=detector.ddt_frame, color='magenta', linestyle='--', linewidth=2,
                          label=f'DDT @ frame {detector.ddt_frame}')

            # Mark current velocities
            v1, v2, vc = detector.last_velocities
            legend_str = f'Current: B1={v1:.0f}' if v1 else 'Current: N/A'
            if v2 is not None:
                legend_str += f', B2={v2:.0f}'
            ax.scatter([result.frame_idx], [v1] if v1 else [], c='blue', s=40,
                      marker='*', zorder=5)

            ax.legend(loc='upper left', fontsize=7)
    else:
        ax.text(0.5, 0.5, 'No velocity data yet', ha='center', va='center', transform=ax.transAxes, fontsize=12)
    ax.set_xlabel('Frame Index')
    ax.set_ylabel('Velocity (m/s)')
    ddt_str = f' | DDT @ {detector.ddt_frame}' if detector is not None and detector.ddt_detected else ''
    ax.set_title(f'12. Velocity Comparison{ddt_str}', fontsize=10)
    ax.grid(True, alpha=0.3)

    # Save figure
    output_file = output_path / f"{source_name}-Frame-{result.frame_idx:06d}.png"
    plt.savefig(output_file, dpi=120, bbox_inches='tight')
    plt.close(fig)


def generate_stacked_sequence(
    video,
    frame_indices: List[int],
    background_scalar: float,
    output_path: Path,
    title: str = "",
    show_frame_diff: bool = True,
    figsize_width: float = 10.0
) -> None:
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
        figsize_width: Width of the figure in inches
    """
    n_frames = len(frame_indices)
    height, width = video.frame_shape

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

    for i, frame_idx in enumerate(frame_indices):
        frame = video[frame_idx]
        frame_subtracted = subtract_scalar_background(frame, background_scalar)

        # Compute frame difference
        if prior_frame is not None:
            frame_diff = subtract_prior_frame(frame, prior_frame, threshold=0.0)
        else:
            frame_diff = np.zeros_like(frame)

        # Column 1: Background subtracted
        axes[i, 0].imshow(frame_subtracted, cmap='gray', aspect='equal', vmin=0)
        axes[i, 0].set_ylabel(f'{i+1}', rotation=0, labelpad=20, fontsize=10, fontweight='bold', color='white')
        axes[i, 0].set_xticks([])
        axes[i, 0].set_yticks([])

        # Column 2: Frame difference (if enabled)
        if n_cols > 1:
            axes[i, 1].imshow(frame_diff, cmap='gray', aspect='equal', vmin=0)
            axes[i, 1].set_xticks([])
            axes[i, 1].set_yticks([])

        prior_frame = frame.copy()

    # Remove spacing between subplots
    plt.subplots_adjust(wspace=0.02, hspace=0)

    if title:
        fig.suptitle(title, fontsize=12, fontweight='bold', color='white')

    # Save figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='black', edgecolor='none')
    plt.close(fig)
    print(f"Saved stacked sequence: {output_path}")


def generate_stacked_sequence_single_column(
    video,
    frame_indices: List[int],
    background_scalar: float,
    output_path: Path,
    use_frame_diff: bool = False,
    title: str = "",
    figsize_width: float = 6.0
) -> None:
    """
    Generate single-column stacked frame sequence (compact paper-style).

    Args:
        video: PhotonVideo instance
        frame_indices: List of frame indices to include
        background_scalar: Background value for subtraction
        output_path: Path to save the output image
        use_frame_diff: If True, show frame difference instead of BG subtracted
        title: Optional title
        figsize_width: Width of figure
    """
    n_frames = len(frame_indices)
    height, width = video.frame_shape
    center_row = height // 2

    # Stack all frames into single image
    stacked_height = height * n_frames
    stacked_image = np.zeros((stacked_height, width), dtype=np.float64)

    prior_frame = None

    for i, frame_idx in enumerate(frame_indices):
        frame = video[frame_idx]
        frame_subtracted = subtract_scalar_background(frame, background_scalar)

        # Compute frame difference
        if prior_frame is not None:
            frame_diff = subtract_prior_frame(frame, prior_frame, threshold=0.0)
        else:
            frame_diff = np.zeros_like(frame)

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

    # Add frame numbers and separator lines
    for i, frame_idx in enumerate(frame_indices):
        y_center = i * height + center_row
        ax.text(-width * 0.02, y_center, f'{i+1}', color='white', fontsize=8,
                fontweight='bold', ha='right', va='center')

        # Horizontal separator line
        if i > 0:
            ax.axhline(y=i * height - 0.5, color='white', linewidth=0.5, alpha=0.5)

    ax.set_xlim(-width * 0.05, width)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor('black')

    if title:
        ax.set_title(title, color='white', fontsize=10, fontweight='bold')

    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='black', edgecolor='none')
    plt.close(fig)
    print(f"Saved stacked sequence: {output_path}")


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
            print(f"  Image width: {video.width} px")
            print(f"  Flame exit threshold: {int(video.width * 0.99)} px (99% of width)")

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

        # Create flame detector instance
        detector_config = FlameDetectorConfig(
            gaussian_sigma=1.5,
            morphology_kernel_size=3,
            max_velocity_change_m_s=200.0,
        )
        flame_detector = FlameDetector(
            config=detector_config,
            frame_rate=video.frame_rate,
            calibration_m_per_px=file_calibration
        )

        if is_root:
            print(f"  Flame detector configured: max_velocity={detector_config.max_velocity_change_m_s} m/s")
            print(f"  Max displacement per frame: {flame_detector._max_displacement_px} px")

        # Determine frame indices to process
        if processor is not None:
            frame_indices = processor.distribute_indices(len(video))
        else:
            frame_indices = list(range(len(video)))

        empty_frame_count = 0
        local_results = []

        for frame_idx in frame_indices:
            # Skip explicitly configured frames
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
                # Update detector's prior frame even for empty frames
                flame_detector._prior_frame = frame_subtracted.copy()
                continue

            # Detect flame position using the FlameDetector
            # Returns FlameDetectionResult with all intermediate outputs
            detection_result = flame_detector.detect(
                frame=frame,
                frame_idx=frame_idx,
                background_scalar=background_scalar
            )

            # Save frame image with all processing steps
            save_frame_image(
                frame=frame,
                result=detection_result,
                output_path=frames_output_dir,
                source_name=config.name,
                detector=flame_detector
            )

            # Collect results for output
            flame_position = detection_result.final_position
            velocity = flame_detector.last_velocity

            # Check for flame exiting domain BEFORE recording
            # This prevents recording frames where the flame is at the edge with artificially low velocity
            exit_margin = detector_config.exit_margin_px
            if flame_position is not None and flame_position >= video.width - exit_margin:
                # Clear the central difference from previous frame - it used this bad position
                flame_detector.clear_last_central_difference()
                if is_root:
                    print(f"  Wave exited domain at frame {frame_idx}, position {flame_position} px (not recorded)")
                break

            # Check for sudden velocity drop (>50% decrease) - indicates edge artifact
            # This catches cases where position hasn't quite reached the margin but velocity drops
            # Use first-order backward velocity (v1) for this check
            vel_history = flame_detector.get_velocity_history()
            if velocity is not None and len(vel_history) >= 2:
                prev_v1 = vel_history[-2][1]  # v1 (first-order backward) from previous frame
                if prev_v1 is not None and prev_v1 > 100:  # Only check if we had substantial velocity
                    velocity_drop_ratio = (prev_v1 - velocity) / prev_v1
                    if velocity_drop_ratio > 0.5:  # More than 50% drop
                        # Clear the central difference from previous frame - it used this bad position
                        flame_detector.clear_last_central_difference()
                        if is_root:
                            print(f"  Velocity drop detected at frame {frame_idx}: {prev_v1:.1f} -> {velocity:.1f} m/s (not recorded)")
                        break

            if flame_position is not None:
                pos_m = flame_position * file_calibration + file_position_offset
                is_post_ddt = flame_detector.ddt_detected and frame_idx >= flame_detector.ddt_frame
                # Store position data only - velocities will be merged from velocity_history later
                # This ensures v_central is properly filled in (it's computed one frame later)
                local_results.append((frame_idx, time_s, flame_position, pos_m, is_post_ddt))

            # Report DDT detection
            if flame_detector.ddt_detected and flame_detector.ddt_frame == frame_idx:
                if is_root:
                    vel_str = f"{velocity:.1f}" if velocity is not None else "N/A"
                    print(f"  *** DDT DETECTED at frame {frame_idx}, velocity jump to {vel_str} m/s ***")

            if frame_idx % 50 == 0:
                pos_str = f"{flame_position} px" if flame_position else "None"
                ddt_str = " [POST-DDT]" if flame_detector.ddt_detected else ""
                print(f"[Rank {rank}] Frame {frame_idx}/{len(video)}, position={pos_str}{ddt_str} (skipped {empty_frame_count} empty)")

        if is_root:
            print(f"  Skipped {empty_frame_count} empty/noise-only frames")

        # Gather results if using MPI
        if processor is not None:
            all_results = processor.gather(local_results)
            if is_root:
                flat_results = [item for sublist in all_results for item in sublist]
                flat_results.sort(key=lambda x: x[0])
            else:
                flat_results = []
        else:
            flat_results = local_results

        # Write results (root only)
        if is_root and flat_results:
            # Merge position data with velocity history
            # Results format from loop: (frame, time, px, m, is_post_ddt)
            # Velocity history format: (frame_idx, v1, v2, v_central)
            vel_dict = {entry[0]: (entry[1], entry[2], entry[3])
                       for entry in flame_detector.get_velocity_history()}

            # Merge velocities into results
            merged_results = []
            for f, t, px, m, is_post in flat_results:
                v1, v2, vc = vel_dict.get(f, (None, None, None))
                merged_results.append((f, t, px, m, v1, v2, vc, is_post))

            # Split results by pre-DDT and post-DDT
            pre_ddt_results = [(f, t, px, m, v1, v2, vc) for f, t, px, m, v1, v2, vc, is_post in merged_results if not is_post]
            post_ddt_results = [(f, t, px, m, v1, v2, vc) for f, t, px, m, v1, v2, vc, is_post in merged_results if is_post]

            def write_position_results(data, filepath, label):
                """Helper to write position results to file with all velocity methods."""
                # Header describing velocity extraction methods
                header_lines = [
                    "# Flame Position and Velocity Data",
                    "#",
                    "# Velocity Extraction Methods:",
                    "#   Vel_Backward1: First-order backward difference",
                    "#                  v_n = (x_n - x_{n-1}) / dt",
                    "#                  Evaluates velocity at current time step",
                    "#",
                    "#   Vel_Backward2: Second-order backward difference",
                    "#                  v_n = (3*x_n - 4*x_{n-1} + x_{n-2}) / (2*dt)",
                    "#                  Higher accuracy at current time, requires 3 points",
                    "#",
                    "#   Vel_Central:   Second-order central difference",
                    "#                  v_{n-1} = (x_n - x_{n-2}) / (2*dt)",
                    "#                  Most accurate, but evaluates at PRIOR time step",
                    "#",
                ]

                with open(filepath, 'w') as f:
                    # Write header
                    for line in header_lines:
                        f.write(line + '\n')

                    # Write column headers and data
                    columns = ['#Frame', 'Time_s', 'Position_px', 'Position_m',
                              'Vel_Backward1', 'Vel_Backward2', 'Vel_Central']
                    f.write(' '.join(columns) + '\n')

                    for f_idx, t_s, pixel_pos, p_m, v1, v2, vc in data:
                        row = [
                            str(f_idx),
                            f"{t_s:.9f}",
                            str(pixel_pos),
                            f"{p_m:.9f}",
                            f"{v1:.3f}" if v1 is not None else "",
                            f"{v2:.3f}" if v2 is not None else "",
                            f"{vc:.3f}" if vc is not None else "",
                        ]
                        f.write(' '.join(row) + '\n')

                print(f"  {label}: {filepath} ({len(data)} points)")

            # Write all results
            output_file_all = output_dir / f"{cihx_file.stem}-flame-position.txt"
            all_data = [(f, t, px, m, v1, v2, vc) for f, t, px, m, v1, v2, vc, _ in merged_results]
            write_position_results(all_data, output_file_all, "All results")

            # Write pre-DDT results
            if pre_ddt_results:
                output_file_pre = output_dir / f"{cihx_file.stem}-flame-position-pre-DDT.txt"
                write_position_results(pre_ddt_results, output_file_pre, "Pre-DDT")

            # Write post-DDT results
            if post_ddt_results:
                output_file_post = output_dir / f"{cihx_file.stem}-flame-position-post-DDT.txt"
                write_position_results(post_ddt_results, output_file_post, "Post-DDT")

            # Summary
            print(f"\nResults summary:")
            print(f"  Total detections: {len(flat_results)}")
            print(f"  Pre-DDT: {len(pre_ddt_results)}, Post-DDT: {len(post_ddt_results)}")
            if flame_detector.ddt_detected:
                print(f"  DDT detected at frame {flame_detector.ddt_frame}")
            print(f"  Frame images saved to: {frames_output_dir}")

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
    nova_config.use_frame_diff = True  # Enable prior frame subtraction for visualization
    nova_config.use_absolute_time = True  # Use absolute time from recording start
    nova_config.video_path = "./Nova-Video-Files"
    nova_config.output_dir = "./Processed-Photos/Nova-Output"

    mini_config = VideoSourceConfig(name="Mini")
    mini_config.enabled = True
    mini_config.use_frame_diff = True  # Enable prior frame subtraction for visualization
    mini_config.use_absolute_time = True  # Use absolute time from recording start
    mini_config.video_path = "./Mini-Video-Files"
    mini_config.output_dir = "./Processed-Photos/Mini-Output"

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

    mini_config.file_calibrations = [
        FileCalibration(
            calibration=0.000869565,
            position_offset=0.050237,
            files=["run-1-:run-10-"]
        ),
    ]

    # Process enabled sources
    if nova_config.enabled:
        process_video_source(nova_config, processor)

    if mini_config.enabled:
        process_video_source(mini_config, processor)

    # Synchronize before exit
    if processor is not None:
        processor.barrier()

    if processor is None or processor.is_root:
        print("\nProcessing complete!")


if __name__ == "__main__":
    main()
