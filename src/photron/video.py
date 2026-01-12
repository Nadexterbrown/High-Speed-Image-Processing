"""
PhotonVideo - Wrapper class for Photron high-speed video files.

Provides PIMS-style lazy loading with array-like indexing for
CIHX/MRAW video files using pyMRAW as the core loading engine.

Includes support for:
- True video timing with trigger frame offset
- Spatial calibration (pixels to physical units)
- Full CIHX XML metadata parsing for accurate timestamps
"""

from typing import Optional, Set, Iterator, Union, Tuple, Dict, Any
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import numpy as np
import xml.etree.ElementTree as ET

try:
    import pyMRAW
except ImportError:
    raise ImportError(
        "pyMRAW is required for loading Photron videos. "
        "Install it with: pip install pyMRAW"
    )

from .metadata import MetadataConfig


def parse_cihx_xml(filepath: Path) -> Dict[str, Any]:
    """
    Parse CIHX file to extract full XML metadata including timing information.

    CIHX files have a binary header followed by XML content. This function
    extracts and parses the XML portion to get timing data that pyMRAW doesn't expose.

    Args:
        filepath: Path to the .cihx file

    Returns:
        Dictionary with parsed metadata including:
        - recording_datetime: datetime of recording start
        - record_rate: frame rate in fps
        - recorded_frame: camera's internal frame number at trigger
        - start_frame: first saved frame index relative to trigger
        - total_frame: total number of frames saved
        - trigger_mode: trigger mode settings
        - irig_enabled: whether IRIG timing was enabled
    """
    result = {
        'recording_datetime': None,
        'record_rate': 0,
        'recorded_frame': 0,
        'start_frame': 0,
        'total_frame': 0,
        'skip_frame': 1,
        'irig_enabled': False,
        'shutter_speed_ns': 0,
    }

    try:
        # Read the file and find the XML content
        with open(filepath, 'rb') as f:
            content = f.read()

        # Find the start of XML (look for <?xml or <cih>)
        xml_start = content.find(b'<?xml')
        if xml_start == -1:
            xml_start = content.find(b'<cih>')

        if xml_start == -1:
            return result

        # Find the end of XML (</cih>)
        xml_end = content.find(b'</cih>', xml_start)
        if xml_end == -1:
            return result

        xml_end += len(b'</cih>')
        xml_content = content[xml_start:xml_end].decode('utf-8', errors='ignore')

        # Parse the XML
        root = ET.fromstring(xml_content)

        # Extract fileInfo (date and time)
        file_info = root.find('fileInfo')
        if file_info is not None:
            date_elem = file_info.find('date')
            time_elem = file_info.find('time')

            if date_elem is not None and time_elem is not None:
                date_str = date_elem.text  # Format: 2023/10/4
                time_str = time_elem.text  # Format: 14:29:21

                try:
                    # Parse the datetime
                    dt_str = f"{date_str} {time_str}"
                    result['recording_datetime'] = datetime.strptime(dt_str, "%Y/%m/%d %H:%M:%S")
                except ValueError:
                    pass

        # Extract frameInfo
        frame_info = root.find('frameInfo')
        if frame_info is not None:
            recorded_frame = frame_info.find('recordedFrame')
            if recorded_frame is not None and recorded_frame.text:
                result['recorded_frame'] = int(recorded_frame.text)

            total_frame = frame_info.find('totalFrame')
            if total_frame is not None and total_frame.text:
                result['total_frame'] = int(total_frame.text)

            start_frame = frame_info.find('startFrame')
            if start_frame is not None and start_frame.text:
                result['start_frame'] = int(start_frame.text)

            skip_frame = frame_info.find('skipFrame')
            if skip_frame is not None and skip_frame.text:
                result['skip_frame'] = int(skip_frame.text)

        # Extract recordInfo
        record_info = root.find('recordInfo')
        if record_info is not None:
            record_rate = record_info.find('recordRate')
            if record_rate is not None and record_rate.text:
                result['record_rate'] = int(record_rate.text)

            shutter_ns = record_info.find('shutterSpeedNsec')
            if shutter_ns is not None and shutter_ns.text:
                result['shutter_speed_ns'] = int(shutter_ns.text)

        # Extract deviceInfo for IRIG
        device_info = root.find('deviceInfo')
        if device_info is not None:
            irig = device_info.find('irig')
            if irig is not None and irig.text:
                result['irig_enabled'] = int(irig.text) != 0

            # Also get record rate from deviceInfo if not in recordInfo
            if result['record_rate'] == 0:
                record_rate = device_info.find('recordRate')
                if record_rate is not None and record_rate.text:
                    result['record_rate'] = int(record_rate.text)

    except Exception as e:
        # If parsing fails, return defaults
        print(f"Warning: Failed to parse CIHX XML: {e}")

    return result


@dataclass
class SpatialCalibration:
    """
    Spatial calibration for converting pixels to physical units.

    Attributes:
        scale: Conversion factor (physical_units / pixel)
        units: Unit name (e.g., 'm', 'mm', 'um')
        origin_x: X origin in pixels (default 0)
        origin_y: Y origin in pixels (default 0)
    """
    scale: float
    units: str = 'm'
    origin_x: float = 0.0
    origin_y: float = 0.0

    def pixels_to_physical(self, pixels: float) -> float:
        """Convert pixel distance to physical units."""
        return pixels * self.scale

    def physical_to_pixels(self, physical: float) -> float:
        """Convert physical distance to pixels."""
        return physical / self.scale

    def x_to_physical(self, x_pixels: float) -> float:
        """Convert x pixel coordinate to physical units."""
        return (x_pixels - self.origin_x) * self.scale

    def y_to_physical(self, y_pixels: float) -> float:
        """Convert y pixel coordinate to physical units."""
        return (y_pixels - self.origin_y) * self.scale


@dataclass
class TimingInfo:
    """
    Timing information for accurate time calculations.

    Supports both relative timing (from trigger) and absolute timing
    (from recording start datetime) when CIHX metadata is available.

    Attributes:
        frame_rate: Recording frame rate in fps
        trigger_frame: Frame index where trigger occurred (time=0 for relative timing)
        start_frame: First saved frame index relative to camera's internal counter
        pre_trigger_frames: Number of frames before trigger
        recording_datetime: Absolute datetime when recording started (from CIHX)
        recorded_frame: Camera's internal frame number at trigger (from CIHX)
        skip_frame: Frame skip factor (1 = no skip)
    """
    frame_rate: int
    trigger_frame: int = 0
    start_frame: int = 0
    pre_trigger_frames: int = 0
    recording_datetime: Optional[datetime] = None
    recorded_frame: int = 0
    skip_frame: int = 1

    def frame_to_time(self, frame_index: int) -> float:
        """
        Convert frame index to time in seconds (relative to trigger).

        Time is relative to trigger frame (trigger_frame = time 0).
        Negative times indicate pre-trigger frames.
        """
        if self.frame_rate <= 0:
            return 0.0
        return (frame_index - self.trigger_frame) / self.frame_rate

    def frame_to_absolute_time(self, frame_index: int) -> float:
        """
        Convert frame index to absolute time in seconds from recording start.

        This calculates time based on the frame's position relative to the
        camera's internal frame counter at recording start.

        Args:
            frame_index: Frame index (0-based in the saved video)

        Returns:
            Time in seconds from recording start
        """
        if self.frame_rate <= 0:
            return 0.0

        # Calculate: time = (start_frame + frame_index * skip_frame) / frame_rate
        # Where start_frame is the offset from trigger
        absolute_frame = self.start_frame + (frame_index * self.skip_frame)
        return absolute_frame / self.frame_rate

    def frame_to_datetime(self, frame_index: int) -> Optional[datetime]:
        """
        Convert frame index to absolute datetime.

        Args:
            frame_index: Frame index (0-based in the saved video)

        Returns:
            Datetime for this frame, or None if recording_datetime not available
        """
        if self.recording_datetime is None or self.frame_rate <= 0:
            return None

        time_offset = self.frame_to_absolute_time(frame_index)
        return self.recording_datetime + timedelta(seconds=time_offset)

    def time_to_frame(self, time_seconds: float) -> int:
        """
        Convert time in seconds to frame index.

        Time is relative to trigger frame.
        """
        if self.frame_rate <= 0:
            return 0
        return int(time_seconds * self.frame_rate) + self.trigger_frame

    @property
    def has_absolute_timing(self) -> bool:
        """Check if absolute timing information is available."""
        return self.recording_datetime is not None and self.frame_rate > 0


class PhotonVideo:
    """
    Wrapper around pyMRAW with PIMS-style access to Photron video files.

    Provides lazy loading via memory-mapped arrays, array-like frame access,
    configurable metadata exposure, true video timing, and spatial calibration.

    Attributes:
        filepath: Path to the video file
        metadata: Filtered metadata dictionary
        frame_rate: Recording frame rate in fps
        frame_shape: (height, width) of each frame
        dtype: NumPy dtype of pixel values
        timing: TimingInfo for accurate time calculations
        calibration: SpatialCalibration for pixel-to-physical conversion

    Example:
        >>> video = PhotonVideo("experiment.cihx",
        ...                     trigger_frame=100,
        ...                     calibration=SpatialCalibration(scale=1.5e-5, units='m'))
        >>> print(f"Frames: {len(video)}, FPS: {video.frame_rate}")
        >>> frame = video[0]              # First frame
        >>> time = video.get_time(0)      # Time relative to trigger (negative if pre-trigger)
        >>> position_m = video.calibration.pixels_to_physical(500)  # Convert 500 pixels to meters
    """

    def __init__(
        self,
        filepath: str,
        metadata_fields: Optional[Set[str]] = None,
        validate: bool = True,
        trigger_frame: Optional[int] = None,
        calibration: Optional[SpatialCalibration] = None
    ):
        """
        Initialize PhotonVideo from a CIHX or MRAW file.

        Args:
            filepath: Path to .cihx, .cih, or .mraw file
            metadata_fields: Set of metadata field names to expose.
                           If None, uses MetadataConfig.for_processing()
            validate: Validate file exists before loading
            trigger_frame: Frame index where trigger occurred (time=0).
                          If None, attempts to read from metadata or defaults to 0.
            calibration: SpatialCalibration for pixel-to-physical unit conversion.
                        If None, no calibration is applied.

        Raises:
            FileNotFoundError: If file does not exist and validate=True
            ValueError: If file format is not supported
        """
        self._filepath = Path(filepath)

        if validate and not self._filepath.exists():
            raise FileNotFoundError(f"Video file not found: {filepath}")

        # Load video using pyMRAW
        self._images, self._raw_info = pyMRAW.load_video(str(self._filepath))

        # Configure metadata filtering
        if metadata_fields is None:
            self._metadata_config = MetadataConfig.for_processing()
        else:
            self._metadata_config = MetadataConfig(fields=metadata_fields)

        self._metadata = self._metadata_config.filter_metadata(self._raw_info)

        # Cache commonly accessed properties
        self._len = int(self._raw_info.get('Total Frame', len(self._images)))
        self._frame_shape = (
            int(self._raw_info.get('Image Height', self._images.shape[1])),
            int(self._raw_info.get('Image Width', self._images.shape[2]))
        )
        self._dtype = self._images.dtype

        # Parse CIHX XML for full timing metadata (if available)
        self._cihx_metadata = {}
        if self._filepath.suffix.lower() == '.cihx':
            self._cihx_metadata = parse_cihx_xml(self._filepath)

        # Initialize timing info with both pyMRAW and CIHX data
        # Prefer CIHX data when available as it's more complete
        if self._cihx_metadata.get('record_rate', 0) > 0:
            frame_rate = self._cihx_metadata['record_rate']
        else:
            frame_rate = int(self._raw_info.get('Record Rate(fps)', 0))

        # Use CIHX start_frame if CIHX was successfully parsed (record_rate > 0 indicates valid parse)
        # start_frame can be 0, positive, or negative (pre-trigger)
        if self._cihx_metadata.get('record_rate', 0) > 0:
            start_frame = self._cihx_metadata.get('start_frame', 0)
        else:
            start_frame = int(self._raw_info.get('Start Frame', 0))

        skip_frame = self._cihx_metadata.get('skip_frame', 1)
        recorded_frame = self._cihx_metadata.get('recorded_frame', 0)
        recording_datetime = self._cihx_metadata.get('recording_datetime')

        # Determine trigger frame
        if trigger_frame is not None:
            trig_frame = trigger_frame
        else:
            # Try to get from metadata, default to 0
            trig_frame = int(self._raw_info.get('Trigger Frame', 0))

        self._timing = TimingInfo(
            frame_rate=frame_rate,
            trigger_frame=trig_frame,
            start_frame=start_frame,
            pre_trigger_frames=trig_frame,
            recording_datetime=recording_datetime,
            recorded_frame=recorded_frame,
            skip_frame=skip_frame
        )

        # Initialize spatial calibration
        self._calibration = calibration

    @property
    def filepath(self) -> Path:
        """Path to the video file."""
        return self._filepath

    @property
    def metadata(self) -> dict:
        """Filtered metadata dictionary."""
        return self._metadata.copy()

    @property
    def raw_metadata(self) -> dict:
        """Complete raw metadata from pyMRAW."""
        return self._raw_info.copy()

    @property
    def cihx_metadata(self) -> Dict[str, Any]:
        """
        CIHX XML metadata including timing information.

        Returns dictionary with:
        - recording_datetime: datetime of recording start
        - record_rate: frame rate in fps
        - recorded_frame: camera's internal frame at trigger
        - start_frame: first saved frame offset
        - total_frame: total frames saved
        - skip_frame: frame skip factor
        - irig_enabled: whether IRIG was enabled
        - shutter_speed_ns: shutter speed in nanoseconds
        """
        return self._cihx_metadata.copy()

    @property
    def recording_datetime(self) -> Optional[datetime]:
        """Datetime when recording started (from CIHX metadata)."""
        return self._timing.recording_datetime

    @property
    def has_absolute_timing(self) -> bool:
        """Check if absolute timing information is available from CIHX."""
        return self._timing.has_absolute_timing

    @property
    def frame_rate(self) -> int:
        """Recording frame rate in fps."""
        return self._timing.frame_rate

    @property
    def fps(self) -> int:
        """Alias for frame_rate."""
        return self.frame_rate

    @property
    def frame_shape(self) -> Tuple[int, int]:
        """(height, width) of each frame."""
        return self._frame_shape

    @property
    def height(self) -> int:
        """Frame height in pixels."""
        return self._frame_shape[0]

    @property
    def width(self) -> int:
        """Frame width in pixels."""
        return self._frame_shape[1]

    @property
    def dtype(self) -> np.dtype:
        """NumPy dtype of pixel values."""
        return self._dtype

    @property
    def bit_depth(self) -> int:
        """Effective bit depth of pixel data."""
        return int(self._raw_info.get('EffectiveBit Depth', 16))

    @property
    def shutter_speed(self) -> float:
        """Shutter speed in seconds."""
        return float(self._raw_info.get('Shutter Speed(s)', 0.0))

    @property
    def exposure_time(self) -> float:
        """Alias for shutter_speed."""
        return self.shutter_speed

    @property
    def duration(self) -> float:
        """Total video duration in seconds."""
        if self.frame_rate > 0:
            return len(self) / self.frame_rate
        return 0.0

    @property
    def timing(self) -> TimingInfo:
        """Timing information for accurate time calculations."""
        return self._timing

    @property
    def trigger_frame(self) -> int:
        """Frame index where trigger occurred (time=0)."""
        return self._timing.trigger_frame

    @property
    def calibration(self) -> Optional[SpatialCalibration]:
        """Spatial calibration for pixel-to-physical conversion."""
        return self._calibration

    @calibration.setter
    def calibration(self, value: Optional[SpatialCalibration]) -> None:
        """Set spatial calibration."""
        self._calibration = value

    def set_calibration(
        self,
        scale: float,
        units: str = 'm',
        origin_x: float = 0.0,
        origin_y: float = 0.0
    ) -> 'PhotonVideo':
        """
        Set spatial calibration.

        Args:
            scale: Conversion factor (physical_units / pixel)
            units: Unit name (e.g., 'm', 'mm', 'um')
            origin_x: X origin in pixels
            origin_y: Y origin in pixels

        Returns:
            self for method chaining
        """
        self._calibration = SpatialCalibration(
            scale=scale,
            units=units,
            origin_x=origin_x,
            origin_y=origin_y
        )
        return self

    def set_trigger_frame(self, frame_index: int) -> 'PhotonVideo':
        """
        Set the trigger frame (time=0 reference).

        Args:
            frame_index: Frame index where trigger occurred

        Returns:
            self for method chaining
        """
        self._timing = TimingInfo(
            frame_rate=self._timing.frame_rate,
            trigger_frame=frame_index,
            start_frame=self._timing.start_frame,
            pre_trigger_frames=frame_index,
            recording_datetime=self._timing.recording_datetime,
            recorded_frame=self._timing.recorded_frame,
            skip_frame=self._timing.skip_frame
        )
        return self

    def __len__(self) -> int:
        """Return total number of frames."""
        return self._len

    def __getitem__(self, key: Union[int, slice]) -> np.ndarray:
        """
        Get frame(s) by index or slice.

        Args:
            key: Integer index or slice object

        Returns:
            Single frame as numpy array, or array of frames for slices

        Example:
            >>> frame = video[0]        # First frame
            >>> frame = video[-1]       # Last frame
            >>> frames = video[10:20]   # Frames 10-19
            >>> frames = video[::10]    # Every 10th frame
        """
        if isinstance(key, int):
            if key < 0:
                key = self._len + key
            if not 0 <= key < self._len:
                raise IndexError(f"Frame index {key} out of range [0, {self._len})")
            return np.array(self._images[key])
        elif isinstance(key, slice):
            return np.array(self._images[key])
        else:
            raise TypeError(f"Indices must be integers or slices, not {type(key).__name__}")

    def __iter__(self) -> Iterator[np.ndarray]:
        """Iterate over all frames."""
        for i in range(self._len):
            yield np.array(self._images[i])

    def get_time(self, frame_index: int) -> float:
        """
        Convert frame index to timestamp in seconds.

        Time is relative to the trigger frame:
        - Frames before trigger have negative times
        - Trigger frame has time = 0
        - Frames after trigger have positive times

        Args:
            frame_index: Frame index (0-based)

        Returns:
            Time in seconds relative to trigger frame
        """
        return self._timing.frame_to_time(frame_index)

    def get_absolute_time(self, frame_index: int) -> float:
        """
        Get absolute time from start of saved recording.

        Uses CIHX metadata (start_frame, skip_frame) when available
        to calculate accurate timing that matches PFV4.

        Args:
            frame_index: Frame index (0-based in saved video)

        Returns:
            Time in seconds from recording start
        """
        return self._timing.frame_to_absolute_time(frame_index)

    def get_datetime(self, frame_index: int) -> Optional[datetime]:
        """
        Get absolute datetime for a specific frame.

        Requires CIHX metadata with recording_datetime to be available.

        Args:
            frame_index: Frame index (0-based in saved video)

        Returns:
            Datetime for this frame, or None if not available
        """
        return self._timing.frame_to_datetime(frame_index)

    def get_frame_at_time(self, time_seconds: float) -> np.ndarray:
        """
        Get frame closest to the specified time.

        Time is relative to trigger frame (negative = pre-trigger).

        Args:
            time_seconds: Time in seconds relative to trigger

        Returns:
            Frame data as numpy array
        """
        if self.frame_rate <= 0:
            raise ValueError("Cannot get frame by time: frame rate is 0")

        index = self._timing.time_to_frame(time_seconds)
        index = max(0, min(index, self._len - 1))
        return self[index]

    def get_time_range(self, start: float, end: float) -> np.ndarray:
        """
        Get frames within a time range.

        Times are relative to trigger frame.

        Args:
            start: Start time in seconds (relative to trigger)
            end: End time in seconds (relative to trigger)

        Returns:
            Array of frames within the time range
        """
        if self.frame_rate <= 0:
            raise ValueError("Cannot get frames by time: frame rate is 0")

        start_idx = self._timing.time_to_frame(start)
        end_idx = self._timing.time_to_frame(end) + 1

        # Clamp to valid range
        start_idx = max(0, start_idx)
        end_idx = min(self._len, end_idx)

        return self[start_idx:end_idx]

    def pixels_to_physical(self, pixels: float) -> float:
        """
        Convert pixel distance to physical units.

        Args:
            pixels: Distance in pixels

        Returns:
            Distance in physical units

        Raises:
            ValueError: If no calibration is set
        """
        if self._calibration is None:
            raise ValueError("No calibration set. Use set_calibration() first.")
        return self._calibration.pixels_to_physical(pixels)

    def physical_to_pixels(self, physical: float) -> float:
        """
        Convert physical distance to pixels.

        Args:
            physical: Distance in physical units

        Returns:
            Distance in pixels

        Raises:
            ValueError: If no calibration is set
        """
        if self._calibration is None:
            raise ValueError("No calibration set. Use set_calibration() first.")
        return self._calibration.physical_to_pixels(physical)

    def to_float64(self, normalize: bool = True) -> 'PhotonVideoFloat64':
        """
        Create a view that returns frames as float64.

        Args:
            normalize: If True, normalize to [0, 1] range

        Returns:
            PhotonVideoFloat64 wrapper instance
        """
        return PhotonVideoFloat64(self, normalize=normalize)

    def close(self) -> None:
        """
        Release resources (memory map).

        Note: After closing, the video object should not be used.
        """
        if hasattr(self, '_images') and self._images is not None:
            del self._images
            self._images = None

    def __enter__(self) -> 'PhotonVideo':
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.close()

    def __repr__(self) -> str:
        return (
            f"<PhotonVideo '{self._filepath.name}' "
            f"frames={len(self)} shape={self.frame_shape} "
            f"dtype={self.dtype} fps={self.frame_rate}>"
        )


class PhotonVideoFloat64:
    """
    Wrapper that returns frames as float64 arrays.

    Optionally normalizes pixel values to [0, 1] range based on bit depth.
    """

    def __init__(self, video: PhotonVideo, normalize: bool = True):
        """
        Initialize float64 view.

        Args:
            video: PhotonVideo instance to wrap
            normalize: If True, normalize to [0, 1] based on bit depth
        """
        self._video = video
        self._normalize = normalize
        self._max_value = (2 ** video.bit_depth) - 1

    def __len__(self) -> int:
        return len(self._video)

    def __getitem__(self, key: Union[int, slice]) -> np.ndarray:
        frame = self._video[key]
        result = frame.astype(np.float64)
        if self._normalize:
            result /= self._max_value
        return result

    def __iter__(self) -> Iterator[np.ndarray]:
        for frame in self._video:
            result = frame.astype(np.float64)
            if self._normalize:
                result /= self._max_value
            yield result

    @property
    def frame_rate(self) -> int:
        return self._video.frame_rate

    @property
    def frame_shape(self) -> Tuple[int, int]:
        return self._video.frame_shape
