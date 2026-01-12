"""
Photron High-Speed Video Library

A Python library for loading Photron high-speed camera videos (CIHX/MRAW files)
using pyMRAW as the core loading engine.

Example:
    >>> from src.photron import open_video, open_collection
    >>> video = open_video("experiment.cihx")
    >>> print(f"Frames: {len(video)}, FPS: {video.frame_rate}")
    >>> frame = video[0]  # First frame as numpy array
"""

from .video import PhotonVideo, SpatialCalibration, TimingInfo, parse_cihx_xml
from .metadata import MetadataConfig
from .collection import VideoCollection
from .parallel import MPIVideoProcessor

from typing import Optional, Union, List, Set
from pathlib import Path


def open_video(
    filepath: str,
    metadata_fields: Optional[Set[str]] = None,
    trigger_frame: Optional[int] = None,
    calibration: Optional[SpatialCalibration] = None
) -> PhotonVideo:
    """
    Open a single Photron video file.

    Args:
        filepath: Path to .cihx or .mraw file
        metadata_fields: Optional set of metadata field names to expose.
                        Use MetadataConfig presets or custom set.
        trigger_frame: Frame index where trigger occurred (time=0).
                      If None, defaults to 0.
        calibration: SpatialCalibration for pixel-to-physical unit conversion.

    Returns:
        PhotonVideo instance with PIMS-style frame access

    Example:
        >>> video = open_video("experiment.cihx")
        >>> frame = video[0]
        >>> for frame in video[10:20]:
        ...     process(frame)
    """
    return PhotonVideo(
        filepath,
        metadata_fields=metadata_fields,
        trigger_frame=trigger_frame,
        calibration=calibration
    )


def open_collection(
    source: Union[str, List[str]],
    pattern: str = "*.cihx",
    recursive: bool = False,
    metadata_fields: Optional[Set[str]] = None,
    trigger_frame: Optional[int] = None,
    calibration: Optional[SpatialCalibration] = None
) -> VideoCollection:
    """
    Open multiple Photron videos as a collection.

    Args:
        source: Directory path or list of file paths
        pattern: Glob pattern for file matching (if source is directory)
        recursive: Search subdirectories recursively
        metadata_fields: Metadata fields to load for all videos
        trigger_frame: Shared trigger frame for all videos
        calibration: Shared spatial calibration for all videos

    Returns:
        VideoCollection instance for batch processing

    Example:
        >>> collection = open_collection("./videos/", pattern="*.cihx")
        >>> for video in collection:
        ...     print(f"{video.filepath}: {len(video)} frames")
    """
    if isinstance(source, (str, Path)) and Path(source).is_dir():
        return VideoCollection.from_directory(
            source, pattern=pattern, recursive=recursive,
            metadata_fields=metadata_fields,
            trigger_frame=trigger_frame,
            calibration=calibration
        )
    elif isinstance(source, list):
        return VideoCollection.from_files(
            source,
            metadata_fields=metadata_fields,
            trigger_frame=trigger_frame,
            calibration=calibration
        )
    else:
        raise ValueError("source must be a directory path or list of file paths")


__all__ = [
    'PhotonVideo',
    'VideoCollection',
    'MetadataConfig',
    'MPIVideoProcessor',
    'SpatialCalibration',
    'TimingInfo',
    'parse_cihx_xml',
    'open_video',
    'open_collection',
]
