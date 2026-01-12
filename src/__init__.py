"""
High-Speed Image Processing Library

Provides tools for loading and processing Photron high-speed camera videos
(CIHX/MRAW files) with support for:
- PIMS-style lazy loading with array indexing
- Configurable metadata extraction
- True video timing with trigger frame support
- Spatial calibration (pixels to physical units)
- Batch processing of multiple videos
- MPI-compatible parallel processing

Example:
    >>> from src import open_video, open_collection
    >>> video = open_video("experiment.cihx")
    >>> print(f"Frames: {len(video)}, FPS: {video.frame_rate}")
    >>> frame = video[0]  # First frame as numpy array

    >>> # With calibration
    >>> video.set_calibration(scale=1.5e-5, units='m')
    >>> position_m = video.pixels_to_physical(500)

    >>> # Batch processing
    >>> collection = open_collection("./videos/", pattern="*.cihx")
    >>> for video in collection:
    ...     process(video)
"""

from .photron import (
    # Core classes
    PhotonVideo,
    VideoCollection,
    MetadataConfig,
    MPIVideoProcessor,

    # Data classes
    SpatialCalibration,
    TimingInfo,

    # Convenience functions
    open_video,
    open_collection,
)

__version__ = "0.1.0"

__all__ = [
    # Core classes
    'PhotonVideo',
    'VideoCollection',
    'MetadataConfig',
    'MPIVideoProcessor',

    # Data classes
    'SpatialCalibration',
    'TimingInfo',

    # Convenience functions
    'open_video',
    'open_collection',
]
