"""
VideoCollection - Batch processing for multiple Photron video files.

Provides unified access to multiple video files with consistent
frame indexing across the collection.
"""

from typing import List, Optional, Iterator, Union, Callable, Any, Set, Tuple
from pathlib import Path
import numpy as np

from .video import PhotonVideo, SpatialCalibration


class VideoCollection:
    """
    Collection of multiple Photron videos for batch processing.

    Provides unified access to multiple video files with:
    - Iteration over videos
    - Global frame indexing across collection
    - Batch metadata access

    Example:
        >>> collection = VideoCollection.from_directory("./videos/", pattern="*.cihx")
        >>> for video in collection:
        ...     print(f"{video.filepath.name}: {len(video)} frames @ {video.frame_rate} fps")

        >>> # Global frame access
        >>> frame = collection.get_global_frame(1000)

        >>> # Map function over all frames
        >>> results = collection.map_frames(process_func)
    """

    def __init__(
        self,
        videos: List[PhotonVideo],
        metadata_fields: Optional[Set[str]] = None
    ):
        """
        Initialize collection with list of video readers.

        Args:
            videos: List of PhotonVideo instances
            metadata_fields: Shared metadata configuration
        """
        self._videos = videos
        self._metadata_fields = metadata_fields
        self._build_index()

    def _build_index(self) -> None:
        """Build cumulative frame index for global addressing."""
        self._cumulative_lengths = [0]
        for video in self._videos:
            self._cumulative_lengths.append(
                self._cumulative_lengths[-1] + len(video)
            )
        self._total_frames = self._cumulative_lengths[-1]

    @classmethod
    def from_directory(
        cls,
        directory: Union[str, Path],
        pattern: str = "*.cihx",
        recursive: bool = False,
        metadata_fields: Optional[Set[str]] = None,
        calibration: Optional[SpatialCalibration] = None,
        trigger_frame: Optional[int] = None
    ) -> 'VideoCollection':
        """
        Create collection from all matching files in directory.

        Args:
            directory: Path to search for video files
            pattern: Glob pattern for file matching (e.g., "*.cihx", "*.mraw")
            recursive: Search subdirectories recursively
            metadata_fields: Metadata fields to load for all videos
            calibration: Shared spatial calibration for all videos
            trigger_frame: Shared trigger frame for all videos

        Returns:
            VideoCollection instance

        Example:
            >>> collection = VideoCollection.from_directory(
            ...     "./experiment_data/",
            ...     pattern="*.cihx",
            ...     recursive=True,
            ...     calibration=SpatialCalibration(scale=1.5e-5, units='m')
            ... )
        """
        path = Path(directory)
        if not path.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")

        if recursive:
            files = sorted(path.rglob(pattern))
        else:
            files = sorted(path.glob(pattern))

        videos = []
        for f in files:
            try:
                video = PhotonVideo(
                    str(f),
                    metadata_fields=metadata_fields,
                    calibration=calibration,
                    trigger_frame=trigger_frame
                )
                videos.append(video)
            except Exception as e:
                # Log warning but continue with other files
                print(f"Warning: Could not load {f}: {e}")

        return cls(videos, metadata_fields)

    @classmethod
    def from_files(
        cls,
        filepaths: List[Union[str, Path]],
        metadata_fields: Optional[Set[str]] = None,
        calibration: Optional[SpatialCalibration] = None,
        trigger_frame: Optional[int] = None
    ) -> 'VideoCollection':
        """
        Create collection from explicit list of files.

        Args:
            filepaths: List of paths to video files
            metadata_fields: Metadata fields to load
            calibration: Shared spatial calibration
            trigger_frame: Shared trigger frame

        Returns:
            VideoCollection instance
        """
        videos = []
        for fp in filepaths:
            video = PhotonVideo(
                str(fp),
                metadata_fields=metadata_fields,
                calibration=calibration,
                trigger_frame=trigger_frame
            )
            videos.append(video)
        return cls(videos, metadata_fields)

    def __len__(self) -> int:
        """Return number of videos in collection."""
        return len(self._videos)

    def __iter__(self) -> Iterator[PhotonVideo]:
        """Iterate over videos in collection."""
        return iter(self._videos)

    def __getitem__(self, idx: int) -> PhotonVideo:
        """Get video by index."""
        return self._videos[idx]

    @property
    def videos(self) -> List[PhotonVideo]:
        """List of all videos in collection."""
        return self._videos.copy()

    @property
    def total_frames(self) -> int:
        """Total frames across all videos."""
        return self._total_frames

    @property
    def filepaths(self) -> List[Path]:
        """List of all video file paths."""
        return [v.filepath for v in self._videos]

    def get_global_frame(self, global_idx: int) -> np.ndarray:
        """
        Get frame by global index across collection.

        Args:
            global_idx: Frame index in range [0, total_frames)

        Returns:
            Frame data as numpy array

        Example:
            >>> # Get frame 1000 from the combined collection
            >>> frame = collection.get_global_frame(1000)
        """
        video_idx, local_idx = self._resolve_global_index(global_idx)
        return self._videos[video_idx][local_idx]

    def get_global_time(self, global_idx: int) -> float:
        """
        Get time for a global frame index.

        Note: Returns time relative to the video's trigger frame.

        Args:
            global_idx: Global frame index

        Returns:
            Time in seconds relative to video's trigger frame
        """
        video_idx, local_idx = self._resolve_global_index(global_idx)
        return self._videos[video_idx].get_time(local_idx)

    def _resolve_global_index(self, global_idx: int) -> Tuple[int, int]:
        """
        Convert global frame index to (video_idx, local_frame_idx).

        Args:
            global_idx: Global frame index

        Returns:
            Tuple of (video_index, local_frame_index)

        Raises:
            IndexError: If global_idx is out of range
        """
        if global_idx < 0:
            global_idx = self._total_frames + global_idx

        if global_idx < 0 or global_idx >= self._total_frames:
            raise IndexError(
                f"Global frame index {global_idx} out of range [0, {self._total_frames})"
            )

        for i in range(len(self._cumulative_lengths) - 1):
            if global_idx < self._cumulative_lengths[i + 1]:
                local_idx = global_idx - self._cumulative_lengths[i]
                return i, local_idx

        raise IndexError(f"Global frame index {global_idx} out of range")

    def global_to_local(self, global_idx: int) -> Tuple[int, int]:
        """
        Convert global frame index to (video_idx, local_frame_idx).

        Public wrapper for _resolve_global_index.

        Args:
            global_idx: Global frame index

        Returns:
            Tuple of (video_index, local_frame_index)
        """
        return self._resolve_global_index(global_idx)

    def local_to_global(self, video_idx: int, local_idx: int) -> int:
        """
        Convert local frame index to global index.

        Args:
            video_idx: Video index in collection
            local_idx: Frame index within video

        Returns:
            Global frame index
        """
        if video_idx < 0 or video_idx >= len(self._videos):
            raise IndexError(f"Video index {video_idx} out of range")

        return self._cumulative_lengths[video_idx] + local_idx

    def map_frames(
        self,
        func: Callable[[np.ndarray, int, int], Any],
        frame_indices: Optional[List[int]] = None,
        video_indices: Optional[List[int]] = None
    ) -> List[Any]:
        """
        Apply function to frames across collection.

        Args:
            func: Function taking (frame_data, video_idx, frame_idx) -> result
            frame_indices: Optional list of global frame indices to process.
                          If None, processes all frames.
            video_indices: Optional list of video indices to process.
                          If None, processes all videos.

        Returns:
            List of results from func

        Example:
            >>> def detect_edge(frame, video_idx, frame_idx):
            ...     return find_flame_position(frame)
            >>> positions = collection.map_frames(detect_edge)
        """
        results = []

        if frame_indices is not None:
            # Process specific global frame indices
            for global_idx in frame_indices:
                video_idx, local_idx = self._resolve_global_index(global_idx)
                frame = self._videos[video_idx][local_idx]
                result = func(frame, video_idx, local_idx)
                results.append(result)
        else:
            # Process all frames (or subset of videos)
            videos_to_process = (
                video_indices if video_indices is not None
                else range(len(self._videos))
            )

            for video_idx in videos_to_process:
                video = self._videos[video_idx]
                for frame_idx in range(len(video)):
                    frame = video[frame_idx]
                    result = func(frame, video_idx, frame_idx)
                    results.append(result)

        return results

    def iter_frames(self) -> Iterator[Tuple[np.ndarray, int, int, float]]:
        """
        Iterate over all frames in collection.

        Yields:
            Tuple of (frame_data, video_idx, frame_idx, time)
        """
        for video_idx, video in enumerate(self._videos):
            for frame_idx in range(len(video)):
                frame = video[frame_idx]
                time = video.get_time(frame_idx)
                yield frame, video_idx, frame_idx, time

    def set_calibration_all(
        self,
        scale: float,
        units: str = 'm',
        origin_x: float = 0.0,
        origin_y: float = 0.0
    ) -> 'VideoCollection':
        """
        Set the same spatial calibration for all videos.

        Args:
            scale: Conversion factor (physical_units / pixel)
            units: Unit name (e.g., 'm', 'mm', 'um')
            origin_x: X origin in pixels
            origin_y: Y origin in pixels

        Returns:
            self for method chaining
        """
        for video in self._videos:
            video.set_calibration(scale, units, origin_x, origin_y)
        return self

    def set_trigger_frame_all(self, frame_index: int) -> 'VideoCollection':
        """
        Set the same trigger frame for all videos.

        Args:
            frame_index: Frame index where trigger occurred

        Returns:
            self for method chaining
        """
        for video in self._videos:
            video.set_trigger_frame(frame_index)
        return self

    def summary(self) -> str:
        """Return summary string of collection."""
        lines = [
            f"VideoCollection: {len(self)} videos, {self.total_frames} total frames",
            "-" * 60
        ]
        for i, video in enumerate(self._videos):
            lines.append(
                f"  [{i}] {video.filepath.name}: "
                f"{len(video)} frames @ {video.frame_rate} fps"
            )
        return "\n".join(lines)

    def close_all(self) -> None:
        """Close all video readers."""
        for video in self._videos:
            video.close()

    def __enter__(self) -> 'VideoCollection':
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.close_all()

    def __repr__(self) -> str:
        return (
            f"<VideoCollection videos={len(self)} "
            f"total_frames={self.total_frames}>"
        )
