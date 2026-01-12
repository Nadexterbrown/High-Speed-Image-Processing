"""
MPI-compatible parallel processing for Photron video collections.

Provides utilities for distributing video processing across
multiple MPI ranks, compatible with the existing mpi4py workflow.
"""

from typing import List, Optional, Callable, Any, TypeVar, Tuple, Union
import numpy as np

from .collection import VideoCollection

T = TypeVar('T')


class MPIVideoProcessor:
    """
    MPI-compatible processor for distributed video processing.

    Integrates with mpi4py to distribute frame processing across
    multiple ranks. Falls back to serial processing if MPI is
    not available or not initialized.

    Example:
        >>> from mpi4py import MPI
        >>> from src.photron import open_collection, MPIVideoProcessor

        >>> processor = MPIVideoProcessor(MPI.COMM_WORLD)
        >>> collection = open_collection("./videos/")

        >>> def analyze_frame(frame, global_idx):
        ...     return detect_flame_edge(frame)

        >>> results = processor.process_collection(collection, analyze_frame)
        >>> if processor.is_root:
        ...     save_results(results)
    """

    def __init__(self, comm=None):
        """
        Initialize MPI processor.

        Args:
            comm: MPI communicator (e.g., MPI.COMM_WORLD).
                  If None, operates in serial mode.
        """
        if comm is not None:
            self._comm = comm
            self._rank = comm.Get_rank()
            self._size = comm.Get_size()
        else:
            self._comm = None
            self._rank = 0
            self._size = 1

    @property
    def rank(self) -> int:
        """Current MPI rank (0 for serial mode)."""
        return self._rank

    @property
    def size(self) -> int:
        """Total number of MPI ranks (1 for serial mode)."""
        return self._size

    @property
    def is_root(self) -> bool:
        """Check if this is the root process (rank 0)."""
        return self._rank == 0

    @property
    def is_parallel(self) -> bool:
        """Check if running in parallel mode."""
        return self._comm is not None and self._size > 1

    def distribute_indices(
        self,
        total_count: int,
        distribution: str = "round_robin"
    ) -> List[int]:
        """
        Get indices assigned to this rank.

        Args:
            total_count: Total number of items to distribute
            distribution: Distribution strategy:
                - "round_robin": Interleaved distribution (default)
                - "contiguous": Block distribution

        Returns:
            List of indices for this rank to process

        Example:
            >>> # With 4 ranks and 100 frames:
            >>> # Rank 0 gets [0, 4, 8, 12, ...]
            >>> # Rank 1 gets [1, 5, 9, 13, ...]
            >>> indices = processor.distribute_indices(100)
        """
        if distribution == "round_robin":
            return [i for i in range(total_count) if i % self._size == self._rank]
        elif distribution == "contiguous":
            chunk_size = total_count // self._size
            remainder = total_count % self._size

            # Distribute remainder across first 'remainder' ranks
            if self._rank < remainder:
                start = self._rank * (chunk_size + 1)
                end = start + chunk_size + 1
            else:
                start = remainder * (chunk_size + 1) + (self._rank - remainder) * chunk_size
                end = start + chunk_size

            return list(range(start, end))
        else:
            raise ValueError(f"Unknown distribution strategy: {distribution}")

    def process_collection(
        self,
        collection: VideoCollection,
        process_func: Callable[[np.ndarray, int], T],
        gather_results: bool = True,
        distribution: str = "round_robin"
    ) -> Optional[List[Tuple[int, T]]]:
        """
        Process video collection in parallel.

        Distributes frames across MPI ranks and optionally gathers
        results back to root.

        Args:
            collection: VideoCollection to process
            process_func: Function taking (frame_data, global_frame_idx) -> result
            gather_results: If True, gather all results to root rank.
                           If False, each rank keeps its own results.
            distribution: Distribution strategy ("round_robin" or "contiguous")

        Returns:
            On root (if gather_results=True): List of (global_idx, result) tuples,
                sorted by global_idx.
            On non-root (if gather_results=True): None
            On all ranks (if gather_results=False): Local results as list of
                (global_idx, result) tuples.

        Example:
            >>> def analyze_frame(frame, global_idx):
            ...     edge_pos = detect_edge(frame)
            ...     return edge_pos

            >>> results = processor.process_collection(collection, analyze_frame)
            >>> if processor.is_root:
            ...     for idx, pos in results:
            ...         print(f"Frame {idx}: edge at {pos}")
        """
        my_indices = self.distribute_indices(collection.total_frames, distribution)

        local_results = []
        for global_idx in my_indices:
            frame = collection.get_global_frame(global_idx)
            result = process_func(frame, global_idx)
            local_results.append((global_idx, result))

        if gather_results and self._comm is not None:
            all_results = self._comm.gather(local_results, root=0)
            if self.is_root:
                # Flatten and sort by frame index
                flat = [item for sublist in all_results for item in sublist]
                flat.sort(key=lambda x: x[0])
                return flat
            return None

        return local_results

    def process_videos(
        self,
        collection: VideoCollection,
        process_video_func: Callable[['PhotonVideo', int], T],
        gather_results: bool = True
    ) -> Optional[List[Tuple[int, T]]]:
        """
        Process entire videos in parallel (one video per task).

        Useful when processing must be done per-video rather than per-frame.

        Args:
            collection: VideoCollection to process
            process_video_func: Function taking (PhotonVideo, video_idx) -> result
            gather_results: If True, gather results to root

        Returns:
            Similar to process_collection, but indexed by video_idx
        """
        my_video_indices = self.distribute_indices(len(collection))

        local_results = []
        for video_idx in my_video_indices:
            video = collection[video_idx]
            result = process_video_func(video, video_idx)
            local_results.append((video_idx, result))

        if gather_results and self._comm is not None:
            all_results = self._comm.gather(local_results, root=0)
            if self.is_root:
                flat = [item for sublist in all_results for item in sublist]
                flat.sort(key=lambda x: x[0])
                return flat
            return None

        return local_results

    def broadcast(self, data: Any, root: int = 0) -> Any:
        """
        Broadcast data from root to all ranks.

        Args:
            data: Data to broadcast (only used on root rank)
            root: Rank to broadcast from

        Returns:
            Broadcasted data on all ranks
        """
        if self._comm is not None:
            return self._comm.bcast(data, root=root)
        return data

    def gather(self, data: Any, root: int = 0) -> Optional[List[Any]]:
        """
        Gather data from all ranks to root.

        Args:
            data: Local data to gather
            root: Rank to gather to

        Returns:
            List of data from all ranks (on root), None on other ranks
        """
        if self._comm is not None:
            return self._comm.gather(data, root=root)
        return [data]

    def scatter(self, data: Optional[List[Any]], root: int = 0) -> Any:
        """
        Scatter data from root to all ranks.

        Args:
            data: List of data to scatter (only used on root, must have
                  length equal to number of ranks)
            root: Rank to scatter from

        Returns:
            Data for this rank
        """
        if self._comm is not None:
            return self._comm.scatter(data, root=root)
        return data[0] if data else None

    def barrier(self) -> None:
        """Synchronize all ranks."""
        if self._comm is not None:
            self._comm.Barrier()

    def reduce_sum(self, data: np.ndarray, root: int = 0) -> Optional[np.ndarray]:
        """
        Sum arrays across all ranks to root.

        Args:
            data: Local array to sum
            root: Rank to reduce to

        Returns:
            Summed array on root, None on other ranks
        """
        if self._comm is not None:
            from mpi4py import MPI
            if self.is_root:
                result = np.zeros_like(data)
                self._comm.Reduce(data, result, op=MPI.SUM, root=root)
                return result
            else:
                self._comm.Reduce(data, None, op=MPI.SUM, root=root)
                return None
        return data

    def allreduce_sum(self, data: np.ndarray) -> np.ndarray:
        """
        Sum arrays across all ranks, result available on all ranks.

        Args:
            data: Local array to sum

        Returns:
            Summed array on all ranks
        """
        if self._comm is not None:
            from mpi4py import MPI
            result = np.zeros_like(data)
            self._comm.Allreduce(data, result, op=MPI.SUM)
            return result
        return data

    def __repr__(self) -> str:
        mode = "parallel" if self.is_parallel else "serial"
        return f"<MPIVideoProcessor rank={self._rank}/{self._size} mode={mode}>"


# Type hint for PhotonVideo to avoid circular import
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .video import PhotonVideo
