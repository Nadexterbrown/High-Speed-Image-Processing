"""
Metadata configuration for Photron video files.

Provides configurable metadata field filtering to control which
metadata is exposed from CIHX/MRAW files.
"""

from typing import Set, Optional, FrozenSet


class MetadataConfig:
    """
    Configuration for which metadata fields to expose from Photron videos.

    pyMRAW returns metadata with these fields:
        - 'Date': Recording date
        - 'Camera Type': Camera model
        - 'Record Rate(fps)': Frame rate
        - 'Shutter Speed(s)': Shutter speed in seconds
        - 'Total Frame': Total number of frames
        - 'Original Total Frame': Original frame count before trimming
        - 'Image Width': Frame width in pixels
        - 'Image Height': Frame height in pixels
        - 'File Format': File format (mraw, tiff, etc.)
        - 'EffectiveBit Depth': Bit depth of pixel data
        - 'EffectiveBit Side': Bit alignment (Lower/Higher)
        - 'Color Bit': Color bit depth
        - 'Comment Text': User comments

    Example:
        >>> config = MetadataConfig.minimal()  # Essential fields only
        >>> config = MetadataConfig.full()     # All available fields
        >>> config = MetadataConfig.for_processing()  # Essential + Recording
        >>> custom = MetadataConfig(fields={'Record Rate(fps)', 'Total Frame'})
    """

    # Field category definitions
    ESSENTIAL: FrozenSet[str] = frozenset({
        'Total Frame',
        'Image Width',
        'Image Height',
        'EffectiveBit Depth',
        'File Format',
    })

    RECORDING: FrozenSet[str] = frozenset({
        'Record Rate(fps)',
        'Shutter Speed(s)',
    })

    DEVICE: FrozenSet[str] = frozenset({
        'Camera Type',
        'Date',
    })

    EXTENDED: FrozenSet[str] = frozenset({
        'Original Total Frame',
        'EffectiveBit Side',
        'Color Bit',
        'Comment Text',
    })

    ALL_FIELDS: FrozenSet[str] = ESSENTIAL | RECORDING | DEVICE | EXTENDED

    def __init__(
        self,
        fields: Optional[Set[str]] = None,
        include_essential: bool = True
    ):
        """
        Initialize metadata configuration.

        Args:
            fields: Specific field names to include. If None, uses essential fields.
            include_essential: Always include essential fields (default True)
        """
        self._fields: Set[str] = set()

        if include_essential:
            self._fields.update(self.ESSENTIAL)

        if fields is not None:
            self._fields.update(fields)

    @classmethod
    def minimal(cls) -> 'MetadataConfig':
        """Create config with only essential fields."""
        return cls(include_essential=True)

    @classmethod
    def full(cls) -> 'MetadataConfig':
        """Create config with all available fields."""
        return cls(fields=cls.ALL_FIELDS, include_essential=True)

    @classmethod
    def for_processing(cls) -> 'MetadataConfig':
        """Create config optimized for image processing workflows."""
        return cls(
            fields=cls.ESSENTIAL | cls.RECORDING,
            include_essential=True
        )

    @property
    def fields(self) -> Set[str]:
        """Return set of field names to include."""
        return self._fields.copy()

    def should_include(self, field_name: str) -> bool:
        """Check if a field should be included."""
        return field_name in self._fields

    def filter_metadata(self, raw_metadata: dict) -> dict:
        """
        Filter raw metadata dictionary to include only configured fields.

        Args:
            raw_metadata: Full metadata dictionary from pyMRAW

        Returns:
            Filtered dictionary with only requested fields
        """
        return {
            key: value
            for key, value in raw_metadata.items()
            if self.should_include(key)
        }

    def __repr__(self) -> str:
        return f"MetadataConfig(fields={sorted(self._fields)})"
