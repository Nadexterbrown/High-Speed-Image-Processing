# High-Speed Image Processing

A Python library for processing Photron high-speed camera videos (CIHX/MRAW files) with automatic flame front tracking and deflagration-to-detonation transition (DDT) detection.

## Features

- **Photron Video Support**: Load and process CIHX/MRAW video files with PIMS-style lazy loading
- **Flame Front Tracking**: Multiple detection algorithms (threshold, gradient, half-maximum)
- **Automatic Exit Detection**: Stops processing when flame exits the camera domain
- **Frame Difference Analysis**: Isolates moving flame front from static background
- **Multi-Camera Support**: Configured for Nova and Mini high-speed cameras
- **MPI Parallelization**: Distribute frame processing across multiple cores
- **Visualization**: Generates diagnostic frame images and stacked sequence plots
- **Per-File Calibration**: Support for different calibrations across experimental runs

## Installation

### Requirements

- Python 3.8+
- NumPy
- Matplotlib
- pyMRAW (for Photron video loading)
- mpi4py (optional, for parallel processing)

### Install Dependencies

```bash
pip install numpy matplotlib pyMRAW
pip install mpi4py  # Optional, for MPI support
```

## Usage

### Serial Processing

```bash
python scripts/process_videos.py
```

### Parallel Processing with MPI

```bash
mpiexec -n 4 python scripts/process_videos.py
```

## Configuration

Edit `scripts/process_videos.py` to configure video sources:

```python
# Nova camera configuration
nova_config = VideoSourceConfig(name="Nova")
nova_config.enabled = True
nova_config.detection_method = "half_maximum"  # Options: "threshold", "gradient", "half_maximum"
nova_config.video_path = "./Nova-Video-Files"
nova_config.output_dir = "./Processed-Photos/Nova-Output"

# Mini camera configuration
mini_config = VideoSourceConfig(name="Mini")
mini_config.enabled = True
mini_config.detection_method = "threshold"  # Better for Mini due to signal characteristics
mini_config.video_path = "./Mini-Video-Files"
mini_config.output_dir = "./Processed-Photos/Mini-Output"
```

### Per-File Calibration

Set different calibrations for different experimental runs:

```python
nova_config.file_calibrations = [
    FileCalibration(
        calibration=0.000833333,  # meters per pixel
        position_offset=1.0159,    # position offset in meters
        files=["run-1-"]
    ),
    FileCalibration(
        calibration=0.000833333,
        position_offset=1.347567,
        files=["run-3-:run-10-"]  # Range pattern: run-3 through run-10
    ),
]
```

## Output

### Results File

Tab-separated text file with flame position data:

```
#Frame  Time_s      Position_px  Position_m
39      0.003368750 6            1.352566998
40      0.003375000 14           1.359233662
...
```

### Frame Images

Diagnostic images for each processed frame showing:
- Original frame
- Background-subtracted frame
- Frame difference (motion)
- Centerline intensity profile with detection markers

### Stacked Sequence

Paper-style visualization showing flame progression across multiple frames.

## Project Structure

```
High-Speed-Image-Processing/
├── scripts/
│   └── process_videos.py    # Main processing script
├── src/
│   ├── __init__.py
│   └── photron/
│       ├── video.py         # PhotonVideo class
│       ├── collection.py    # VideoCollection class
│       ├── parallel.py      # MPI support
│       └── metadata.py      # Metadata handling
├── Nova-Video-Files/        # Input: Nova camera CIHX files
├── Mini-Video-Files/        # Input: Mini camera CIHX files
├── Processed-Photos/        # Output directory
│   ├── Nova-Output/
│   └── Mini-Output/
└── README.md
```

## Detection Methods

### Threshold
Finds the rightmost edge of contiguous high-intensity regions. Best for data with strong signal behind the flame front (Mini camera).

### Half-Maximum
Finds where intensity drops to 50% of peak value on the falling edge. Best for clean flame fronts with good contrast (Nova camera).

### Gradient
Finds the steepest intensity drop (maximum negative gradient). Useful for sharp flame edges.

## Flame Exit Detection

The code automatically detects when the flame exits the camera domain:
- Triggers when detected position reaches the last 10 pixels of the image
- Truncates results to remove frames after exit
- Cleans up frame images past the exit point
- Works correctly with MPI parallelization

## License

MIT License
