# Thesis Image Generation

This document provides instructions on how to run the various Python scripts to generate the figures used in the thesis.

## Prerequisites

Before running the scripts, ensure you have all the required Python packages installed. You can install them using pip:

```bash
pip install -r requirements.txt
```

Most scripts are configured to save plots as PDF files with transparent backgrounds, suitable for inclusion in a LaTeX document.

## Running the Scripts

### 1. Particle Filter Results

This script generates plots and result tables for the particle filter localization experiments.

**Script:** `pf/pf results.py`

**Description:**
This script processes the pose data from different localization methods (naive, differential drive) and compares them against the ground truth. It generates:
- Histograms of pose and heading errors for each map (`esp`, `gbr`, `aut`, `mco`).
- A LaTeX table summarizing the RMSE, standard deviation, and max errors.

**To Run:**
Navigate to the project root directory and execute the following command:
```bash
python pf/pf_results.py
```

**Outputs:**
- `pose_error_histogram.pdf` and `heading_error_histogram.pdf` will be saved in each map's subdirectory (e.g., `pf/esp/`).
- A LaTeX table will be printed to the console.

### 2. Global Trajectory Planning

This script generates plots comparing different global trajectory planning methods.

**Script:** `global_planning/track.py`

**Description:**
This script reads pre-computed trajectory files (`_centreline.csv`, `_minCurve.csv`, `_short.csv`) for various maps and plots them over the map image. It compares the Centre Line, Shortest Path, and Minimum Curvature trajectories.

**To Run:**
Execute the script from the project root directory:
```bash
python global_planning/track.py
```

**Outputs:**
- A plot for each map (e.g., `esp_traj.pdf`, `gbr_traj.pdf`) will be saved in the `global_planning/` directory.
- Markdown tables with trajectory comparison statistics will be printed to the console.

### 3. Overtake Maneuver Results

This directory contains scripts to visualize different aspects of the simulated and physical overtaking maneuvers.

### Simulation Results

#### a) Opponent Detection

**Script:** `Overtake/sim/detect/detect.py`

**Description:**
This script visualizes the performance of the opponent vehicle detection algorithm. It plots the true opponent trajectory, correctly detected positions (True Positives), and incorrect detections (False Positives) on the map. It also generates a scatter plot of the detection errors.

**To Run:**
```bash
python Overtake/sim/detect/detect.py
```

**Outputs:**
The following files will be saved in `Overtake/sim/detect/Figs/`:
- `detection_results_on_map.pdf`
- `detection_error_scatter.pdf`

#### b) Opponent Tracking

**Script:** `Overtake/sim/tracking/track.py`

**Description:**
This script visualizes the performance of the opponent vehicle tracking system (e.g., using a Kalman filter). It generates plots comparing the estimated and actual opponent trajectories, positions, and errors over time. It also includes plots for a "trailing" scenario.

**To Run:**
```bash
python Overtake/sim/tracking/track.py
```

**Outputs:**
Multiple PDF plots will be saved in `Overtake/sim/tracking/Figs/`, including:
- `estimated_results_on_map.pdf`
- `position_error.pdf`
- `yaw_error.pdf`
- `speed_error.pdf`
- `distance_over_time.pdf` (for trailing)

#### c) Overtake Maneuver

**Script:** `Overtake/sim/overtake/overtake.py`

**Description:**
This script visualizes the complete overtaking maneuver at different opponent speeds (0, 25, 50, 70 km/h). It generates a sequence of plots showing the different phases of the overtake (approaching, planning, overtaking, returning to lane) and associated speed profiles.

**To Run:**
```bash
python Overtake/sim/overtake/overtake.py
```

**Outputs:**
For each speed folder (e.g., `Overtake/sim/overtake/25/`), the following plots will be generated:
- `global.pdf` (approaching phase)
- `planning.pdf` (planning phase)
- `overtake.pdf` (execution phase)
- `post_overtake.pdf` (returning phase)
- `speed_tracking.pdf`
- `progress_difference.pdf`

### Physical Results

#### a) Opponent Detection

**Script:** `Overtake/real/detect/detect.py`

**Description:**
This script visualizes the performance of the opponent vehicle detection algorithm for the physical robot. It plots the true opponent trajectory, correctly detected positions (True Positives), and incorrect detections (False Positives) on the map. It also generates a scatter plot of the detection errors.

**To Run:**
```bash
python Overtake/real/detect/detect.py
```

**Outputs:**
The following files will be saved in `Overtake/real/detect/Figs/`:
- `detection_results_on_map.pdf`
- `detection_error_scatter.pdf`

#### b) Opponent Tracking

**Script:** `Overtake/real/tracking/track.py`

**Description:**
This script visualizes the performance of the opponent vehicle tracking system for the physical robot. It generates plots comparing the estimated and actual opponent trajectories, positions, and errors over time. It also includes plots for a "trailing" scenario.

**To Run:**
```bash
python Overtake/real/tracking/track.py
```

**Outputs:**
Multiple PDF plots will be saved in `Overtake/real/tracking/Figs/`, including:
- `estimated_results_on_map.pdf`
- `position_error.pdf`
- `yaw_error.pdf`
- `speed_error.pdf`
- `distance_over_time.pdf` (for trailing)

#### c) Overtake Maneuver

**Script:** `Overtake/real/overtake/overtake.py`

**Description:**
This script visualizes the complete overtaking maneuver for the physical robot at different opponent speeds. It generates a sequence of plots showing the different phases of the overtake (approaching, planning, overtaking, returning to lane) and associated speed profiles.

**To Run:**
```bash
python Overtake/real/overtake/overtake.py
```

**Outputs:**
For each speed folder (e.g., `Overtake/real/overtake/25/`), the following plots will be generated:
- `global.pdf` (approaching phase)
- `planning.pdf` (planning phase)
- `overtake.pdf` (execution phase)
- `post_overtake.pdf` (returning phase)
- `speed_tracking.pdf`
- `progress_difference.pdf`


