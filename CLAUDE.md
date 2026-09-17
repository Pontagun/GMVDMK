# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

GMVDMK is a sensor-fusion research script that fuses gyroscope, magnetometer, accelerometer, and camera-tracked ground-truth readings into a single orientation quaternion over time. It reads a CSV recording of raw sensor data, applies a gated/motionless-aware complementary filter (gyro integration corrected by accel and magnetometer via slerp), and plots the resulting quaternion components. There is no packaging, CLI, or server — `main.py` is a single runnable script.

## Environment

- Python virtualenv is already set up at `.venv`. Use `.venv/Scripts/python.exe` (Windows) to run/install.
- No `requirements.txt` exists; core third-party deps observed in the venv: `numpy`, `pandas`, `matplotlib`, `numpy-quaternion` (imported as `quaternion`).

## Common commands

```
# Run the main pipeline (reads config.ini, plots result)
.venv/Scripts/python.exe main.py

# Run tests (currently no active test cases — see test_cases.py)
.venv/Scripts/python.exe -m unittest test_cases.py
```

There is no lint/format tooling configured in this repo.

## Configuration

All tunable parameters live in `config.ini`, read via `configparser` independently in several modules (`main.py`, `pipeline.py`, `sensor.py`, `helper.py` each call `config.read('config.ini')` themselves rather than sharing a loaded config). Key sections:

- `[ENVIRONMENT] Filename` — path to the input CSV under `REC_01-05/` (the recorded sensor session used for a run).
- `[SENSOR_THRESHOLD]` — per-axis gyro thresholds and `MTNLNSThreshold`, the accel-delta threshold used to detect "motionless" state.
- `[GMVDMK_WINDOW]` — `SmoothWindow` (moving-average window for smoothing raw sensor signals in `QSensor`), `AlphaWindow` (lookback window for the stillness/`alpha` gate in `main.py`).
- `[SLOPE]` — `MuKa`/`MuK` slopes used in `pipeline.Correction.get_mu_ka` / `get_mu_k` to shape correction weighting curves.

Switching to a different recording or window size is a config-only change; no code edit needed.

## Architecture

The pipeline is a per-timestep loop in `main.py` over rows of the input CSV, mixing three loosely coupled pieces:

1. **`sensor.QSensor`** (`sensor.py`) wraps a raw 3-axis signal (gyro/mag/accel) into an array of quaternions, one per row. Construction smooths each axis with a moving average (`SmoothWindow`) before building quaternions with `w=0` (i.e. pure vector quaternions representing 3D readings, not rotations). `QSensor.get_quat_normalized` is used throughout as a static helper to normalize *any* quaternion (gyro-integrated orientation, correction deltas, etc.), not just sensor readings.

2. **`camera.Camera`** (`camera.py`) exposes ground-truth position/orientation columns from the CSV, currently used only for `get_delta_t(i)` — the inter-frame time delta driving gyro integration in `main.py`.

3. **`pipeline.Correction`** (`pipeline.py`) is instantiated fresh *per sensor per timestep* (`Correction(accel.quat[i], a_init)`, `Correction(magnet.quat[i], m_init)`) and holds the logic for reconciling a gyro-integrated orientation estimate (`qG`) against a reference reading:
   - `get_sim_reading_frame_body` — rotate the sensor's initial reading into the body frame implied by `qG`, to compare against the current raw reading.
   - `get_delta_qref` — build a correction quaternion from the discrepancy between the simulated and actual reading (via `get_qref_w`/`get_qref_v`, a vector-to-vector rotation quaternion).
   - `get_qg_adjusted` — apply the correction delta to `qG`.
   - `get_mu_ka`/`get_mu_km`/`get_mu_fusion`/`get_mu_k` — compute magnetometer trust weights (`mu_k`) based on angular/magnitude discrepancy, gated by the `[SLOPE]` config values and by the current motion state (`alpha`).

4. **`helper.py`** holds free functions shared across the above: `get_gamma_filter` (exponential smoothing of the stillness signal into `alpha_mtnlns`), `get_sensor_diff` (thresholded accel-delta stillness metric), `get_vector` (quaternion → vector-part extraction).

**Main loop shape** (`main.py`): for each row past `alpha_window`,
- integrate gyro into `qG` via quaternion exponential (`qDot`, `power`, `np.exp`), then normalize;
- compute `alpha_mtnlns`, a smoothed 0–1 "stillness" signal from recent accelerometer deltas;
- run the accel and magnetometer `Correction` pipelines independently to get candidate-corrected orientations `qGA`, `qGM`;
- **single slerp**: blend `qG` toward each candidate using `alpha_mtnlns` (accel) and `mu_k` (magnetometer) as interpolation weights;
- **double slerp**: blend the two single-slerp results together, again weighted by `alpha_mtnlns`;
- normalize, append the result to `qG_lst`, then update `mu_k` (magnetometer trust) for the next iteration using `Correction.get_mu_ka`/`get_mu_km`/`get_mu_fusion`/`get_mu_k`.

The intent: gyro integration alone drifts; accel correction is trusted more when the device is still (`alpha_mtnlns` high); magnetometer correction (`mu_k`) is trusted based on how well its reading's angle/magnitude match the fused-frame expectation, further gated by motion state. The two-stage ("double") slerp is what gives the project its name.

Output is a single matplotlib plot of the four fused-quaternion components (`x`, `y`, `z`, `w`) over the recording — there is no other reporting/export currently wired up.

## Data files

`REC_01-05/*.csv` are recorded sensor sessions (raw phone/IMU + camera-tracked ground truth). Column names have spaces stripped on load (`df.columns = df.columns.str.replace(' ', '')`). Expected columns include `Timestamp`, `gyro_x/y/z`, `mag_x/y/z`, `acc_x/y/z`, `pos_x/y/z`, `cam_qx/qy/qz/qw`, `isTracked`. `rec01_mlab.csv` and `rec010GMV1.txt` appear to be MATLAB-derived comparison/reference outputs for the same recording (see recent commit "Rename the file which updated by data in Matlab").
