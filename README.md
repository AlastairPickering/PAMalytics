# PAMalytics — PAM Classifier, Detection Dashboard & Occupancy
An end-to-end workflow for deploying a trained pileated gibbon classifier, reviewing results, validating clips with spectrograms, fitting occupancy models that account for imperfect detection, and launching the processing pipeline.

### Features
- Deploy a pretrained pileated gibbon classifier to detect gibbon calls in PAM audio
- Flexible thresholding and post-processing heuristics to balance recall/precision
- Occupancy module to convert classifications into detection histories and fit single-season occupancy models (with options for false-positive handling and score-aware/threshold-free workflows)
- Interactive dashboard to summarise, validate and export results
- Streamlined clip validation with high-resolution spectrograms
- All processes run via the app — no terminal required

### Dashboard (analysis)

<img width="2668" height="1218" alt="image" src="https://github.com/user-attachments/assets/cbf2837b-bd53-474e-9e4b-c3552bf4be8e" />

- Headline stats: total detections, total recordings, detection rate
- Global date range and recorder filters (AND logic) that control the whole page
- Location Stats table with detection counts & rates
- Interactive map (pydeck) sized by detections per recorder
- Detections over time and by time of day (Altair)
- Validation grid with compact spectrogram thumbnails + full audio playback
- One-click annotation updates:
    - Non-destructive overrides stored in UserLabel (not overwriting FinalLabel)
    - Effective label = UserLabel (if set) else FinalLabel

### Settings
- Choose the audio folder used to locate clips (defaults to repo_root/audio)
- Pick a results file (CSV/XLSX) to use as filename-level ground truth
- Convert segment-level → filename-level with:
    - Adjustable threshold — auto-detected from model bundle when available
    - Optional K-of-N smoothing (e.g., 2 detections in 3 segments)

### Validate (deep review)

<img width="2687" height="1442" alt="image" src="https://github.com/user-attachments/assets/8ac32ce4-ddc4-4c45-a98d-0b822db9bdd8" /> <br>

- Sort & filter by clip probability (max segment probability per file)
- High-resolution spectrograms optimised for quick visual check
- Shows pending changes before saving
- Saves only UserLabel changes so you always preserve the original predictions

### Classify (Launch classifier)

<img width="2685" height="1242" alt="image" src="https://github.com/user-attachments/assets/e115f8cb-35f3-40bb-9a81-7ef9937f3031" />

- Start/stop scripts/pipeline.py with your chosen audio folder
- Pass extra CLI args (--tau, --kn, etc.)
- Live status (progress bar) + auto-refreshing logs
- Writes status JSON and log file to results/
- Classfier splits incoming .wav files into 10 second segments and calculates probability of containing a gibbon call

### Occupancy (modelling)

<img width="2697" height="1281" alt="image" src="https://github.com/user-attachments/assets/3f6680f6-db0c-4721-9762-8180bce8883c" />

- Build detection histories from classifier outputs with two modes:
    - Thresholded: apply a filename-level probability threshold (with optional K-of-N smoothing).
    - Score-aware (threshold-free): use classifier scores directly as evidence.
- Account for imperfect detection by estimating detectability from replicated visits/windows; optionally allow false positives via a verified subset/double-method design.
- Covariates: include site- and visit-level predictors for occupancy (ψ) and detection (p).

Outputs: <br>
    - Site-level occupancy probabilities (ψ) with uncertainty <br>
    - Model fit summaries and diagnostics <br>
    - Detection/non-detection (or score) matrices for archiving/reuse <br>
    - Exportable CSVs (histories, coefficients, site-level predictions) <br>

# Quick Start
Prerequisites: <br>
Python 3.9 <br>
macOS or Windows

### macOS
Double click MacOS_launcher.command (first run may need permission override in System Settings/Privacy & Security/Security/Allow applications downloaded from App store and identified developers).

It will:
- Create .venv
- Install requirements.txt
- Launch Streamlit on port 8503
- The app opens in your browser at http://localhost:8503.

### Windows
- Double-click Windows_launcher.bat (or run python scripts\launch_dashboard.py).
- Same behaviour: venv + requirements + Streamlit on port 8503.

### manual launch
python -m venv .venv <br>
. .venv/bin/activate  # Mac        
.venv\Scripts\activate # Windows <br>
pip install -r requirements.txt <br>
streamlit run scripts/Dashboard.py --server.port 8503 <br>

<details open><summary>Repository layout</summary>
<pre>
repo/
├─ MacOS_launcher.command         # macOS double-click dashboard launcher
├─ Windows_launcher.bat           # Windows double-click dashboard launcher
├─ scripts/
│  ├─ Dashboard.py                # main app page
│  ├─ pages/
│  │  ├─ 1_Validate.py            # deep validation (optional)
│  │  ├─ 2_Classify.py            # pipeline launcher + live logs
│  │  ├─ 3_Settings.py            # settings & conversions
│  │  └─ 4_Occupancy.py           # Occupancy modelling - build and display occupancy dashboard
│  ├─ pipeline.py                 # batch processing loop
│  ├─ preprocessing.py            # audio preproc + embedding helpers
│  ├─ prepare_occupancy.py        # occupancy preproc + helpers
│  ├─ launch_dashboard.py         # Python launcher (creates venv, installs deps)
│  ├─ requirements.txt             
│  └─ config.py                   # paths & constants (RAW_AUDIO_DIR, RESULTS_DIR, etc.)
├─ results/                       # outputs (CSV, status, logs, assets)
│  ├─ filename_level.csv          # active filename-level ground truth
│  ├─ classification_results.csv  # segment-level running index
│  ├─ merged_classification_results.csv
│  ├─ pipeline_status.json        # status file written by pipeline
│  └─ pipeline.log                # pipeline output log
├─ audio/                         # default audio folder (user can change)
├─ models/                        # classifier bundle(s), BEATs weights, etc.
└─ README.md
</pre>
</details>

