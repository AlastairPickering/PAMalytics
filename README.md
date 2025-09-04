# Pileated Gibbon Classifier and Dashboard

A complete workflow for deploying a trained pileated gibbon classifier end-to-end, reviewing results, validating clips with spectrograms, and launching the processing pipeline — all from a simple Streamlit UI.

### Features
- Deploy a pretrained pileated gibbon classifier to detect gibbon calls in PAM audio
- Change threshold and deploy post-processing heuristics to balance recall and precision requirements
- Interactive and intuitive dashboard to summarise and export results
- Streamlined validation and updating of results
- All processes run via app without need for terminal

### Dashboard (analysis)
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
- Sort & filter by clip probability (max segment probability per file)
- High-resolution spectrograms optimised for quick visual check
- Shows pending changes before saving
- Saves only UserLabel changes so you always preserve the original predictions

### Classify (Launch classifier)
- Start/stop scripts/pipeline.py with your chosen audio folder
- Pass extra CLI args (--tau, --kn, etc.)
- Live status (progress bar) + auto-refreshing logs
- Writes status JSON and log file to results/
- Classfier splits incoming .wav files into 10 second segments and calculates probability of containing a gibbon call

# Quick Start
Prerequisites:
Python 3.9+
macOS or Windows

### macOS
Double click scripts/launch.command (first run may need permission override in System Settings/Privacy & Security/Security/Allow applications downloaded from App store and identified developers).

It will:
- Create .venv
- Install requirements.txt
- Launch Streamlit on port 8503
- The app opens in your browser at http://localhost:8503.

### Windows
- Double-click scripts\launch_dashboard.bat (or run python scripts\launch_dashboard.py).
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
├─ scripts/
│  ├─ Dashboard.py                # main app page
│  ├─ pages/
│  │  ├─ 1_Validate.py            # deep validation (optional)
│  │  ├─ 2_Classify.py            # pipeline launcher + live logs
│  │  └─ 3_Settings.py            # settings & conversions
│  ├─ pipeline.py                 # batch processing loop
│  ├─ preprocessing.py            # audio preproc + embedding helpers
│  ├─ launch_dashboard.py         # Python launcher (creates venv, installs deps)
│  ├─ launch.command              # macOS double-click launcher
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

