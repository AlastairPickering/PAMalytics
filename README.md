# PAMalytics — PAM Classifier Validation dashboard
[![CI (macOS & Windows)](https://github.com/AlastairPickering/PAMalytics/actions/workflows/ci.yaml/badge.svg)](https://github.com/AlastairPickering/PAMalytics/actions/workflows/ci.yaml)

An open source, no-code, classifier-agnostic, interactive dashboard to efficiently review bioacoustic classifier results. 

### Features
- Interactive dashboard to summarise, validate and export results
- Total review effort tracked - how many detections validated per group (e.g., species, recorder); what proportion correct, changed
- Update detections to correct species and/or presence-absence designation
- Streamlined clip validation with high-resolution spectrograms and classifier probabilities displayed directly at detection points
- Audio playback alongside spectrograms
- Flexible reclassification of detection thresholds with modelling of impact on detection rates
- All processes run via the app — no terminal required
- Saved validated csv with tracked user-levels changes ready to plug into downstream analysis and reporting tasks

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

### Validate

<img width="2946" height="1630" alt="image" src="https://github.com/user-attachments/assets/8c2d4016-a9ad-472f-927e-ba2c85e64597" />

- Sort & filter by clip probability (min-max segment probability per file)
- High-resolution spectrograms optimised for quick visual check
- Shows pending changes before saving
- Saves only UserLabel changes so you always preserve the original predictions

<img width="2915" height="1286" alt="image" src="https://github.com/user-attachments/assets/d26837ee-43d8-45b4-b9eb-b87d80db16df" />

<img width="2876" height="1451" alt="image" src="https://github.com/user-attachments/assets/82fd2b1d-c6a5-48da-946a-b1bf4d0156ea" />

# Quick Start
Prerequisites: <br>
Python 3.9 - 3.12 <br>
macOS or Windows

### macOS
Double click pamalytics_macos_launcher.command (first run may need permission override in System Settings/Privacy & Security/Security/Allow applications downloaded from App store and identified developers).

It will:
- Create .venv
- Install requirements.txt
- Launch Streamlit on port 8510
- The app opens in your browser at http://localhost:8510.

### Windows
- Double-click pamalytics_windows_launcher.bat (or run python scripts\launch_dashboard.py).
- Same behaviour: venv + requirements + Streamlit on port 8510.

### manual launch
python -m venv .venv <br>
. .venv/bin/activate  # Mac        
.venv\Scripts\activate # Windows <br>
pip install -r requirements.txt <br>
streamlit run scripts/Dashboard.py --server.port 8503 <br>

</details>

