# Dashboard

The Detection Dashboard gives you a high-level view of your project and a quick way to explore detections before you start validating.

![Dashboard overview](assets/screenshots/dashboard1.png){ width="1100" }

## What you can do on the Dashboard

PAMalytics provides:

- Headline stats (total detections, total recordings, detection rate)
- Dataset, date-range and grouping controls
- Location stats table (counts and rates)
- Interactive map sized by detections per recorder
- Detections over time and by time of day
- Spectrogram preview grid with audio playback

## Explore your detections

When you launch a PAMalytics project you’ll land on the Detection Dashboard. Use it to get an overview of your data and run basic analytics without leaving the app.

At the top of the page you can:

- choose the **dataset** to display, including the original detections or the published validated dataset where available
- set a **date range** using the date picker
- group results by **species** or **recorder**
- select **Clear filters** to restore the default view

The grouping variable controls how summaries and charts are broken down across the dashboard.

If you imported `lat/lon` and/or `date_time` metadata, the dashboard will automatically populate:

- an **interactive map** showing detections geographically
- a **detections over time** chart
- an **intra-day detections** chart (time-of-day pattern)

All of these views are separated by your selected grouping variable (e.g. species or site).

## Preview audio and spectrograms

You can inspect detections directly from the dashboard using the spectrogram preview grid.

Use **Spectrograms per page**, **Columns per row**, and **Page** to control how many previews are loaded and how they are arranged. Reducing the number of spectrograms per page can improve performance on large datasets or slower computers.

Each spectrogram is annotated to help you triage quickly:

- the predicted label (e.g. species)
- the number of detections found
- detection markers overlaid on the spectrogram, coloured/positioned by detector confidence (as provided by your classifier)

Use the audio playback alongside the spectrograms to confirm detections before moving into full validation.
