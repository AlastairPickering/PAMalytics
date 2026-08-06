# Troubleshooting

## The app doesn’t open

For the packaged desktop application, close any existing PAMalytics window or process and launch the application again.

PAMalytics runs locally and should open in your default browser at `http://localhost:8510`. If the browser does not open automatically, enter that address manually.

If port 8510 is already in use, close any existing PAMalytics or Streamlit process before reopening the application.

## First run blocked on macOS

The packaged macOS application is signed and notarised by Apple. Install it by opening the DMG, dragging **PAMalytics** into **Applications**, and launching it from Applications.

For the alternative source launcher, macOS may require you to allow the launcher in:

System Settings → Privacy & Security → Security.

## Windows security warning

The Windows installer is currently unsigned, so Windows may display a security warning.

After confirming that the installer came from the official PAMalytics release, select **More info** and then **Run anyway**.

## Importing a large audio archive is slow

Each indexing run saves an SQLite audio index. When importing detections that use the same large audio store, particularly remote or research data storage such as RDS, reusing that saved index can avoid scanning the full archive again.

In **Advanced options**:

1. Select **Use an existing saved audio index**.
2. Choose the existing SQLite audio index created by an earlier run.
3. Continue with audio matching.

Build a new index only when the audio store has changed, for example when files have been added, removed, renamed or moved.

## Validation page is slow

Validation pages can take longer to load when many detection cards or spectrograms are displayed at once.

To improve performance:

- reduce the number of cards shown per page;
- reduce the number of spectrograms loaded at once;
- apply filters before opening a large review set;
- work through the validation sample in smaller batches;
- close other browser tabs or applications using substantial memory.

The first load may also take longer while PAMalytics prepares spectrogram images. Subsequent pages should load more quickly.

## Docs preview doesn’t work

Install Material for MkDocs and run the development server:

```bash
pip install mkdocs-material
mkdocs serve
```
