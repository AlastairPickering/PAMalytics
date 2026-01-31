# Troubleshooting

## The app doesn’t open

You should see Streamlit running locally and the browser opening at `http://localhost:8510`. :contentReference[oaicite:11]{index=11}

## First run blocked on macOS

Allow the launcher in:
System Settings → Privacy & Security → Security → “Allow … identified developers”. :contentReference[oaicite:12]{index=12}

## Docs preview doesn’t work

Install Material for MkDocs and run the dev server:

```bash
pip install mkdocs-material
mkdocs serve
