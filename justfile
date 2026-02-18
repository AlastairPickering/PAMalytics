default:
    @just --list

dev:
    uv run streamlit run src/pamalytics/app.py --server.port 8510

install:
    uv sync
