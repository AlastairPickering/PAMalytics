#!/bin/bash

bash pamalytics_macos_launcher.command >output.log 2>&1 &
PID=$!

echo "Waiting for Streamlit to initialize..."
FOUND=false

for i in {1..30}; do
    if grep -q "You can now view your Streamlit app in your browser." output.log; then
        echo "✅ Success message detected!"
        FOUND=true
        break
    fi
    sleep 2
done

pkill -P $PID || kill $PID || true

if [ "$FOUND" = false ]; then
    echo "❌ Failed to detect success message within 60s"
    echo "--- FULL LOG OUTPUT ---"
    cat output.log
    exit 1
fi
