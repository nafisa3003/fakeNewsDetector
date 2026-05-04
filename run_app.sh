#!/bin/bash
# Check if .venv exists
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi

# Ensure dependencies are installed
echo "Checking dependencies..."
.venv/bin/python3 -m pip install -q -r requirements.txt

# Run the app
echo "Launching Verity..."
.venv/bin/python3 -m streamlit run app.py
