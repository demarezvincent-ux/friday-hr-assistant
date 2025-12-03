#!/bin/bash

# 1. Setup Virtual Environment
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# 2. Activate the virtual environment
source venv/bin/activate

# 3. Fix Pip Permission Error
export PIP_USER=false 

# 4. Install requirements
if [ -f "requirements.txt" ]; then
    echo "Installing requirements..."
    pip install -r requirements.txt
fi

# 5. Run Streamlit App
# We use 'exec' so the signal handlers work correctly
# We bind to 0.0.0.0 so external users (and you) can see the app
echo "Starting FRIDAY Streamlit App on port 5000..."
exec streamlit run main.py --server.port 5000 --server.address 0.0.0.0