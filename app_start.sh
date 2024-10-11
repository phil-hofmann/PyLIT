#!/bin/bash

# 🎉 Total steps
total_steps=3

# 1 Activate the Virtual Environment
echo "Step 1/$total_steps: Activating the virtual environment..."
source venv/bin/activate && echo "✅ Virtual environment activated!"

# 2: Navigate to the frontend directory
echo "Step 2/$total_steps: Navigating to the frontend directory..."
cd src/pylit/frontend/ || { echo "Directory not found: src/pylit/frontend/"; exit 1; }
echo "✅ Navigated to frontend directory!"

# 3: Start the Streamlit app
echo "Step 3/$total_steps: Starting the Streamlit app... 🚀"
echo "✅ Streamlit app is running!" && poetry run streamlit run "👋_hello.py"

# Goodbye message
echo "Goodbye! 🚀"