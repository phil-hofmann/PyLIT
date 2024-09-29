#!/bin/bash

# Total steps
total_steps=6

# Step 1: Create a virtual environment
echo "🎉 Step 1/$total_steps: Creating a virtual environment..."
python -m venv venv
echo "✅ Virtual environment created."

# Step 2: Activate the virtual environment
echo "🎉 Step 2/$total_steps: Activating the virtual environment..."
source venv/bin/activate
echo "✅ Virtual environment activated."

# Step 3: Install pip if not already installed
echo "🎉 Step 3/$total_steps: Checking for pip installation..."
if ! command -v pip &> /dev/null; then
    echo "pip not found. Installing pip..."
    python -m ensurepip
    echo "✅ pip installed."
else
    echo "pip is already installed."
    echo "✅ pip installation confirmed."
fi

# Step 4: Install Poetry using pip (if not already installed)
echo "🎉 Step 4/$total_steps: Checking for Poetry installation..."
if ! command -v poetry &> /dev/null; then
    echo "Poetry not found. Installing Poetry..."
    pip install poetry
    echo "✅ Poetry installed."
else
    echo "Poetry is already installed."
    echo "✅ Poetry installation confirmed."
fi

# Step 5: Install project dependencies using Poetry
echo "🎉 Step 5/$total_steps: Installing project dependencies with Poetry..."
poetry install
echo "✅ Project dependencies installed."

# TODO Generate the documentation

# Step 6: Deactivate the virtual environment
echo "🎉 Step 6/$total_steps: Deactivating the virtual environment..."
deactivate
echo "✅ Virtual environment deactivated."

echo "🎉 Setup complete! 🎉"