#!/bin/bash

# Create and activate virtual environment
python3.9 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the PyTorch model training and Apple CoreML conversion script
python latin_language_model.py --corpus "../LatinTextDataset/latincorpus.txt"

# Deactivate virtual environment and remove it
deactivate
rm -r venv
