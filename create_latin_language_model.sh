#!/bin/bash

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the script
python latin_language_model.py --corpus "../LatinTextDataset/latincorpus.txt" --epochs 8

# Deactivate virtual environment when done
deactivate
rm -r venv
