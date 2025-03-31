# Latin Language Model

A transformer based next-word prediction model for Latin language autocorrect on mobile device keyboards.

## Overview

This model training pipeline can be run on a large corpus of Latin text (ISO 639-1 Language Code `'la'`, _Latinum_). The pipeline has several stages

1. Check system capabilities (CUDA, MPS availability)
2. Clean Latin text dataset
  - Removes stray English paragraphs
  - Removes special and invalid characters
  - Retains macron accents (ā, ē, ī, ō, ū)
3. Breaks the corpus into chunks for processing
4. Trains the PyTorch model and exports to `Edge Dialect` IR
5. Converts this model to Apple's Core ML for deployment using `coremltools`

The generated Core ML model (`LatinTransformer.mlpackage`) is specifically optimized to fit within the 30MB limit of an iOS keyboard extension, where memory constraints are significant. The model is tuned to fit within these constraints while still providing effective next-word predictions for Latin autocorrect.

## Requirements

This pipeline was developed and tested with the following framework versions.

- Python 3.9
- PyTorch 2.4
- CoreMLTools 8.2
- Other dependencies listed in `requirements.txt`

## Project Structure

```
LatinLanguageModel/
├── latin_language_model.py
├── train_latin_language_model.sh
├── requirements.txt
└── output_model/
    ├── model/  # PyTorch
    │   ├── corpus_chunks/  # Processed training data chunks
    │   ├── latin_vocab.json  # Limited vocabulary mapping
    │   └── ...
    └── coreml/
        └── LatinTransformer.mlpackage  # Core ML export
```

## Usage

The simplest way to run the pipeline is with the included shell script:

```bash
./train_latin_language_model.sh
```

This creates a virtual environment, installs dependencies, and runs the training pipeline with default settings.

To run the Python script directly with custom options:

```bash
python latin_language_model.py --corpus "../LatinTextDataset/latincorpus.txt" --epochs 4 --vocab-size 24000
```

## Key Options

- `--corpus`: Path to the raw Latin corpus file
- `--output`: Output directory (default: `./output_model`)
- `--epochs`: Number of training epochs (default: 4)
- `--vocab-size`: Max vocabulary size (default: 24000)
- `--hidden-size`: Hidden size for model reduction (default: 256)
- `--num-heads`: Number of attention heads (default: 4)
- `--num-layers`: Number of transformer layers (default: 4)
- `--context-size`: Maximum context window size (default: 256)
- `--force-cpu`: Force CPU usage even if GPU is available
- `--clean-only`: Only clean the corpus without training
- `--skip-clean`: Skip corpus cleaning step
- `--skip-train`: Skip training and go directly to optimization/export
