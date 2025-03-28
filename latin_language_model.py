import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
import platform
import sys
from transformers import TextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
import coremltools as ct
import json
import numpy as np
import re
import os
import argparse
import unicodedata
from collections import Counter
import nltk
from nltk.corpus import words as nltk_words

# Download NLTK resources if not already present
try:
    nltk.data.find('corpora/words')
except LookupError:
    nltk.download('words')

# Corpus Cleanup Functions
def clean_latin_corpus(input_path, output_path):
    """
    Clean a Latin corpus file by:
    1. Normalizing Unicode characters
    2. Removing excessive whitespace and line breaks
    3. Removing obvious non-Latin content
    4. Standardizing punctuation
    """
    print(f"Cleaning corpus from {input_path}")
    
    # Load corpus
    with open(input_path, 'r', encoding='utf-8', errors='replace') as f:
        text = f.read()
    
    original_size = len(text)
    print(f"Original corpus size: {original_size} characters")
    
    # Step 1: Normalize Unicode
    text = unicodedata.normalize('NFKC', text)
    
    # Step 2: Remove headers, page numbers, and metadata
    text = re.sub(r'^\s*\[.*?\]\s*$', '', text, flags=re.MULTILINE)  # Remove [...] metadata
    text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)  # Remove standalone page numbers
    
    # Step 3: Normalize whitespace and line breaks
    text = re.sub(r'\r\n', '\n', text)  # Standardize line breaks
    text = re.sub(r'\n{3,}', '\n\n', text)  # Reduce excessive line breaks
    text = re.sub(r'\s+', ' ', text)  # Normalize spaces
    
    # Step 4: Standardize punctuation
    text = re.sub(r'["""]', '"', text)  # Standardize quotes
    text = re.sub(r"[''']", "'", text)  # Standardize apostrophes
    text = re.sub(r'\.{3,}', '...', text)  # Standardize ellipses
    
    # Step 5: Filter out obvious non-Latin content
    # Create a set of common English words that aren't also Latin
    english_words = set(nltk_words.words())
    latin_loanwords = {'et', 'in', 'ad', 'non', 'per', 'cum', 'ex', 'de', 'si', 'sum', 'pro', 
                      'ego', 'tu', 'nos', 'vos', 'ante', 'post', 'sub', 'super', 'est', 'sunt',
                      'via', 'visa', 'status', 'item', 'veto', 'alias', 'versus', 'campus',
                      'bonus', 'exit', 'extra', 'data', 'media', 'maximum', 'minimum', 'interim'}
    non_latin_identifiers = []
    
    # Identify paragraphs that are likely English
    paragraphs = text.split('\n\n')
    filtered_paragraphs = []
    
    for paragraph in paragraphs:
        words = re.findall(r'\b[a-zA-Z]+\b', paragraph.lower())
        if not words:
            filtered_paragraphs.append(paragraph)
            continue
            
        # Count English-only words (excluding Latin loanwords)
        english_only_words = [w for w in words if w in english_words and w not in latin_loanwords]
        english_ratio = len(english_only_words) / len(words) if words else 0
        
        # If paragraph has > 60% English-only words, it's likely not Latin
        if english_ratio > 0.6 and len(words) > 5:
            non_latin_identifiers.append(paragraph[:50] + "...")
        else:
            filtered_paragraphs.append(paragraph)
    
    # Rejoin the filtered paragraphs
    text = '\n\n'.join(filtered_paragraphs)
    
    # Ensure the corpus has enough content
    if len(text.strip()) < 500:
        print("WARNING: Cleaned corpus is very small (<500 chars). This may cause training issues.")
        
        # Add some sample Latin text if corpus is too small
        if len(text.strip()) < 100:
            print("Adding sample Latin text to ensure minimum corpus size")
            sample_latin = """
            Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt 
            ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation 
            ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in 
            reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur 
            sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.
            
            Curabitur pretium tincidunt lacus. Nulla gravida orci a odio. Nullam varius, turpis et commodo 
            pharetra, est eros bibendum elit, nec luctus magna felis sollicitudin mauris. Integer in mauris 
            eu nibh euismod gravida. Duis ac tellus et risus vulputate vehicula. Donec lobortis risus a elit.
            """
            text = text + "\n\n" + sample_latin
    
    # Save cleaned corpus
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(text)
    
    # Stats output
    final_size = len(text)
    print(f"Cleaned corpus size: {final_size} characters ({final_size/original_size*100:.2f}% of original)")
    print(f"Removed {len(non_latin_identifiers)} likely non-Latin paragraphs")
    print(f"Saved cleaned corpus to {output_path}")
    
    # Output some examples of removed content for verification
    if non_latin_identifiers:
        print("\nExamples of removed content (first 3):")
        for i, sample in enumerate(non_latin_identifiers[:3]):
            print(f"{i+1}. {sample}")
    
    return output_path

# Function to verify dataset creation will work
def verify_dataset_creation(file_path, tokenizer, block_size):
    """Check if the dataset creation will work with the given parameters"""
    try:
        # Check file size
        file_size = os.path.getsize(file_path)
        if file_size == 0:
            print(f"ERROR: File {file_path} is empty")
            return False
            
        # Read a small sample to test tokenization
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            sample_text = f.read(min(file_size, 1000))  # Read up to 1000 chars
            
        # Test tokenization
        tokens = tokenizer.encode(sample_text)
        if len(tokens) == 0:
            print(f"ERROR: Tokenizer produced 0 tokens from sample text")
            return False
            
        print(f"Sample tokenization produced {len(tokens)} tokens")
        
        # Calculate estimated number of samples
        estimated_tokens = (file_size / len(sample_text)) * len(tokens)
        estimated_samples = max(1, int(estimated_tokens / block_size))
        
        print(f"Estimated number of samples: {estimated_samples}")
        
        if estimated_samples < 10:
            print(f"WARNING: Very few samples estimated ({estimated_samples}). Training may not be effective.")
            if block_size > 64 and estimated_samples < 5:
                print(f"Suggestion: Reduce block_size from {block_size} to 64")
                
        return estimated_samples > 0
        
    except Exception as e:
        print(f"Error verifying dataset: {str(e)}")
        return False

# Set device configuration
def configure_device(force_cpu=False):
    """Configure the appropriate device for PyTorch operations"""
    if force_cpu:
        print("Forcing CPU usage as requested")
        return "cpu"
        
    if torch.cuda.is_available():
        print("CUDA device available, using GPU")
        return "cuda"
    elif hasattr(torch, 'backends') and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        # Check if on Mac with Apple Silicon
        print("MPS (Metal Performance Shaders) device available, using Apple GPU")
        
        # Set environment variables to control MPS behavior
        import os
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        
        # Test MPS functionality before committing to it
        try:
            print("Testing MPS device functionality...")
            test_tensor = torch.ones(2, 3, device="mps")
            test_result = test_tensor * 2
            test_result.cpu()  # Test moving data back to CPU
            print("MPS device test successful")
            return "mps"
        except Exception as e:
            print(f"MPS device test failed: {str(e)}")
            print("Falling back to CPU")
            return "cpu"
    else:
        print("No GPU available, using CPU")
        return "cpu"

# 1. Train or fine-tune the model
def train_latin_model(corpus_path, output_dir, block_size=128, force_cpu=False):
    # Initialize tokenizer and model (using a small model like GPT-2 small)
    print("Initializing tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    
    # Configure device
    device = configure_device(force_cpu)
    print(f"Using device: {device}")
    
    # Load model and move to appropriate device
    model = AutoModelForCausalLM.from_pretrained("distilgpt2")
    
    # Only move to device if not using CPU (to avoid unnecessary operations)
    if device != "cpu":
        try:
            model = model.to(device)
            print(f"Model moved to {device} successfully")
        except Exception as e:
            print(f"Warning: Failed to move model to {device}: {str(e)}")
            print("Falling back to CPU")
            device = "cpu"
    
    # Verify dataset creation will work
    print("Verifying dataset creation...")
    if not verify_dataset_creation(corpus_path, tokenizer, block_size):
        # Try with a smaller block size
        smaller_block_size = 64
        print(f"Trying with smaller block_size = {smaller_block_size}")
        if verify_dataset_creation(corpus_path, tokenizer, smaller_block_size):
            block_size = smaller_block_size
        else:
            raise ValueError(f"Cannot create dataset from corpus file {corpus_path}. Please check the file content.")
    
    # Create dataset
    print(f"Creating dataset with block_size={block_size}...")
    try:
        dataset = TextDataset(
            tokenizer=tokenizer,
            file_path=corpus_path,
            block_size=block_size,
        )
        
        # Verify dataset has samples
        if not hasattr(dataset, 'examples') or len(dataset.examples) == 0:
            raise ValueError("Dataset created but contains 0 samples")
            
        print(f"Dataset created successfully with {len(dataset.examples)} samples")
        
    except Exception as e:
        print(f"Error creating dataset: {str(e)}")
        raise
    
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, 
        mlm=False,
    )
    
    # Set up training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=4,
        per_device_train_batch_size=8,
        save_steps=10_000,
        save_total_limit=2,
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset,
    )
    
    # Train model
    trainer.train()
    
    # Save model and tokenizer
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Save vocabulary mapping for use in Swift
    vocab_dict = {word: idx for word, idx in tokenizer.get_vocab().items()}
    with open(f"{output_dir}/latin_vocab.json", "w") as f:
        json.dump(vocab_dict, f)
    
    return model, tokenizer

# 2. Optimize model for on-device use
def optimize_model(model_path):
    # Load the fine-tuned model
    model = AutoModelForCausalLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Ensure model is on CPU for quantization
    model = model.cpu()
    
    # Quantize the model to reduce size
    # This uses dynamic quantization
    quantized_model = torch.quantization.quantize_dynamic(
        model, 
        {nn.Linear}, 
        dtype=torch.qint8
    )
    
    # Save quantized model
    quantized_path = f"{model_path}_quantized"
    os.makedirs(quantized_path, exist_ok=True)
    quantized_model.save_pretrained(quantized_path)
    
    # Copy vocabulary file to quantized directory
    vocab_src = os.path.join(model_path, "latin_vocab.json")
    vocab_dst = os.path.join(quantized_path, "latin_vocab.json")
    if os.path.exists(vocab_src):
        import shutil
        shutil.copy(vocab_src, vocab_dst)
    
    return quantized_model, tokenizer

# 3. Export to CoreML format
def export_to_coreml(model, tokenizer, output_path, model_name="LatinTransformer"):
    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)
    
    # Move model to CPU for tracing and export
    model = model.cpu()
    print("Model moved to CPU for CoreML export")
    
    # Create a simple input example
    print("Creating example input for tracing...")
    
    # Use a slightly longer sequence to ensure proper tracing
    example_text = "Lorem ipsum dolor sit amet, consectetur adipiscing"
    input_ids = tokenizer.encode(example_text, return_tensors="pt").cpu()
    
    # Define the function to trace
    def model_prediction(x):
        # Ensure input is on CPU
        x = x.cpu()
        with torch.no_grad():
            # Get logits from the model (which is now on CPU)
            outputs = model(x)
            logits = outputs.logits
            # Take the logits for the last token
            next_token_logits = logits[:, -1, :]
            return next_token_logits
    
    print(f"Tracing model with input shape: {input_ids.shape}")
    
    # Trace the model
    try:
        traced_model = torch.jit.trace(model_prediction, input_ids)
        print("Model traced successfully")
    except Exception as e:
        print(f"Error tracing model: {str(e)}")
        print("Trying alternative approach with script instead of trace...")
        
        # Alternative approach using script instead of trace
        def alternative_model_prediction(x):
            # Ensure input is on CPU
            x = x.cpu()
            with torch.no_grad():
                outputs = model(x)
                return outputs.logits[:, -1, :]
                
        try:
            traced_model = torch.jit.script(alternative_model_prediction)
            # Test with example input
            test_output = traced_model(input_ids)
            print(f"Alternative approach successful. Output shape: {test_output.shape}")
        except Exception as e2:
            print(f"Alternative approach also failed: {str(e2)}")
            raise ValueError("Could not trace or script the model for CoreML conversion")
    
    print("Converting to CoreML format...")
    
    # Convert to CoreML with error handling
    try:
        mlmodel = ct.convert(
            traced_model,
            inputs=[ct.TensorType(name="input", shape=input_ids.shape, dtype=np.int32)],
            compute_units=ct.ComputeUnit.CPU_AND_NE  # Use Neural Engine if available
        )
        
        # Set model metadata
        mlmodel.author = "Dylan Walker Brown"
        mlmodel.license = "MIT"
        mlmodel.short_description = "Latin language model for next word prediction"
        mlmodel.version = "0.1.0"
        
        # Save the model
        model_path = os.path.join(output_path, f"{model_name}.mlmodel")
        mlmodel.save(model_path)
        
        print(f"Model successfully exported to {model_path}")
        print(f"Model size: {os.path.getsize(model_path) / (1024*1024):.2f} MB")
        
        return model_path
        
    except Exception as e:
        print(f"Error converting to CoreML: {str(e)}")
        print("\nDetailed error information:")
        import traceback
        traceback.print_exc()
        
        print("\nTrying alternative conversion approach...")
        try:
            # Try with fixed input shape
            fixed_shape = [1, 10]  # Try with a fixed sequence length of 10
            fixed_input = torch.zeros(fixed_shape, dtype=torch.int32, device="cpu")
            fixed_traced = torch.jit.trace(model_prediction, fixed_input)
            
            mlmodel = ct.convert(
                fixed_traced,
                inputs=[ct.TensorType(name="input", shape=fixed_shape, dtype=np.int32)],
                compute_units=ct.ComputeUnit.CPU_ONLY  # Fall back to CPU only
            )
            
            model_path = os.path.join(output_path, f"{model_name}_fallback.mlmodel")
            mlmodel.save(model_path)
            
            print(f"Fallback model successfully exported to {model_path}")
            print(f"Fallback model size: {os.path.getsize(model_path) / (1024*1024):.2f} MB")
            
            return model_path
            
        except Exception as e2:
            print(f"Alternative conversion also failed: {str(e2)}")
            raise ValueError("Could not convert model to CoreML format")

# 4. Analyze corpus and vocabulary for diagnostics
def analyze_corpus(corpus_path):
    """Perform basic analysis on the corpus to check its suitability for training"""
    with open(corpus_path, 'r', encoding='utf-8', errors='replace') as f:
        text = f.read()
    
    # Basic stats
    char_count = len(text)
    word_count = len(text.split())
    
    # Word frequency analysis
    words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
    word_freq = Counter(words)
    
    print(f"\nCorpus Analysis:")
    print(f"Total characters: {char_count}")
    print(f"Total words: {word_count}")
    print(f"Unique words: {len(word_freq)}")
    
    # Most common words
    print("\nTop 20 most common words:")
    for word, count in word_freq.most_common(20):
        print(f"  {word}: {count}")
    
    # Check for non-Latin characters
    latin_extended = set('āēīōūȳĀĒĪŌŪȲăĕĭŏŭĂĔĬŎŬ')
    unusual_chars = set()
    
    for char in text:
        if not char.isascii() and char not in latin_extended and not char.isspace() and not unicodedata.category(char).startswith('P'):
            unusual_chars.add(char)
    
    if unusual_chars:
        print("\nUnusual characters found:")
        print(''.join(sorted(unusual_chars)))
    else:
        print("\nNo unusual characters found.")

# Function to diagnose system setup
def diagnose_system():
    """Print diagnostic information about the system and PyTorch setup"""
    print("\n=== System Diagnostics ===")
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Operating system: {platform.system()} {platform.release()}")
    
    if platform.system() == "Darwin":
        print(f"macOS version: {platform.mac_ver()[0]}")
        # Check if running on Apple Silicon
        is_arm = platform.machine() == 'arm64'
        print(f"Apple Silicon (ARM): {'Yes' if is_arm else 'No (Intel)'}")
    
    # Check available devices
    print("\nDevice availability:")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"CUDA devices: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  Device {i}: {torch.cuda.get_device_name(i)}")
    
    # Check for MPS (Apple Metal) support
    has_mps = hasattr(torch, 'backends') and hasattr(torch.backends, 'mps')
    print(f"MPS support in PyTorch: {'Yes' if has_mps else 'No'}")
    if has_mps:
        print(f"MPS available: {torch.backends.mps.is_available()}")
        print(f"MPS built: {torch.backends.mps.is_built()}")
    
    # Basic tensor test
    print("\nTesting basic tensor operations:")
    try:
        # Test CPU
        print("CPU tensor test:", end=" ")
        x_cpu = torch.ones(2, 3)
        y_cpu = x_cpu * 2
        print("Passed")
        
        # Test CUDA if available
        if torch.cuda.is_available():
            print("CUDA tensor test:", end=" ")
            x_cuda = torch.ones(2, 3, device="cuda")
            y_cuda = x_cuda * 2
            print("Passed")
            
        # Test MPS if available
        if has_mps and torch.backends.mps.is_available():
            print("MPS tensor test:", end=" ")
            try:
                x_mps = torch.ones(2, 3, device="mps")
                y_mps = x_mps * 2
                print("Passed")
            except Exception as e:
                print(f"Failed: {type(e).__name__}: {str(e)}")
                
    except Exception as e:
        print(f"Failed: {type(e).__name__}: {str(e)}")
    
    print("===========================\n")

# 5. Main function to run the entire pipeline
def main():
    # Run system diagnostics first
    diagnose_system()
    
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Latin Language Model Training Pipeline')
    parser.add_argument('--corpus', type=str, required=True, help='Path to the raw Latin corpus file')
    parser.add_argument('--output', type=str, default='./output_model', help='Output directory for model files')
    parser.add_argument('--clean-only', action='store_true', help='Only clean the corpus without training')
    parser.add_argument('--skip-clean', action='store_true', help='Skip corpus cleaning step')
    parser.add_argument('--epochs', type=int, default=4, help='Number of training epochs')
    parser.add_argument('--model-name', type=str, default='LatinTransformer', help='Name for the final CoreML model')
    parser.add_argument('--block-size', type=int, default=128, help='Block size for dataset creation')
    parser.add_argument('--force-cpu', action='store_true', help='Force the use of CPU even if GPU is available')
    
    args = parser.parse_args()
    
    # Create output directories
    os.makedirs(args.output, exist_ok=True)
    model_output_dir = os.path.join(args.output, "model")
    coreml_output_dir = os.path.join(args.output, "coreml")
    
    # Step 1: Check if corpus file exists
    if not os.path.exists(args.corpus):
        raise FileNotFoundError(f"Corpus file not found: {args.corpus}")
        
    # Step 2: Clean corpus
    if not args.skip_clean:
        cleaned_corpus_path = os.path.join(args.output, "cleaned_corpus.txt")
        clean_latin_corpus(args.corpus, cleaned_corpus_path)
        corpus_path = cleaned_corpus_path
        
        # Analyze cleaned corpus
        analyze_corpus(corpus_path)
    else:
        corpus_path = args.corpus
        print(f"Skipping cleaning, using corpus at: {corpus_path}")
        
        # Still analyze the corpus
        analyze_corpus(corpus_path)
    
    # Exit if clean-only mode
    if args.clean_only:
        print("Clean-only mode selected. Exiting.")
        return
    
    # Step 3: Train model
    print("\n=== Training Model ===")
    try:
        model, tokenizer = train_latin_model(corpus_path, model_output_dir, args.block_size, args.force_cpu)
    except ValueError as e:
        print(f"Error during model training: {str(e)}")
        # If it fails with the original block_size, try a smaller one
        if "num_samples=0" in str(e) and args.block_size > 64:
            print("Trying again with a smaller block size (64)...")
            model, tokenizer = train_latin_model(corpus_path, model_output_dir, 64, args.force_cpu)
    
    # Step 4: Optimize model
    print("\n=== Optimizing Model ===")
    quantized_model, tokenizer = optimize_model(model_output_dir)
    
    # Step 5: Export to CoreML
    print("\n=== Exporting to CoreML ===")
    export_to_coreml(quantized_model, tokenizer, coreml_output_dir, args.model_name)
    
    print("\n=== Pipeline Complete ===")
    print(f"Raw corpus: {args.corpus}")
    print(f"Cleaned corpus: {corpus_path}")
    print(f"Model output: {model_output_dir}")
    print(f"Quantized model: {model_output_dir}_quantized")
    print(f"CoreML model: {coreml_output_dir}/{args.model_name}.mlmodel")

if __name__ == "__main__":
    main()