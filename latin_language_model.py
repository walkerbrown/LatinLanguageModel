import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
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
    with open(input_path, 'r', encoding='utf-8') as f:
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

# 1. Train or fine-tune the model
def train_latin_model(corpus_path, output_dir):
    # Initialize tokenizer and model (using a small model like GPT-2 small)
    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    model = AutoModelForCausalLM.from_pretrained("distilgpt2")
    
    # Create dataset
    dataset = TextDataset(
        tokenizer=tokenizer,
        file_path=corpus_path,
        block_size=128,
    )
    
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
    
    # Create a simple input example
    input_ids = tokenizer.encode("Lorem ipsum dolor sit", return_tensors="pt")
    
    # Define the function to trace
    def model_prediction(x):
        with torch.no_grad():
            # Get logits from the model
            outputs = model(x)
            logits = outputs.logits
            # Take the logits for the last token
            next_token_logits = logits[:, -1, :]
            return next_token_logits
    
    print(f"Tracing model with input shape: {input_ids.shape}")
    
    # Trace the model
    traced_model = torch.jit.trace(model_prediction, input_ids)
    
    print("Converting to CoreML format...")
    
    # Convert to CoreML
    mlmodel = ct.convert(
        traced_model,
        inputs=[ct.TensorType(name="input", shape=input_ids.shape, dtype=np.int32)],
        compute_units=ct.ComputeUnit.CPU_AND_NE  # Use Neural Engine if available
    )
    
    # Set model metadata
    mlmodel.author = "Dylan Walker Brown"
    mlmodel.license = "MIT"
    mlmodel.short_description = "Latin language model for mobile keyboard autocorrect"
    mlmodel.version = "0.1"
    
    # Save the model
    model_path = os.path.join(output_path, f"{model_name}.mlmodel")
    mlmodel.save(model_path)
    
    print(f"Model successfully exported to {model_path}")
    print(f"Model size: {os.path.getsize(model_path) / (1024*1024):.2f} MB")

    return model_path

# 4. Analyze corpus and vocabulary for diagnostics
def analyze_corpus(corpus_path):
    """Perform basic analysis on the corpus to check its suitability for training"""
    with open(corpus_path, 'r', encoding='utf-8') as f:
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

# 5. Main function to run the entire pipeline
def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Latin Language Model Training Pipeline')
    parser.add_argument('--corpus', type=str, required=True, help='Path to the raw Latin corpus file')
    parser.add_argument('--output', type=str, default='./output_model', help='Output directory for model files')
    parser.add_argument('--clean-only', action='store_true', help='Only clean the corpus without training')
    parser.add_argument('--skip-clean', action='store_true', help='Skip corpus cleaning step')
    parser.add_argument('--epochs', type=int, default=4, help='Number of training epochs')
    parser.add_argument('--model-name', type=str, default='LatinTransformer', help='Name for the final CoreML model')
    
    args = parser.parse_args()
    
    # Create output directories
    os.makedirs(args.output, exist_ok=True)
    model_output_dir = os.path.join(args.output, "model")
    coreml_output_dir = os.path.join(args.output, "coreml")
    
    # Step 1: Clean corpus
    if not args.skip_clean:
        cleaned_corpus_path = os.path.join(args.output, "cleaned_corpus.txt")
        clean_latin_corpus(args.corpus, cleaned_corpus_path)
        corpus_path = cleaned_corpus_path
        
        # Analyze cleaned corpus
        analyze_corpus(corpus_path)
    else:
        corpus_path = args.corpus
        print(f"Skipping cleaning, using corpus at: {corpus_path}")
    
    # Exit if clean-only mode
    if args.clean_only:
        print("Clean-only mode selected. Exiting.")
        return
    
    # Step 2: Train model
    print("\n=== Training Model ===")
    model, tokenizer = train_latin_model(corpus_path, model_output_dir)
    
    # Step 3: Optimize model
    print("\n=== Optimizing Model ===")
    quantized_model, tokenizer = optimize_model(model_output_dir)
    
    # Step 4: Export to CoreML
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
