import os
import re
import time
import json
import torch
import math
import logging
import numpy as np
from tqdm import tqdm
from dataclasses import dataclass
from typing import List, Dict
from torch.nn.utils import rnn
from datasets import load_dataset
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    DataCollatorForLanguageModeling, 
    Trainer, TrainingArguments,
    PreTrainedTokenizerBase
)
from sklearn.metrics.pairwise import cosine_similarity, rbf_kernel, linear_kernel



class FacilityLocation:
    def __init__(self, embedding_matrix, kernel_type='cosine', gamma=None, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize the facility location problem with different kernel options.
        
        Args:
            embedding_matrix: A 2D numpy array where each row is an embedding vector
            kernel_type: Type of kernel to use ('cosine', 'rbf', 'linear')
            gamma: Parameter for RBF kernel (if None, defaults to 1/n_features)
            device: The device to use ('cuda' or 'cpu')
        """
        self.device = device
        self.embedding_matrix = embedding_matrix
        self.n = embedding_matrix.shape[0]
        
        # Compute similarity matrix based on kernel type
        if kernel_type == 'cosine':
            similarity = cosine_similarity(embedding_matrix)
        elif kernel_type == 'rbf':
            similarity = rbf_kernel(embedding_matrix, gamma=gamma)
        elif kernel_type == 'linear':
            similarity = linear_kernel(embedding_matrix)
        else:
            raise ValueError(f"Unsupported kernel type: {kernel_type}")
            
        # Convert to PyTorch tensor and move to specified device
        self.similarity_matrix = torch.tensor(similarity, device=self.device, dtype=torch.float32)
    
    def evaluate(self, subset):
        """
        Evaluate the facility location function on a subset.
        
        Args:
            subset: List of indices in the subset
        Returns:
            The facility location value of the subset as a numpy float
        """
        if not subset:
            return 0.0
        subset_tensor = torch.tensor(subset, device=self.device)
        return torch.max(self.similarity_matrix[:, subset_tensor], dim=1)[0].sum().cpu().numpy()

    def marginal_gain(self, subset, candidate):
        """
        Compute the marginal gain of adding a candidate to the subset.
        
        Args:
            subset: Current subset of indices
            candidate: Candidate index to add
        Returns:
            Marginal gain as a torch.Tensor on the same device
        """
        if not subset:
            return self.similarity_matrix[:, candidate].sum()
            
        subset_tensor = torch.tensor(subset, device=self.device)
        current_max = torch.max(self.similarity_matrix[:, subset_tensor], dim=1)[0]
        new_max = torch.max(current_max, self.similarity_matrix[:, candidate])
        return (new_max - current_max).sum()

    def lazier_than_lazy_greedy(self, budget):
        """
        Optimize the facility location function using the lazier than lazy greedy algorithm.
        
        Args:
            budget: Maximum number of elements to select
        Returns:
            Tuple of (selected subset as numpy array, gain values as list of tuples)
        """
        subset = []
        gains = torch.zeros(self.n, device=self.device)
        gain_values = []
        
        # Initialize upper bounds of marginal gains
        for i in range(self.n):
            gains[i] = self.marginal_gain(subset, i)

        for _ in tqdm(range(budget), desc="Selecting Points"):
            while True:
                # Get the candidate with the maximum upper bound gain
                candidate = int(torch.argmax(gains).cpu().numpy())
                
                # Recompute its exact marginal gain
                exact_gain = self.marginal_gain(subset, candidate)
                
                # If the recomputed gain is the maximum, accept it
                if exact_gain >= torch.max(gains) - 1e-6:
                    subset.append(candidate)
                    gain_values.append((candidate, float(exact_gain.cpu().numpy())))
                    gains[candidate] = float('-inf')  # Mark as used
                    break
                else:
                    # Update the upper bound with the exact gain
                    gains[candidate] = exact_gain

        return gain_values

def clean_text(text):
    """Remove quotes and backslashes from text."""
    cleaned = text.strip('\"\\')
    return cleaned

def load_prompts(filepath="./prompts.json"):
    """
    Load prompts from a JSON file and return a list of strings.
    
    Args:
        filepath: Path to the JSON file containing prompts
        
    Returns:
        List of strings containing the prompts
    """
    cleaned_filepath = "cleaned_prompts.json"
    
    if os.path.exists(cleaned_filepath):
        print(f"Loading existing cleaned prompts from {cleaned_filepath}...")
        with open(cleaned_filepath, 'r', encoding='utf-8') as f:
            cleaned_prompts = json.load(f)
            return [item['prompt'] for item in cleaned_prompts]

    try:
        with open(filepath, 'r', encoding='utf-8') as file:
            data = json.load(file)
            
        # Handle different input formats
        if isinstance(data, list):
            # If data is already a list of dictionaries with 'prompt' key
            if all(isinstance(item, dict) and 'prompt' in item for item in data):
                cleaned_prompts = data
            else:
                # Convert simple strings to dictionaries with 'prompt' key
                cleaned_prompts = [{'prompt': clean_text(str(item))} for item in data]
        else:
            raise ValueError("Expected JSON file to contain a list of items")
            
        print(f"Loaded and cleaned {len(cleaned_prompts)} prompts")
        
        # Save cleaned prompts
        with open(cleaned_filepath, 'w', encoding='utf-8') as f:
            json.dump(cleaned_prompts, f, indent=2)
        
        # Return a list of strings
        return [item['prompt'] for item in cleaned_prompts]

    except Exception as e:
        print(f"Error loading prompts: {str(e)}")
        return []

def compute_embeddings(embedding_model="thenlper/gte-large", save_dir="./new_embeddings", prompts=None, filename="embeddings.npy"):
    """
    Compute embeddings for a list of prompts.
    
    Args:
        embedding_model: Model name to use for embeddings
        save_dir: Directory to save embeddings
        prompts: List of dictionaries containing 'prompt' key
        filename: Name of the file to save/load embeddings
    
    Returns:
        numpy array of embeddings
    """
    filepath = os.path.join(save_dir, filename)
    
    if os.path.exists(filepath):
        print(f"Loading existing embeddings from {filepath}...")
        return np.load(filepath)
    
    if not prompts:
        raise ValueError("No prompts provided")
    
    os.makedirs(save_dir, exist_ok=True)
    print(f"Loading embedding model: {embedding_model}")
    model = SentenceTransformer(embedding_model)

    embeddings = []
    batch_size = 32
    
    # Convert prompts to tensor-friendly format
    texts_to_embed = [str(item['prompt']).strip() for item in prompts]
    
    # Process in batches
    for i in tqdm(range(0, len(texts_to_embed), batch_size), desc="Computing embeddings"):
        batch = texts_to_embed[i:i+batch_size]
        batch_embeddings = model.encode(
            batch,
            batch_size=batch_size,
            device="cuda:0" if torch.cuda.is_available() else "cpu",
            show_progress_bar=False
        )
        embeddings.extend(batch_embeddings)
    
    embeddings = np.array(embeddings)
    np.save(filepath, embeddings)
    return embeddings

def select_representative_prompts(prompts_file="./prompts.json", top_k=100, kernel_type='cosine', save_path="./dedup_prompts.json", embed_filename="embeddings.npy"):
    """
    Select top_k representative prompts using facility location.
    
    Args:
        prompts_file: Path to the JSON file containing prompts
        top_k: Number of representative prompts to select
        kernel_type: Kernel type for similarity computation ('cosine', 'rbf', 'linear')
        save_path: Path to save the selected prompts
        
    Returns:
        List of selected prompt indices
    """
    # # Load and clean the prompts
    # cleaned_prompts = load_prompts(prompts_file)
    # if not cleaned_prompts:
    #     print("No prompts to process.")
    #     return []

    with open(prompts_file, 'r') as f:
        cleaned_prompts = json.load(f)
    
    # Compute embeddings for all prompts
    print(f"Computing embeddings for {len(cleaned_prompts)} prompts...")
    embeddings = compute_embeddings(prompts=cleaned_prompts, filename=embed_filename)
    
    # Create facility location instance
    print("Creating facility location instance...")
    facility = FacilityLocation(embeddings, kernel_type=kernel_type)
    
    # Run the facility location algorithm
    print(f"Running facility location to select {top_k} representative prompts...")
    gain_values = facility.lazier_than_lazy_greedy(top_k)
    
    # Extract selected indices
    selected_indices = [idx for idx, _ in gain_values]
    
    # Save the original selected prompts to file
    selected_prompts = [cleaned_prompts[idx] for idx in selected_indices]
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(selected_prompts, f, indent=2)
    
    print(f"Selected {len(selected_indices)} representative prompts and saved to {save_path}")
    # return selected_indices

# Example usage function
def main():
    """Example usage of the prompt deduplication functionality."""
    import argparse
    parser = argparse.ArgumentParser(description="Select representative prompts using facility location")
    parser.add_argument("--prompts-file", type=str, default="./prompts.json", help="Path to prompts JSON file")
    parser.add_argument("--top-k", type=int, default=100, help="Number of prompts to select")
    parser.add_argument("--kernel", type=str, default="cosine", choices=["cosine", "rbf", "linear"], help="Similarity kernel type")
    parser.add_argument("--output", type=str, default="./dedup_prompts.json", help="Output file path")
    parser.add_argument("--embed-filename", type=str, default="./embedding.npy", help="Filename for embeddings")
    
    args = parser.parse_args()
    
    select_representative_prompts(
        prompts_file=args.prompts_file,
        top_k=args.top_k,
        kernel_type=args.kernel,
        save_path=args.output,
        embed_filename=args.embed_filename
    )

if __name__ == "__main__":
    main()