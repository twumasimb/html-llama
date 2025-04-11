import os
import re
import time
import json
import math
import torch
import pickle
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
import argparse


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
    

def compute_embeddings(dataset, embedding_model="thenlper/gte-large", save_dir="./embeddings"):
    """Compute embeddings for the Alpaca dataset using all fields."""
    if os.path.exists(os.path.join(save_dir, "embeddings.npy")):
        print("Loading existing embeddings...")
        return np.load(os.path.join(save_dir, "embeddings.npy"))
    
    os.makedirs(save_dir, exist_ok=True)
    print(f"Loading embedding model: {embedding_model}")
    model = SentenceTransformer(embedding_model)

    embeddings = []
    pbar = tqdm(total=len(dataset), desc="Computing embeddings")
    
    for i in range(0, len(dataset), 32):
        batch = dataset[i:i+32]
        # Get the number of examples in this batch
        batch_size = len(batch['instruction'])
        
        # Combine all fields for embedding
        combined_texts = []
        for j in range(batch_size):
            instruction = str(batch['instruction'][j])
            context = str(batch['input'][j])
            # output = str(batch['output'][j])
            # combined_text = f"{instruction} {input_text} {output}".strip()
            combined_text = f"{instruction} {context}".strip()
            combined_texts.append(combined_text)
            
        batch_embeddings = model.encode(combined_texts, batch_size=32, device="cuda:0", show_progress_bar=False)
        embeddings.extend(batch_embeddings)
        pbar.update(min(32, len(dataset) - i))
    pbar.close()
    embeddings = np.array(embeddings)
    np.save(os.path.join(save_dir, "embeddings.npy"), embeddings)
    return embeddings

def taylor_softmax(x, dim=1, n=2):
    """Compute Taylor Softmax probabilities."""
    assert n % 2 == 0 and n > 0
    fn = torch.ones_like(x)
    denor = 1.
    for i in range(1, n + 1):
        denor *= i
        fn = fn + x.pow(i) / denor
    return fn / fn.sum(dim=dim, keepdims=True)

def sample_indices_with_taylor_softmax(gains, k, dim=0, n=2):
    """Sample indices using Taylor Softmax distribution."""
    indices, values = zip(*gains)
    values_tensor = torch.tensor(values)
    probabilities = taylor_softmax(values_tensor, dim=dim, n=n).numpy()
    sampled_indices = np.random.choice(indices, size=k, replace=False, p=probabilities)
    return sampled_indices.tolist()

def compute_embeddings_for_prompts(prompts, embedding_model="thenlper/gte-large", save_dir="./embeddings"):
    """Compute embeddings for a list of prompts."""
    embedding_path = os.path.join(save_dir, "prompt_embeddings.npy")
    
    if os.path.exists(embedding_path):
        print("Loading existing embeddings...")
        return np.load(embedding_path)
    
    os.makedirs(save_dir, exist_ok=True)
    print(f"Loading embedding model: {embedding_model}")
    model = SentenceTransformer(embedding_model)

    embeddings = []
    batch_size = 32
    pbar = tqdm(total=len(prompts), desc="Computing embeddings")
    
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i+batch_size]
        batch_embeddings = model.encode(batch, batch_size=batch_size, device="cuda:0" if torch.cuda.is_available() else "cpu", show_progress_bar=False)
        embeddings.extend(batch_embeddings)
        pbar.update(min(batch_size, len(prompts) - i))
    
    pbar.close()
    embeddings = np.array(embeddings)
    np.save(embedding_path, embeddings)
    return embeddings

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Select diverse prompts using facility location.")
    parser.add_argument("--num", type=int, default=5000, help="Number of prompts to select")
    parser.add_argument("--kernel", type=str, default="cosine", choices=["cosine", "rbf", "linear"], help="Kernel type for similarity")
    args = parser.parse_args()
    
    # Load the dataset
    with open('./final_dataset/final_dataset.json', 'r') as f:
        dataset = json.load(f)

    # Extract prompts
    prompts = [item['prompt'] for item in dataset]
    print(f"Found {len(prompts)} prompts")
    
    # Compute embeddings
    embeddings = compute_embeddings_for_prompts(prompts)
    
    # Initialize facility location
    facility = FacilityLocation(embeddings, kernel_type=args.kernel)
    
    # Run the algorithm to select the best prompts
    print(f"Selecting {args.num} prompts using facility location with {args.kernel} kernel...")
    gains = facility.lazier_than_lazy_greedy(args.num)
    
    # Extract the selected indices and prompts
    selected_indices = [int(idx) for idx, _ in gains]
    selected_prompts = [prompts[idx] for idx in selected_indices]
    
    # Convert gains to JSON-serializable format
    json_gains = [(int(idx), float(gain)) for idx, gain in gains]
    
    # Save selected prompts to file
    output = {
        "selected_prompts": selected_prompts,
        "selected_indices": selected_indices,
        "gains": json_gains
    }
    
    output_file = f"./final_dataset/selected_{args.num}_prompts.json"
    print(f"Saving {len(selected_prompts)} selected prompts to {output_file}")
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)
    
    print("Done!")

if __name__ == '__main__':
    main()
