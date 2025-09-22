"""
Dataset loading and preprocessing for shard-based training
Supports JSONL format with reasoning traces
"""

import json
import torch
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer
import os
from typing import List, Dict, Any, Optional
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

def load_shard(shard_path: str, tokenizer: AutoTokenizer) -> Dataset:
    """
    Load a single data shard (.jsonl) and tokenize it
    
    Expected format:
    {
        "prompt": "Solve step by step: If x + 3 = 7, what is x?",
        "completion": "Let's solve this step by step. The equation is x + 3 = 7. To isolate x, subtract 3 from both sides: x + 3 - 3 = 7 - 3, which simplifies to x = 4. So the answer is 4.",
        "domain": "math",
        "difficulty": "easy"
    }
    """
    if not os.path.exists(shard_path):
        raise FileNotFoundError(f"Shard not found: {shard_path}")
    
    # Load JSONL
    data_files = {"train": shard_path}
    dataset = load_dataset("json", data_files=data_files, split="train")
    
    # Basic filtering
    initial_size = len(dataset)
    dataset = dataset.filter(lambda x: x['prompt'].strip() and x['completion'].strip())
    logger.info(f"Filtered {initial_size} -> {len(dataset)} examples")
    
    # Tokenize
    def tokenize_function(examples):
        # Combine prompt + completion for causal LM training
        texts = [f"{prompt}\n{completion}" for prompt, completion in 
                zip(examples['prompt'], examples['completion'])]
        
        tokenized = tokenizer(
            texts,
            truncation=True,
            padding='max_length',
            max_length=1024,  # Reasonable for reasoning traces
            return_tensors="pt"
        )
        
        # Labels = input_ids (causal LM)
        tokenized["labels"] = tokenized["input_ids"].clone()
        
        # Add metadata
        tokenized["domain"] = examples["domain"]
        tokenized["difficulty"] = examples["difficulty"]
        
        return tokenized
    
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names
    )
    
    logger.info(f"Loaded and tokenized shard: {shard_path} ({len(tokenized_dataset)} examples)")
    return tokenized_dataset

def preprocess_dataset(dataset: Dataset, max_length: int = 1024) -> Dataset:
    """Additional preprocessing: filter by length, balance domains"""
    def filter_and_truncate(examples):
        # Filter examples that are too long
        valid_mask = [len(ids) <= max_length for ids in examples['input_ids']]
        filtered = {k: [v[i] for i in range(len(v)) if valid_mask[i]] for k, v in examples.items()}
        return filtered
    
    processed = dataset.map(
        filter_and_truncate,
        batched=True,
        remove_columns=dataset.column_names
    )
    
    # Balance domains if multiple present
    domains = processed.unique('domain')
    if len(domains) > 1:
        processed = balance_domains(processed)
    
    logger.info(f"Preprocessed dataset: {len(processed)} examples")
    return processed

def balance_domains(dataset: Dataset, max_per_domain: Optional[int] = None) -> Dataset:
    """Balance dataset across domains"""
    if max_per_domain is None:
        # Auto-balance: equal samples per domain
        domain_counts = dataset.to_pandas()['domain'].value_counts()
        max_per_domain = min(domain_counts)
    
    balanced_data = []
    for domain in dataset.unique('domain'):
        domain_data = dataset.filter(lambda x: x['domain'] == domain)
        if len(domain_data) > max_per_domain:
            domain_data = domain_data.shuffle().select(range(max_per_domain))
        balanced_data.append(domain_data)
    
    balanced = concatenate_datasets(balanced_data)
    balanced = balanced.shuffle()
    
    logger.info(f"Balanced dataset: {len(balanced)} examples across {len(balanced_data)} domains")
    return balanced

def concatenate_datasets(datasets: List[Dataset]) -> Dataset:
    """Concatenate multiple datasets"""
    if not datasets:
        return Dataset.from_dict({})
    
    combined = datasets[0]
    for ds in datasets[1:]:
        combined = concatenate_datasets([combined, ds])
    
    return combined

def validate_shard(shard_path: str, tokenizer: AutoTokenizer, max_samples: int = 100) -> Dict[str, Any]:
    """Quick validation of shard quality"""
    dataset = load_shard(shard_path, tokenizer)
    
    # Sample for validation
    sample = dataset.select(range(min(max_samples, len(dataset))))
    
    metrics = {
        'total_examples': len(dataset),
        'avg_prompt_length': sum(len(tokenizer.encode(p)) for p in sample['prompt']) / len(sample),
        'avg_completion_length': sum(len(tokenizer.encode(c)) for c, p in 
                                   zip(sample['completion'], sample['prompt'])) / len(sample),
        'domains': sample.unique('domain'),
        'avg_total_length': sum(len(ids) for ids in sample['input_ids']) / len(sample)
    }
    
    logger.info(f"Shard validation: {metrics}")
    return metrics

if __name__ == "__main__":
    # Test shard loading
    from transformers import AutoTokenizer
    
    # Dummy tokenizer for testing
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    
    # Create dummy shard for testing
    dummy_data = [
        {
            "prompt": "Solve step by step: If x + 3 = 7, what is x?",
            "completion": "x = 7 - 3 = 4",
            "domain": "math",
            "difficulty": "easy"
        },
        {
            "prompt": "Write a function to reverse a string:",
            "completion": "def reverse_string(s): return s[::-1]",
            "domain": "code",
            "difficulty": "easy"
        }
    ]
    
    # Save dummy shard
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        for item in dummy_data:
            f.write(json.dumps(item) + '\n')
        shard_path = f.name
    
    # Test loading
    dataset = load_shard(shard_path, tokenizer)
    print(f"Loaded {len(dataset)} examples")
    print("Sample:", tokenizer.decode(dataset[0]['input_ids']))
    
    os.unlink(shard_path)
    print("✅ Dataset loading working!")