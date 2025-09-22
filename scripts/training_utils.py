"""
Training utilities for shard-based distributed training
Includes evaluation and logging
"""

import torch
import torch.nn as nn
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import Dataset
import wandb
from typing import Dict, Any, Optional
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)

def train_shard(
    model: torch.nn.Module,
    dataset: Dataset,
    shard_id: str,
    output_dir: str,
    contributor_name: str,
    num_epochs: int = 1,
    batch_size: int = 4,
    learning_rate: float = 5e-5,
    weight_decay: float = 0.01,
    max_steps: Optional[int] = None,
    eval_samples: int = 100,
    wandb_project: str = "project-aether"
) -> Dict[str, float]:
    """
    Train model on a single shard with logging and evaluation
    """
    # Setup W&B
    wandb.init(
        project=wandb_project,
        name=f"{contributor_name}_shard_{shard_id}",
        config={
            "shard_id": shard_id,
            "contributor": contributor_name,
            "epochs": num_epochs,
            "batch_size": batch_size,
            "lr": learning_rate
        }
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=f"{output_dir}/tmp",
        overwrite_output_dir=True,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=4,  # Effective batch size = 16
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        fp16=True,
        logging_steps=10,
        save_steps=max_steps if max_steps else -1,
        save_total_limit=1,
        report_to="wandb",
        dataloader_num_workers=0,  # Colab compatibility
        remove_unused_columns=False,
        max_steps=max_steps,
        warmup_steps=100,
        load_best_model_at_end=True if eval_samples > 0 else False,
        evaluation_strategy="steps" if eval_samples > 0 else "no",
        eval_steps=50 if eval_samples > 0 else None,
        dataloader_pin_memory=False,  # Colab stability
    )
    
    # Data collator
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")  # Dummy, replace with actual
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # Causal LM
        pad_to_multiple_of=8
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    
    # Train
    logger.info(f"Starting training for shard {shard_id}")
    trainer.train()
    
    # Evaluation
    eval_results = {}
    if eval_samples > 0:
        eval_results = evaluate_model(model, dataset.select(range(eval_samples)), tokenizer)
    
    # Log final metrics
    final_metrics = {
        "train_loss": trainer.state.log_history[-1]["train_loss"] if trainer.state.log_history else 0.0,
        "epoch": trainer.state.epoch,
        **eval_results
    }
    
    wandb.log(final_metrics)
    wandb.finish()
    
    logger.info(f"Training complete for shard {shard_id}: {final_metrics}")
    return final_metrics

def evaluate_model(
    model: torch.nn.Module,
    eval_dataset: Dataset,
    tokenizer: Any,
    batch_size: int = 8,
    max_length: int = 512
) -> Dict[str, float]:
    """Quick evaluation on reasoning tasks"""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    dataloader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=batch_size,
        collate_fn=lambda x: tokenizer.pad(x, return_tensors="pt")
    )
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(model.device)
            labels = batch['labels'].to(model.device)
            
            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss if hasattr(outputs, 'loss') else outputs[1]
            
            total_loss += loss.item()
            num_batches += 1
    
    avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
    
    # Quick perplexity
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    
    results = {
        "eval_loss": avg_loss,
        "eval_perplexity": perplexity,
        "eval_samples": len(eval_dataset)
    }
    
    logger.info(f"Evaluation: loss={avg_loss:.4f}, perplexity={perplexity:.2f}")
    return results

def validate_contribution(
    contribution_path: str,
    master_path: str,
    test_dataset: Dataset,
    validation_threshold: float = 2.0
) -> bool:
    """
    Validate a contribution before merging
    Returns True if contribution improves or doesn't degrade performance
    """
    from .checkpointing import load_state_dict_only
    
    # Load models
    model = torch.nn.Module()  # Dummy, replace with actual model
    # TODO: Load actual model architecture
    
    master_sd = load_state_dict_only(master_path)
    contrib_sd = load_state_dict_only(contribution_path)
    
    # Load weights
    model.load_state_dict(master_sd)
    master_loss = evaluate_model(model, test_dataset)["eval_loss"]
    
    model.load_state_dict(contrib_sd)
    contrib_loss = evaluate_model(model, test_dataset)["eval_loss"]
    
    # Accept if contribution improves performance or degrades by less than threshold
    improvement = master_loss - contrib_loss
    is_valid = improvement > -validation_threshold
    
    logger.info(f"Validation: Master loss={master_loss:.4f}, Contrib loss={contrib_loss:.4f}, "
                f"delta={improvement:.4f}, valid={is_valid}")
    
    return is_valid

if __name__ == "__main__":
    # Test training utilities
    import logging
    logging.basicConfig(level=logging.INFO)
    
    print("Training utilities ready!")
    print("Use in conjunction with shard training notebooks")