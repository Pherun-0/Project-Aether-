"""
Checkpointing utilities for saving/loading model state
Supports partial saves and memory-efficient loading
"""

import torch
import os
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)

def save_checkpoint(
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    epoch: int = 0,
    loss: float = 0.0,
    save_path: str = "checkpoint.pt",
    save_optimizer: bool = True,
    include_model: bool = True
):
    """
    Save model checkpoint with optional optimizer state
    
    Args:
        model: PyTorch model
        optimizer: Optional optimizer state
        epoch: Current epoch
        loss: Current loss
        save_path: Output path
        save_optimizer: Whether to save optimizer state
        include_model: Whether to save model architecture (full pickle)
    """
    checkpoint = {
        'epoch': epoch,
        'loss': loss,
        'model_state_dict': model.state_dict() if include_model else None,
    }
    
    if save_optimizer and optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    
    torch.save(checkpoint, save_path)
    logger.info(f"Checkpoint saved to {save_path} (epoch={epoch}, loss={loss:.4f})")

def load_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: str,
    load_optimizer: bool = False,
    strict: bool = True,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> tuple[Optional[torch.optim.Optimizer], int, float]:
    """
    Load model checkpoint
    
    Args:
        model: Model to load weights into
        checkpoint_path: Path to checkpoint
        load_optimizer: Whether to return optimizer state
        strict: Whether to enforce strict loading
        device: Device to load to
    
    Returns:
        (optimizer, epoch, loss) or (None, 0, 0) if no optimizer
    """
    if not os.path.exists(checkpoint_path):
        logger.warning(f"Checkpoint not found: {checkpoint_path}")
        return None, 0, float('inf')
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load model weights
    model_state = checkpoint.get('model_state_dict', None)
    if model_state is not None:
        model.load_state_dict(model_state, strict=strict)
        logger.info(f"Loaded model weights from {checkpoint_path}")
    
    # Extract metadata
    epoch = checkpoint.get('epoch', 0)
    loss = checkpoint.get('loss', float('inf'))
    
    # Load optimizer if requested
    optimizer = None
    if load_optimizer and 'optimizer_state_dict' in checkpoint:
        # Create dummy optimizer for loading (will be replaced later)
        from torch.optim import AdamW
        optimizer = AdamW(model.parameters(), lr=1e-5)  # Dummy
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        logger.info(f"Loaded optimizer state from {checkpoint_path}")
    
    return optimizer, epoch, loss

def save_state_dict_only(
    state_dict: Dict[str, torch.Tensor],
    save_path: str,
    metadata: Optional[Dict[str, Any]] = None
):
    """Save only state dict (lightweight - used for contributions)"""
    if metadata is None:
        metadata = {}
    
    save_data = {
        'state_dict': state_dict,
        'metadata': metadata,
        'timestamp': torch.timestamp(torch.tensor([0])).item()
    }
    
    torch.save(save_data, save_path)
    logger.info(f"Saved state dict to {save_path} ({len(state_dict)} parameters)")

def load_state_dict_only(checkpoint_path: str) -> Dict[str, torch.Tensor]:
    """Load only state dict (lightweight)"""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"State dict not found: {checkpoint_path}")
    
    data = torch.load(checkpoint_path, map_location='cpu')
    state_dict = data['state_dict']
    logger.info(f"Loaded state dict from {checkpoint_path}")
    return state_dict

def initialize_master_checkpoint(
    output_path: str,
    vocab_size: int = 32000,
    device: str = 'cpu'
):
    """Create initial random master checkpoint"""
    from .model import ReasoningTransformer
    
    model = ReasoningTransformer(vocab_size=vocab_size).to(device)
    state_dict = model.state_dict()
    
    save_state_dict_only(
        state_dict=state_dict,
        save_path=output_path,
        metadata={'initialized': True, 'epoch': 0, 'vocab_size': vocab_size}
    )
    logger.info(f"Initialized master checkpoint: {output_path}")

if __name__ == "__main__":
    # Test checkpointing
    import logging
    logging.basicConfig(level=logging.INFO)
    
    from .model import ReasoningTransformer
    
    # Test save/load
    model = ReasoningTransformer()
    save_path = "/tmp/test_checkpoint.pt"
    
    # Save
    save_state_dict_only(model.state_dict(), save_path)
    
    # Load
    loaded_sd = load_state_dict_only(save_path)
    print(f"Loaded {len(loaded_sd)} parameters")
    print("✅ Checkpointing utilities working!")