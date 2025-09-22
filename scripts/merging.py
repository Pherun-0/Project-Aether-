"""
Weighted merging of contributor checkpoints into master model
Alpha=0.9 protects against bad contributions
"""

import torch
import os
import glob
from typing import List, Dict, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

def get_new_contributions(contributions_dir: str) -> List[str]:
    """Get list of new contribution checkpoints"""
    pattern = os.path.join(contributions_dir, "contrib_*_shard_*.pt")
    contributions = glob.glob(pattern)
    # Filter only completed contributions (not in progress)
    completed = [c for c in contributions if "_tmp" not in c]
    return sorted(completed)

def load_state_dicts(checkpoint_paths: List[str]) -> List[Dict[str, torch.Tensor]]:
    """Load state dicts from contribution checkpoints"""
    state_dicts = []
    for path in checkpoint_paths:
        try:
            sd = torch.load(path, map_location='cpu')
            state_dicts.append(sd)
            logger.info(f"Loaded contribution: {path}")
        except Exception as e:
            logger.warning(f"Failed to load {path}: {e}")
    return state_dicts

def weighted_merge(
    master_state_dict: Dict[str, torch.Tensor],
    contribution_state_dicts: List[Dict[str, torch.Tensor]],
    alpha: float = 0.9,
    min_contributions: int = 3
) -> Dict[str, torch.Tensor]:
    """
    Perform weighted average merge of contributions into master
    
    Args:
        master_state_dict: Current master model weights
        contribution_state_dicts: List of contributor state dicts
        alpha: Weight for master (1-alpha for average of contributions)
        min_contributions: Minimum valid contributions required
    
    Returns:
        New merged state dict
    """
    if len(contribution_state_dicts) < min_contributions:
        logger.warning(f"Only {len(contribution_state_dicts)} contributions, need {min_contributions}")
        return master_state_dict
    
    # Average the contributions
    averaged_contrib = {}
    common_keys = set(master_state_dict.keys())
    
    for key in master_state_dict.keys():
        if key not in common_keys:
            continue
            
        # Stack all contribution tensors for this parameter
        contrib_tensors = [sd[key] for sd in contribution_state_dicts if key in sd]
        
        if len(contrib_tensors) == 0:
            averaged_contrib[key] = master_state_dict[key]
            continue
            
        # Compute mean across contributions
        stacked = torch.stack(contrib_tensors)
        avg_contrib = stacked.mean(dim=0)
        
        # Weighted combination: alpha * master + (1-alpha) * avg_contrib
        merged = alpha * master_state_dict[key] + (1 - alpha) * avg_contrib
        averaged_contrib[key] = merged
    
    logger.info(f"Merged {len(contribution_state_dicts)} contributions with alpha={alpha}")
    return averaged_contrib

def merge_contributions(
    master_path: str,
    contributions_dir: str,
    output_path: str,
    alpha: float = 0.9,
    archive_contributions: bool = True
):
    """
    Main merge function: Load master, merge new contributions, save new master
    """
    # Load current master
    if not os.path.exists(master_path):
        logger.warning(f"Master not found at {master_path}, creating random init")
        from .model import ReasoningTransformer
        model = ReasoningTransformer()
        master_sd = model.state_dict()
    else:
        master_sd = torch.load(master_path, map_location='cpu')
    
    # Get new contributions
    contrib_paths = get_new_contributions(contributions_dir)
    if not contrib_paths:
        logger.info("No new contributions to merge")
        return
    
    # Load contribution state dicts
    contrib_sds = load_state_dicts(contrib_paths)
    
    # Perform weighted merge
    new_master_sd = weighted_merge(master_sd, contrib_sds, alpha=alpha)
    
    # Save new master
    torch.save(new_master_sd, output_path)
    logger.info(f"Saved new master to {output_path}")
    
    # Archive processed contributions
    if archive_contributions:
        archive_dir = os.path.join(contributions_dir, "archive")
        os.makedirs(archive_dir, exist_ok=True)
        for path in contrib_paths:
            contrib_name = Path(path).name
            new_path = os.path.join(archive_dir, f"{contrib_name}")
            os.rename(path, new_path)
            logger.info(f"Archived: {path} -> {new_path}")
    
    # Log merge summary
    logger.info(f"Merge complete: {len(contrib_paths)} contributions processed")

if __name__ == "__main__":
    # Example usage for testing
    import logging
    logging.basicConfig(level=logging.INFO)
    
    MASTER_PATH = "/content/drive/MyDrive/Colab_Reasoning_Model/_master_model/master.pt"
    CONTRIBUTIONS_DIR = "/content/drive/MyDrive/Colab_Reasoning_Model/contributions"
    OUTPUT_PATH = "/content/drive/MyDrive/Colab_Reasoning_Model/_master_model/master_v2.pt"
    
    merge_contributions(MASTER_PATH, CONTRIBUTIONS_DIR, OUTPUT_PATH, alpha=0.9)