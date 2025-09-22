"""
Phase 1: Generate reasoning traces using teacher models
Supports multiple teachers and domains
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from datasets import load_dataset
import json
from typing import List, Dict, Any, Optional
import logging
from tqdm import tqdm
import os

logger = logging.getLogger(__name__)

# Teacher model configurations
TEACHER_MODELS = {
    "deepseek-reasoning": {
        "model_name": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        "prompt_template": "Solve step by step: {problem}\n\nReasoning: ",
        "max_new_tokens": 256,
        "temperature": 0.1
    },
    "qwen-coder": {
        "model_name": "Qwen/Qwen2.5-Coder-7B-Instruct", 
        "prompt_template": "Write a complete solution for: {problem}\n\n```python\n",
        "max_new_tokens": 512,
        "temperature": 0.1
    }
}

DOMAINS = {
    "math": {
        "dataset": "gsm8k",
        "split": "main",
        "field": "question",
        "num_samples": 1000
    },
    "science": {
        "dataset": "sciq",
        "split": "train",
        "field": "question", 
        "num_samples": 800
    },
    "code": {
        "dataset": "openai_humaneval",
        "split": "test",
        "field": "prompt",
        "num_samples": 500
    },
    "logic": {
        "dataset": "allenai/ai2_arc",
        "split": "ARC-Challenge",
        "field": "question",
        "num_samples": 600
    }
}

def setup_teacher_model(model_config: Dict[str, Any]) -> tuple:
    """Initialize teacher model with 4-bit quantization for Colab"""
    model_name = model_config["model_name"]
    
    # 4-bit quantization config
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quant_config,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True
    )
    
    logger.info(f"Loaded teacher model: {model_name}")
    return model, tokenizer

def generate_reasoning_traces(
    domain: str,
    teacher_key: str,
    output_path: str,
    num_samples: int = 100,
    batch_size: int = 4
) -> List[Dict[str, Any]]:
    """
    Generate reasoning traces for a specific domain using a teacher model
    """
    if domain not in DOMAINS:
        raise ValueError(f"Unknown domain: {domain}. Available: {list(DOMAINS.keys())}")
    
    if teacher_key not in TEACHER_MODELS:
        raise ValueError(f"Unknown teacher: {teacher_key}. Available: {list(TEACHER_MODELS.keys())}")
    
    domain_config = DOMAINS[domain]
    teacher_config = TEACHER_MODELS[teacher_key]
    
    # Load base dataset
    logger.info(f"Loading {domain} dataset...")
    dataset = load_dataset(domain_config["dataset"], domain_config["split"])
    base_data = dataset[domain_config["field"]][:num_samples]
    
    # Setup teacher
    model, tokenizer = setup_teacher_model(teacher_config)
    
    traces = []
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    for i in tqdm(range(0, len(base_data), batch_size), desc=f"Generating {domain} traces"):
        batch_problems = base_data[i:i+batch_size]
        
        for problem in batch_problems:
            prompt = teacher_config["prompt_template"].format(problem=problem)
            
            # Tokenize prompt
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(model.device)
            
            # Generate
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=teacher_config["max_new_tokens"],
                    temperature=teacher_config["temperature"],
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
            
            # Decode generation
            generated_text = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            
            # Clean up generation
            if domain == "code":
                # Extract code block
                if "```" in generated_text:
                    generated_text = generated_text.split("```")[1].strip()
            
            trace = {
                "prompt": problem,
                "completion": generated_text.strip(),
                "domain": domain,
                "teacher": teacher_key,
                "difficulty": "medium",  # Could enhance with heuristics
                "timestamp": str(torch.timestamp(torch.tensor([0])).item())
            }
            
            traces.append(trace)
    
    # Save as JSONL
    with open(output_path, 'w') as f:
        for trace in traces:
            f.write(json.dumps(trace) + '\n')
    
    logger.info(f"Generated {len(traces)} traces: {output_path}")
    return traces

def batch_generate_traces(
    domains: List[str],
    teachers: List[str],
    output_dir: str,
    samples_per_combo: int = 200
):
    """
    Generate traces across multiple domain-teacher combinations
    """
    os.makedirs(output_dir, exist_ok=True)
    
    all_traces = []
    for domain in domains:
        for teacher in teachers:
            output_path = os.path.join(
                output_dir, f"traces_{domain}_{teacher}_{samples_per_combo}.jsonl"
            )
            
            traces = generate_reasoning_traces(
                domain=domain,
                teacher_key=teacher,
                output_path=output_path,
                num_samples=samples_per_combo
            )
            all_traces.extend(traces)
    
    return all_traces

if __name__ == "__main__":
    # Test data generation
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # Generate math traces with DeepSeek
    traces = generate_reasoning_traces(
        domain="math",
        teacher_key="deepseek-reasoning",
        output_path="/tmp/test_math_traces.jsonl",
        num_samples=5  # Small test
    )
    
    print(f"Generated {len(traces)} test traces")
    for trace in traces[:2]:  # Show first 2
        print(f"Prompt: {trace['prompt'][:100]}...")
        print(f"Completion: {trace['completion'][:100]}...")
        print("-" * 50)
    
    print("✅ Data generation working!")