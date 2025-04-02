#!/usr/bin/env python3
"""
Script to run inference with random document prompts for all entries in the prompt store.
This runs the many_random_doc.py script for each prompt key in sequence.
"""

import os
import sys
import subprocess
import argparse

# Add the parent directory to sys.path to import prompt_store
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from prompt_store import get_all_prompts

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Run random document inference for all prompts in the prompt store")
    parser.add_argument("--num_completions", type=int, default=30, 
                        help="Number of completions to generate per prompt (default: 30)")
    parser.add_argument("--max_tokens", type=int, default=500,
                        help="Maximum number of tokens to generate (default: 500)")
    parser.add_argument("--data_dir", type=str, 
                        default="/home/eisape/projects/diversify_lm_output/dolma/data", 
                        help="Directory containing data files")
    parser.add_argument("--skip", type=str, nargs='+', default=[],
                        help="List of prompt keys to skip")
    args = parser.parse_args()
    
    # Get all prompts from the prompt store
    prompt_bank = get_all_prompts()
    prompt_keys = list(prompt_bank.keys())
    
    print(f"Found {len(prompt_keys)} prompts in the prompt store: {prompt_keys}")
    print(f"Will skip: {args.skip}")
    
    # Create the output directory structure
    os.makedirs("completions_eval_store", exist_ok=True)
    
    # Run many_random_doc.py for each prompt key
    for i, prompt_key in enumerate(prompt_keys):
        if prompt_key in args.skip:
            print(f"Skipping prompt: {prompt_key}")
            continue
            
        print(f"\n[{i+1}/{len(prompt_keys)}] Running inference for prompt: {prompt_key}")
        
        # Create directory for this prompt if it doesn't exist
        os.makedirs(f"completions_eval_store/{prompt_key}", exist_ok=True)
        
        # Construct the command to run
        cmd = [
            "python", 
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "many_random_doc.py"),
            "--prompt", prompt_key,
            "--num_completions", str(args.num_completions),
            "--max_tokens", str(args.max_tokens),
            "--data_dir", args.data_dir
        ]
        
        # Run the command
        print(f"Executing: {' '.join(cmd)}")
        try:
            subprocess.run(cmd, check=True)
            print(f"Completed inference for prompt: {prompt_key}")
        except subprocess.CalledProcessError as e:
            print(f"Error running inference for prompt: {prompt_key}")
            print(f"Error: {e}")
    
    print("\nCompleted inference for all prompts!")

if __name__ == "__main__":
    main() 