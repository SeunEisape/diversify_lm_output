import os
import gzip
import json
import glob
import random
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch.nn.functional as F
import math
import sys

# Add the parent directory to sys.path to import prompt_store
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from prompt_store import get_prompt, get_all_prompts

def get_sample_text(data_dir):
    """
    Randomly selects one file from data_dir, randomly samples a single record,
    and returns the value of its "text" field.
    """
    file_list = glob.glob(os.path.join(data_dir, "*.json.gz"))
    if not file_list:
        print(f"No .json.gz files found in {data_dir}")
        return ""
    
    file_path = random.choice(file_list)
    print(f"Sampling from file: {file_path}")

    with gzip.open(file_path, "rt", encoding="utf-8") as f:
        lines = f.readlines()
    
    if not lines:
        print("No lines found in the file.")
        return ""
    
    sampled_line = random.choice(lines)
    try:
        record = json.loads(sampled_line)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        return ""
    
    # Extract the 'text' field from the record, or return empty string if missing
    # Also return the file path
    return record.get("text", ""), file_path

if __name__ == "__main__":
    import argparse
    
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Generate text completions with random documents")
    parser.add_argument("--prompt", type=str, help="Custom prompt to use for generation")
    parser.add_argument("--data_dir", type=str, default="/home/eisape/projects/diversify_lm_output/dolma/data", 
                        help="Directory containing data files")
    parser.add_argument("--num_completions", type=int, default=300, 
                        help="Number of completions to generate")
    parser.add_argument("--max_tokens", type=int, default=500,
                        help="Maximum number of tokens to generate")
    args = parser.parse_args()
    
    # Set up data directory
    data_dir = args.data_dir if args.data_dir else os.getenv("DATA_DIR", "/home/eisape/projects/diversify_lm_output/dolma/data")
    
    # Define number of completions to generate
    num_completions = args.num_completions
    max_tokens = args.max_tokens if hasattr(args, 'max_tokens') else 500
    
    # Instead of defining a prompt bank, use the centralized one
    # Print available prompts for reference
    prompt_bank = get_all_prompts()
    print("Available prompts:", list(prompt_bank.keys()))
    
    # Use the provided prompt if available, otherwise use default
    prompt_key = args.prompt if args.prompt else "default"
    original_prompt = get_prompt(prompt_key)
    
    print(f"Using prompt: {original_prompt}")

    # Setup your language model and tokenizer
    MODELS = {
        "olmo-1b": "allenai/OLMo-1B-0724-hf",
        "olmo-2-7b": "allenai/OLMo-2-1124-7B",
        "olmo-2-13b": "allenai/OLMo-2-1124-13B"
    }
    model_name = MODELS["olmo-2-7b"]
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Move model to CUDA if available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    
    # Define output file path
    output_dir = f"completions_eval_store/{prompt_key}"
    os.makedirs(output_dir, exist_ok=True)
    output_file = f"{output_dir}/{prompt_key}_random_prompt_output.jsonl"
    
    # Generate multiple completions
    for completion_idx in range(num_completions):
        print(f"\n--- Generating completion {completion_idx+1}/{num_completions} ---\n")
        
        # Get a new sampled text for each completion
        sampled_text, random_doc_file_path = get_sample_text(data_dir)
        if not sampled_text:
            print("No sampled text found. Proceeding with default prompt.")
        
        # Prepend the sampled text (if any) to your original prompt
        prompt = sampled_text + "\n" + original_prompt if sampled_text else original_prompt
        print("Combined prompt:\n", prompt)
        
        # Tokenize the prompt
        inputs = tokenizer(prompt, return_tensors='pt', return_token_type_ids=False)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Generate text with output scores
        generation_output = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            output_scores=True,
            return_dict_in_generate=True
        )
        
        # Decode the full generated text
        generated_text = tokenizer.decode(generation_output.sequences[0], skip_special_tokens=True)
        print(f"\nGenerated text:\n{generated_text}\n")
        
        # Calculate token-level metrics for the newly generated tokens
        prompt_length = inputs["input_ids"].shape[1]
        new_tokens = generation_output.sequences[0][prompt_length:]
        token_entropies = []
        token_perplexities = []
        
        print("Token-level metrics for generated tokens:")
        for i, score in enumerate(generation_output.scores):
            # Convert logits to probabilities for the current timestep
            probs = F.softmax(score, dim=-1)
            # Compute entropy for this token
            entropy = torch.distributions.Categorical(probs=probs).entropy()
            token_entropy = entropy.item()
            token_perplexity = math.exp(token_entropy)
            
            token_id = new_tokens[i].item()
            token_text = tokenizer.decode(token_id)
            token_entropies.append(token_entropy)
            token_perplexities.append(token_perplexity)
            print(f"Token: {repr(token_text)} | Entropy: {token_entropy:.4f} | Perplexity: {token_perplexity:.4f}")
        
        avg_entropy = sum(token_entropies) / len(token_entropies) if token_entropies else 0.0
        avg_perplexity = sum(token_perplexities) / len(token_perplexities) if token_perplexities else 0.0
        
        # Extract just the generated completion (without the prompt)
        completion_only = generated_text[len(prompt):].strip()
        
        # Create a dictionary to store the results
        result = {
            "random_doc_file_path": random_doc_file_path,
            "random_doc": sampled_text,
            "prompt": prompt,
            "original_prompt": original_prompt,
            "full_output": generated_text,
            "completion_only": completion_only,
            "model": model_name,
            "completion_idx": completion_idx,
            "avg_token_entropy": avg_entropy,
            "avg_token_perplexity": avg_perplexity,
            "prompt_type": "random_doc"
        }
        
        # Append the result to the JSONL file
        with open(output_file, "a") as f:
            f.write(json.dumps(result) + "\n")
        
        print(f"Results saved to {output_file}")
        print(f"Average entropy of generated tokens: {avg_entropy:.4f}")
        print(f"Average perplexity of generated tokens: {avg_perplexity:.4f}")
    
    print(f"\nCompleted generating {num_completions} completions with different random documents.")