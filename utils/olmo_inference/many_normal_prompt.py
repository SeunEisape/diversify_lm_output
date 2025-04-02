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

def run_inference(model, tokenizer, prompt, device, max_new_tokens=100, do_sample=True, top_k=50, top_p=0.95):
    """Run inference on a single prompt and return the results"""
    # Tokenize the prompt
    inputs = tokenizer(prompt, return_tensors='pt', return_token_type_ids=False)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Generate text with output scores
    generation_output = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        top_k=top_k,
        top_p=top_p,
        output_scores=True,
        return_dict_in_generate=True
    )
    
    # Decode the full generated text
    generated_text = tokenizer.decode(generation_output.sequences[0], skip_special_tokens=True)
    
    # Calculate token-level metrics for the newly generated tokens
    prompt_length = inputs["input_ids"].shape[1]
    new_tokens = generation_output.sequences[0][prompt_length:]
    token_entropies = []
    token_perplexities = []
    
    token_details = []
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
        token_details.append({
            "token": token_text,
            "entropy": token_entropy,
            "perplexity": token_perplexity
        })
    
    avg_entropy = sum(token_entropies) / len(token_entropies) if token_entropies else 0.0
    avg_perplexity = sum(token_perplexities) / len(token_perplexities) if token_perplexities else 0.0
    
    # Extract just the generated completion (without the prompt)
    completion_only = generated_text[len(prompt):].strip()
    
    return {
        "full_output": generated_text,
        "completion_only": completion_only,
        "avg_token_entropy": avg_entropy,
        "avg_token_perplexity": avg_perplexity,
        "token_details": token_details
    }

if __name__ == "__main__":
    import argparse
    
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Generate text completions with normal prompts")
    parser.add_argument("--prompt", type=str, help="Custom prompt to use for generation")
    parser.add_argument("--data_dir", type=str, default="/home/eisape/projects/diversify_lm_output/dolma/data", 
                        help="Directory containing data files")
    parser.add_argument("--num_completions", type=int, default=300, 
                        help="Number of completions to generate")
    parser.add_argument("--max_tokens", type=int, default=500,
                        help="Maximum number of tokens to generate")
    args = parser.parse_args()
    
    # Set up variables from arguments
    num_completions = args.num_completions
    max_tokens = args.max_tokens
    
    # Instead of defining a prompt bank, use the centralized one
    # Print available prompts for reference
    prompt_bank = get_all_prompts()
    print("Available prompts:", list(prompt_bank.keys()))
    
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
    
    # Use the provided prompt if available, otherwise use default
    prompt_key = args.prompt if args.prompt else "default"
    original_prompt = get_prompt(prompt_key)
    
    print(f"Using prompt: {original_prompt}")
    
    # Define output file path based on the prompt name
    prompt_name = prompt_key
    output_dir = f"completions_eval_store/{prompt_name}"
    os.makedirs(output_dir, exist_ok=True)
    output_file = f"{output_dir}/{prompt_name}_normal_prompt_output.jsonl"
    
    # Generate multiple completions
    for completion_idx in range(num_completions):
        print(f"\n--- Generating completion {completion_idx+1}/{num_completions} ---\n")
        
        # Use only the original prompt without random samples
        prompt = original_prompt
        print("Prompt:\n", prompt)
        
        # Run inference
        inference_results = run_inference(
            model, 
            tokenizer, 
            prompt, 
            device, 
            max_new_tokens=max_tokens
        )
        
        print(f"\nGenerated text:\n{inference_results['full_output']}\n")
        print("Token-level metrics for generated tokens:")
        for token_info in inference_results['token_details']:
            print(f"Token: {repr(token_info['token'])} | Entropy: {token_info['entropy']:.4f} | Perplexity: {token_info['perplexity']:.4f}")
        
        # Create a dictionary to store the results
        result = {
            "prompt": prompt,
            "full_output": inference_results["full_output"],
            "completion_only": inference_results["completion_only"],
            "model": model_name,
            "completion_idx": completion_idx,
            "avg_token_entropy": inference_results["avg_token_entropy"],
            "avg_token_perplexity": inference_results["avg_token_perplexity"],
            "prompt_type": "normal_prompt"
        }
        
        # Append the result to the JSONL file
        with open(output_file, "a") as f:
            f.write(json.dumps(result) + "\n")
        
        print(f"Results saved to {output_file}")
        print(f"Average entropy of generated tokens: {inference_results['avg_token_entropy']:.4f}")
        print(f"Average perplexity of generated tokens: {inference_results['avg_token_perplexity']:.4f}")
    
    print(f"\nCompleted generating {num_completions} completions.")