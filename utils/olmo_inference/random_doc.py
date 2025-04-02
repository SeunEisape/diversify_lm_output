import os
import gzip
import json
import glob
import random
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch.nn.functional as F
import math

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
    return record.get("text", "")

if __name__ == "__main__":
    # Set up data directory; change as needed
    data_dir = os.getenv("DATA_DIR", "/home/eisape/projects/diversify_lm_output/dolma/data")
    
    # Get the sampled text from one document
    sampled_text = get_sample_text(data_dir)
    if not sampled_text:
        print("No sampled text found. Proceeding with default prompt.")
    
    # Define your original language model prompt
    # original_prompt = (
    #     "The United States of America (USA), also known as the United States (U.S.) or America, "
    #     "is a country primarily located in North America. It"
    # )
    original_prompt = ("A numbered list of 100 new research projects in natural language processing: "
              "1. diversyfying the open source language model output "
              "2. Finding correlation between the human brain and language models "
              "3. ")
    
    # Prepend the sampled text (if any) to your original prompt.
    # You can add a newline between them if desired.
    prompt = sampled_text + "\n" + original_prompt if sampled_text else original_prompt
    print("Combined prompt:\n", prompt)
    
    # Setup your language model and tokenizer
    MODELS = {
        "olmo-1b": "allenai/OLMo-1B-0724-hf",
        "olmo-2-7b": "allenai/OLMo-2-1124-7B",
        "olmo-2-13b": "allenai/OLMo-2-1124-13B"
    }
    model_name = MODELS["olmo-2-7b"]
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Tokenize the prompt
    inputs = tokenizer(prompt, return_tensors='pt', return_token_type_ids=False)

    # Move model and inputs to CUDA if available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    inputs = {k: v.to(device) for k, v in inputs.items()}
    model = model.to(device)

    # Generate text with output scores
    generation_output = model.generate(
        **inputs,
        max_new_tokens=100,
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

    print(f"\nAverage entropy of generated tokens: {avg_entropy:.4f}")
    print(f"Average perplexity of generated tokens: {avg_perplexity:.4f}")