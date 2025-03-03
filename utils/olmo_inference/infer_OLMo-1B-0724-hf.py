from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch.nn.functional as F
import math

if __name__ == "__main__":

    MODELS = {
        "olmo-1b": "allenai/OLMo-1B-0724-hf",
        "olmo-2-7b": "allenai/OLMo-2-1124-7B",
        "olmo-2-13b": "allenai/OLMo-2-1124-13B"
    }

    model_name = MODELS["olmo-2-7b"]
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # prompt = ("A numbered list of 100 new research projects in natural language processing: "
    #           "1. diversyfying the open source language model output "
    #           "2. Finding correlation between the human brain and language models "
    #           "3. ")
    prompt = "The United States of America (USA), also known as the United States (U.S.) or America, is a country primarily located in North America. It"

    # Tokenize the prompt
    inputs = tokenizer(prompt, return_tensors='pt', return_token_type_ids=False)

    # Move model and inputs to CUDA if available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    inputs = {k: v.to(device) for k, v in inputs.items()}
    model = model.to(device)

    # Generate text with output scores so we can inspect logits for new tokens.
    generation_output = model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        output_scores=True,             # get logits for each generated token
        return_dict_in_generate=True    # returns a dict with scores and sequences
    )

    # Decode the full generated text
    generated_text = tokenizer.decode(generation_output.sequences[0], skip_special_tokens=True)
    print(f"Generated text:\n{generated_text}\n")

    # Calculate token-level entropy and token-level perplexity for the newly generated tokens
    prompt_length = inputs["input_ids"].shape[1]
    new_tokens = generation_output.sequences[0][prompt_length:]
    token_entropies = []
    token_perplexities = []


    print("Token-level metrics for generated tokens:")
    for i, score in enumerate(generation_output.scores):

        # Convert logits to probabilities for the current timestep
        probs = F.softmax(score, dim=-1)
        # Compute entropy for this token using a categorical distribution (batch size is 1)
        entropy = torch.distributions.Categorical(probs=probs).entropy()
        token_entropy = entropy.item()  # scalar value

        # Calculate token perplexity as the exponential of the entropy
        token_perplexity = math.exp(token_entropy)

        # Decode the generated token

        token_id = new_tokens[i].item()
        token_text = tokenizer.decode(token_id)

        token_entropies.append(token_entropy)
        token_perplexities.append(token_perplexity)
        print(f"Token: {repr(token_text)} | Entropy: {token_entropy:.4f} | Perplexity: {token_perplexity:.4f}")

    # Compute and display the average entropy and average perplexity
    avg_entropy = sum(token_entropies) / len(token_entropies) if token_entropies else 0.0
    avg_perplexity = sum(token_perplexities) / len(token_perplexities) if token_perplexities else 0.0

    print(f"\nAverage entropy of generated tokens: {avg_entropy:.4f}")
    print(f"Average perplexity of generated tokens: {avg_perplexity:.4f}")

"""
Script to infer OLMO LMs
"""
# from transformers import AutoModelForCausalLM, AutoTokenizer
# import torch
# import torch.nn.functional as F

# if __name__ == "__main__":

#     MODELS = {
#         "olmo-1b": "allenai/OLMo-1B-0724-hf",
#         "olmo-2-7b": "allenai/OLMo-2-1124-7B"
#     }

#     model = AutoModelForCausalLM.from_pretrained("allenai/OLMo-1B-0724-hf")
#     tokenizer = AutoTokenizer.from_pretrained("allenai/OLMo-1B-0724-hf")

#     prompt = "A numbered list of 100 new research projects in natural language processing: \
#     1. diversyfying the open source language model output \
#     2. Finding correlation between the human brain and language models \
#     3. "

#     # tokenize the prompt
#     inputs = tokenizer(prompt, return_tensors='pt', return_token_type_ids=False)

#     # verifying cuda
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
#     inputs = {k: v.to(device) for k, v in inputs.items()}
#     model = model.to(device)


#     response = model.generate(**inputs, max_new_tokens=100, do_sample=True, top_k=50, top_p=0.95)
#     generated_text = tokenizer.batch_decode(response, skip_special_tokens=True)[0]
#     print(f"Generated text: {generated_text}")


