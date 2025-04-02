import json
import collections
import math
import matplotlib.pyplot as plt

def get_ngrams(text, n):
    """Split text into n-grams based on whitespace tokenization."""
    tokens = text.split()
    return [' '.join(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]

def compute_entropy(counter):
    """Compute the Shannon entropy of a frequency counter."""
    total = sum(counter.values())
    entropy = 0.0
    for count in counter.values():
        p = count / total
        entropy -= p * math.log(p, 2)  # using base-2 logarithm
    return entropy

def main():
    files = ["/home/eisape/projects/diversify_lm_output/haiku_normal_prompt_output.jsonl" ]
    
    # For plotting
    all_entropies = []
    all_max_entropies = []
    all_normalized_entropies = []
    n_values = []
    
    for file in files:    
        max_n = 5  # Adjust this to compute 1-grams, 2-grams, 3-grams, etc.
        
        # Create a counter for each n-gram level
        ngram_counters = {n: collections.Counter() for n in range(1, max_n + 1)}
        
        with open(file, 'r') as f:
            for line in f:
                data = json.loads(line)
                text = data.get("completion_only", "")
                for n in range(1, max_n + 1):
                    ngrams = get_ngrams(text, n)
                    ngram_counters[n].update(ngrams)
        
        # Compute and display the entropy for each n-gram level along with maximum and normalized entropy
        for n in range(1, max_n + 1):
            entropy = compute_entropy(ngram_counters[n])
            # Maximum entropy assuming a uniform distribution over the unique n-grams
            max_possible = math.log(len(ngram_counters[n]), 2) if len(ngram_counters[n]) > 0 else 0
            normalized_entropy = entropy / max_possible if max_possible > 0 else 0

            print(f"Entropy for {n}-grams: {entropy}")
            print(f"Maximum entropy for {n}-grams: {max_possible}")
            print(f"Normalized entropy for {n}-grams: {normalized_entropy}\n")
            
            # Store values for plotting
            n_values.append(n)
            all_entropies.append(entropy)
            all_max_entropies.append(max_possible)
            all_normalized_entropies.append(normalized_entropy)
    
    # Create line plot for normalized entropies
    plt.figure(figsize=(10, 6))
    plt.plot(n_values, all_normalized_entropies, marker='o', linestyle='-', linewidth=2, markersize=8)
    plt.xlabel('N-gram', fontsize=12)
    plt.ylabel('Normalized Entropy', fontsize=12)
    plt.title('Normalized Entropy by N-gram Size', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(n_values)
    plt.ylim(0, 1.1)  # Normalized entropy is between 0 and 1
    
    # Add values as text above each point
    for i, value in enumerate(all_normalized_entropies):
        plt.text(n_values[i], value + 0.02, f'{value:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('ngram_normalized_entropy.png')
    plt.show()

if __name__ == "__main__":
    main()