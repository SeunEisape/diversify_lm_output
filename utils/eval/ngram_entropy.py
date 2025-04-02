import json
import collections
import math
import matplotlib.pyplot as plt
import argparse
from pathlib import Path

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

def calculate_entropy(jsonl_path, max_n=5):
    """
    Calculate n-gram entropy statistics from a jsonl file.
    
    Args:
        jsonl_path (str): Path to the jsonl file
        max_n (int): Maximum n-gram size to calculate
        
    Returns:
        dict: Dictionary containing entropy statistics for each n-gram size
    """
    # Create a counter for each n-gram level
    ngram_counters = {n: collections.Counter() for n in range(1, max_n + 1)}
    
    try:
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    text = data.get("completion_only", "")
                    for n in range(1, max_n + 1):
                        ngrams = get_ngrams(text, n)
                        ngram_counters[n].update(ngrams)
                except json.JSONDecodeError:
                    print(f"Warning: Could not parse line in {jsonl_path}")
                    continue
    except Exception as e:
        print(f"Error reading file {jsonl_path}: {str(e)}")
        return {}
    
    # Compute statistics for each n-gram level
    stats = {}
    for n in range(1, max_n + 1):
        entropy = compute_entropy(ngram_counters[n])
        max_possible = math.log(len(ngram_counters[n]), 2) if len(ngram_counters[n]) > 0 else 0
        normalized_entropy = entropy / max_possible if max_possible > 0 else 0
        
        stats[n] = {
            'entropy': entropy,
            'max_entropy': max_possible,
            'normalized_entropy': normalized_entropy,
            'unique_ngrams': len(ngram_counters[n])
        }
    
    return stats

def plot_entropy(stats, filename):
    """
    Plot the entropy statistics.
    
    Args:
        stats (dict): Dictionary containing entropy statistics
        filename (str): Name of the file being analyzed (for plot title)
    """
    if not stats:
        return
    
    # Extract data for plotting
    n_values = list(stats.keys())
    normalized_entropies = [stats[n]['normalized_entropy'] for n in n_values]
    
    # Create line plot for normalized entropies
    plt.figure(figsize=(10, 6))
    plt.plot(n_values, normalized_entropies, marker='o', linestyle='-', linewidth=2, markersize=8)
    plt.xlabel('N-gram', fontsize=12)
    plt.ylabel('Normalized Entropy', fontsize=12)
    plt.title(f'Normalized Entropy by N-gram Size for {filename}', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(n_values)
    plt.ylim(0, 1.1)  # Normalized entropy is between 0 and 1
    
    # Add values as text above each point
    for i, value in enumerate(normalized_entropies):
        plt.text(n_values[i], value + 0.02, f'{value:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(f"{filename}_ngram_entropy.png")
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Calculate n-gram entropy statistics from a jsonl file')
    parser.add_argument('--jsonl_file', type=str, required=True, help='Path to the jsonl file')
    parser.add_argument('--max_n', type=int, default=5, help='Maximum n-gram size to calculate')
    args = parser.parse_args()
    
    # Calculate statistics
    stats = calculate_entropy(args.jsonl_file, args.max_n)
    
    if not stats:
        print(f"No valid data found in {args.jsonl_file}")
        return
    
    # Print statistics
    print(f"N-gram Entropy Statistics for '{Path(args.jsonl_file).name}':")
    for n, stat in stats.items():
        gram_name = "unigrams" if n == 1 else f"{n}-grams"
        print(f"\n{gram_name.capitalize()}:")
        print(f"Entropy: {stat['entropy']:.3f}")
        print(f"Maximum possible entropy: {stat['max_entropy']:.3f}")
        print(f"Normalized entropy: {stat['normalized_entropy']:.3f}")
        print(f"Number of unique {gram_name}: {stat['unique_ngrams']}")
    
    # Plot and save the results
    filename = Path(args.jsonl_file).stem
    plot_entropy(stats, filename)
    print(f"\nPlot saved as '{filename}_ngram_entropy.png'")

if __name__ == "__main__":
    main()