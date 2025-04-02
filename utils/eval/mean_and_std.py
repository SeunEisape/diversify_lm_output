import json
import numpy as np
import argparse
from pathlib import Path
from collections import Counter
import matplotlib.pyplot as plt
# TODO: figure out how to view raw n-gram counts (i.e. the n-gram next to its count)
# TODO: Figure out what other people are counting the n-grams as: words? tokens? characters?


def calculate_stats(jsonl_path, ngram_range=(1, 5)):
    """
    Calculate the mean and standard deviation of text length in the 'completion_only' field
    from a jsonl file, in characters, words, and n-grams.
    
    Args:
        jsonl_path (str): Path to the jsonl file
        ngram_range (tuple): Range of n-grams to calculate (min_n, max_n)
        
    Returns:
        tuple: (mean_char_length, std_char_deviation, mean_word_length, std_word_deviation, 
                ngram_stats, count)
    """
    char_lengths = []
    word_lengths = []
    ngram_counts = {n: [] for n in range(ngram_range[0], ngram_range[1] + 1)}
    # Read the jsonl file
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                if 'completion_only' in data:
                    text = data['completion_only']
                    char_lengths.append(len(text))
                    words = text.split()
                    word_lengths.append(len(words))
                    
                    # Calculate n-gram statistics for each n in the range
                    for n in range(ngram_range[0], ngram_range[1] + 1):
                        if len(words) >= n:
                            ngrams = [' '.join(words[i:i+n]) for i in range(len(words)-n+1)]
                            unique_ngrams = len(Counter(ngrams))
                            ngram_counts[n].append(unique_ngrams)
                        else:
                            ngram_counts[n].append(0)
            except json.JSONDecodeError:
                print(f"Warning: Could not parse line: {line[:50]}...")
                continue
    # print unique n-gram counts
    for n in range(ngram_range[0], ngram_range[1] + 1):
        print(f"Unique {n}-grams: {len(ngram_counts[n])}")
    
    if not char_lengths:
        return 0, 0, 0, 0, {}, 0
    
    mean_char_length = np.mean(char_lengths)
    std_char_deviation = np.std(char_lengths)
    mean_word_length = np.mean(word_lengths)
    std_word_deviation = np.std(word_lengths)
    
    # Calculate mean and std for each n-gram size
    ngram_stats = {}
    for n in range(ngram_range[0], ngram_range[1] + 1):
        if ngram_counts[n]:
            ngram_stats[n] = {
                'mean': np.mean(ngram_counts[n]),
                'std': np.std(ngram_counts[n])
            }
    
    return mean_char_length, std_char_deviation, mean_word_length, std_word_deviation, ngram_stats, len(char_lengths)

def plot_statistics(mean_char_length, std_char_deviation, mean_word_length, std_word_deviation, ngram_stats, jsonl_path):
    """
    Plot the statistics as bar graphs with error bars.
    
    Args:
        mean_char_length (float): Mean character length
        std_char_deviation (float): Standard deviation of character length
        mean_word_length (float): Mean word length
        std_word_deviation (float): Standard deviation of word length
        ngram_stats (dict): Dictionary containing n-gram statistics
        jsonl_path (str): Path to the input JSONL file
    """
    # Create graphs directory in the same location as the input file
    input_path = Path(jsonl_path)
    graphs_dir = input_path.parent / "graphs"
    graphs_dir.mkdir(exist_ok=True)
    
    # Get filename without extension for plot titles
    filename = input_path.stem
    
    # Plot character and word length statistics
    plt.figure(figsize=(10, 6))
    labels = ['Characters', 'Words']
    means = [mean_char_length, mean_word_length]
    stds = [std_char_deviation, std_word_deviation]
    
    plt.bar(labels, means, yerr=stds, capsize=10, color=['blue', 'green'])
    plt.title(f'Text Length Statistics for {filename}')
    plt.ylabel('Mean Length')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.ylim(0, 2000)  # Set fixed y-axis limit to 2000
    plt.tight_layout()
    plt.savefig(graphs_dir / f"{filename}_length_stats.png")
    plt.close()
    
    # Plot n-gram statistics
    if ngram_stats:
        plt.figure(figsize=(12, 6))
        n_values = list(ngram_stats.keys())
        means = [ngram_stats[n]['mean'] for n in n_values]
        stds = [ngram_stats[n]['std'] for n in n_values]
        
        labels = ["unigrams" if n == 1 else f"{n}-grams" for n in n_values]
        
        plt.bar(labels, means, yerr=stds, capsize=10, color='orange')
        plt.title(f'N-gram Statistics for {filename}')
        plt.ylabel('Mean Unique N-grams')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.ylim(0, 400)  # Set fixed y-axis limit to 400
        plt.tight_layout()
        plt.savefig(graphs_dir / f"{filename}_ngram_stats.png")
        plt.close()

def main():
    parser = argparse.ArgumentParser(description='Calculate text statistics from a jsonl file')
    parser.add_argument('--jsonl_file', type=str, help='Path to the jsonl file')
    parser.add_argument('--min_n', type=int, default=1, help='Minimum n-gram size')
    parser.add_argument('--max_n', type=int, default=5, help='Maximum n-gram size')
    args = parser.parse_args()
    
    # Use the provided argument if available, otherwise use a hardcoded path
    jsonl_file = args.jsonl_file if args.jsonl_file else "random_prompt_outputs.jsonl"
    ngram_range = (args.min_n, args.max_n)
    
    mean_char_length, std_char_deviation, mean_word_length, std_word_deviation, ngram_stats, count = calculate_stats(jsonl_file, ngram_range)
    
    print(f"Text Statistics for '{Path(jsonl_file).name}':")
    print(f"Mean length: {mean_char_length:.2f} characters")
    print(f"Standard deviation: {std_char_deviation:.2f} characters")
    print(f"Mean length: {mean_word_length:.2f} words")
    print(f"Standard deviation: {std_word_deviation:.2f} words")
    
    # Print n-gram statistics
    for n, stats in ngram_stats.items():
        gram_name = "unigrams" if n == 1 else f"{n}-grams"
        print(f"Mean unique {gram_name}: {stats['mean']:.2f}")
        print(f"Standard deviation of unique {gram_name}: {stats['std']:.2f}")
    
    print(f"Number of samples analyzed: {count}")
    
    # Plot the statistics
    plot_statistics(mean_char_length, std_char_deviation, mean_word_length, std_word_deviation, ngram_stats, jsonl_file)
    print(f"Plots saved in the 'graphs' directory next to the input file")

if __name__ == "__main__":
    main()
