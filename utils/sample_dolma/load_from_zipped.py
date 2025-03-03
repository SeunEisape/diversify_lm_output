import os
import gzip
import json
import glob
import random

"""
Chooses a SINGLE file to randoomly smaple num_docs lines from
"""
def random_sample_dolma_data(data_dir, num_docs):
    # randomly choose a file from data_dir
    file_list = glob.glob(os.path.join(data_dir, "*.json.gz"))
    if not file_list:
        print(f"No .json.gz files found in {data_dir}")
        return
    file_path = random.choice(file_list)
    print(f"Sampling from file: {file_path}")
    # Random sample a single line

    with gzip.open(file_path, "rt", encoding="utf-8") as f:
        lines = f.readlines()
        for i in range(num_docs):
            sampled_line = random.choice(lines)
            print(f"Sampled line: {sampled_line}")

def sample_dolma_data(data_dir, lines_to_sample=5):
    """
    Randomly samples 'lines_to_sample' records from each .json.gz file in the directory.
    
    Parameters:
      data_dir (str): Path to the directory containing .json.gz files.
      lines_to_sample (int): Number of lines (records) to randomly sample per file.
    """
    # Get list of all .json.gz files in data_dir
    file_list = glob.glob(os.path.join(data_dir, "*.json.gz"))
    if not file_list:
        print(f"No .json.gz files found in {data_dir}")
        return
    
    for file_path in file_list:
        print(f"\nSampling from file: {file_path}")
        
        # Read all lines into memory
        with gzip.open(file_path, "rt", encoding="utf-8") as f:
            lines = f.readlines()
        
        # If the file has fewer lines than lines_to_sample, sample all lines
        sample_size = min(lines_to_sample, len(lines))
        
        # Randomly choose 'sample_size' lines
        sampled_lines = random.sample(lines, sample_size)
        
        # Print the randomly sampled records
        for i, line in enumerate(sampled_lines, start=1):
            record = json.loads(line)
            print(f"  Sampled Record {i}: {record}")


if __name__ == "__main__":
    # Adjust DATA_DIR and lines_to_sample as needed
    data_dir = os.getenv("DATA_DIR", "/home/eisape/projects/diversify_lm_output/dolma/data")
    # sample_dolma_data(data_dir, lines_to_sample=3)
    random_sample_dolma_data(data_dir, num_docs=3)