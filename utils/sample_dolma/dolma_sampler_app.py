import streamlit as st
import random
import os
import gzip
import json
import glob

def main():
    st.title("Dolma Dataset Sampler")

    # Let the user provide the local directory containing .json.gz files
    data_dir = st.text_input(
        "Enter the path to your Dolma data directory:",
        "seuneisape/diversify_lm_output/main/utils/random_sample_of_data"
    )

    # Provide a numeric input for how many samples to display
    num_docs = st.number_input(
        label="How many documents to sample?", 
        min_value=1, 
        max_value=50, 
        value=5, 
        step=1
    )

    if st.button("Sample Documents"):
        # Gather the list of .json.gz files
        file_list = glob.glob(os.path.join(data_dir, "*.json.gz"))
        
        if not file_list:
            st.warning(f"No .json.gz files found in {data_dir}")
            return
        
        # Randomly pick one file from the directory
        file_path = random.choice(file_list)
        st.write(f"Randomly selected file: `{file_path}`")

        # Read all lines from the chosen file
        with gzip.open(file_path, "rt", encoding="utf-8") as f:
            lines = f.readlines()

        if not lines:
            st.warning("No documents found in the selected file.")
            return

        # Randomly pick and display the requested number of documents
        st.write(f"Displaying {num_docs} randomly selected document(s) from this file.")
        for i in range(num_docs):
            # Randomly choose a line (could pick the same line more than once)
            sampled_line = random.choice(lines)
            try:
                record = json.loads(sampled_line)
                st.subheader(f"Document {i+1}")
                st.json(record)
            except json.JSONDecodeError:
                # If there's a parsing issue, just display the raw string
                st.warning("Could not parse line as JSON:")
                st.text(sampled_line)

if __name__ == "__main__":
    main()