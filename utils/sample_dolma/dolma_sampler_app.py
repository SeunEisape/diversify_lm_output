import streamlit as st
import random
import os
import gzip
import json

def main():
    st.title("Dolma Dataset Sampler")

    # Provide a numeric input for how many samples to display
    num_docs = st.number_input(
        label="How many documents to sample?", 
        min_value=1, 
        max_value=50, 
        value=5, 
        step=1
    )

    if st.button("Sample Documents"):
        # Construct the file path relative to the script's directory
        script_dir = os.path.dirname(os.path.realpath(__file__))
        file_path = os.path.join(script_dir, "v1_5r2_sample-0000.json.gz")

        # Check if the file exists
        if not os.path.exists(file_path):
            st.error(f"File not found: {file_path}. Please ensure the file is in the same directory as your script.")
            return

        st.write(f"Using file: `{file_path}`")

        # Read all lines from the chosen file
        try:
            with gzip.open(file_path, "rt", encoding="utf-8") as f:
                lines = f.readlines()
        except Exception as e:
            st.error(f"Error reading the file: {e}")
            return

        if not lines:
            st.warning("No documents found in the selected file.")
            return

        # Randomly pick and display the requested number of documents
        st.write(f"Displaying {num_docs} randomly selected document(s) from this file.")
        for i in range(num_docs):
            sampled_line = random.choice(lines)
            try:
                record = json.loads(sampled_line)
                st.subheader(f"Document {i+1}")
                st.json(record)
            except json.JSONDecodeError:
                st.warning("Could not parse line as JSON:")
                st.text(sampled_line)

if __name__ == "__main__":
    main()