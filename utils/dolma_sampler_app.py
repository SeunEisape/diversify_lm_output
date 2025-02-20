import streamlit as st
import random
import datasets
from datasets import load_dataset

def main():
    st.title("Dolma Dataset Sampler")

    # Provide a slider or numeric input for the user to choose the number of samples
    num_docs = st.number_input(
        label="How many documents to sample?", 
        min_value=1, 
        max_value=50, 
        value=5, 
        step=1
    )

    if st.button("Sample Documents"):
        st.write("Loading the Dolma dataset...")
        dataset = load_dataset("allenai/dolma", split="train")

        st.write("Sampling documents...")
        # Shuffle with a random seed, then select the first N
        ds_shuffled = dataset.shuffle(seed=random.randint(0, 2**32 - 1))
        sampled_docs = ds_shuffled.select(range(num_docs))

        # Display the sampled documents
        for i, doc in enumerate(sampled_docs):
            st.subheader(f"Document {i+1}")
            st.json(doc)  # st.json() nicely formats a dictionary

if __name__ == "__main__":
    main()