import streamlit as st
import json
import os
from pathlib import Path
import glob
import time

# Set page config for cloud display
st.set_page_config(
    page_title="LM Completions Viewer",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/yourusername/diversify_lm_output',
        'Report a bug': "https://github.com/yourusername/diversify_lm_output/issues",
        'About': "# LM Completions Viewer\nView and analyze language model completions"
    }
)

# Base directory containing JSONL files
BASE_DIR = "completions_eval_store"

@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_jsonl_files():
    """Find all JSONL files in the completions_eval_store directory and its subdirectories."""
    jsonl_files = []
    try:
        for root, _, files in os.walk(BASE_DIR):
            for file in files:
                if file.endswith('.jsonl'):
                    full_path = os.path.join(root, file)
                    # Get relative path for display
                    rel_path = os.path.relpath(full_path, BASE_DIR)
                    jsonl_files.append((rel_path, full_path))
    except Exception as e:
        st.error(f"Error finding JSONL files: {str(e)}")
        return []
    return sorted(jsonl_files)

@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_jsonl_file(file_path):
    """Load and parse a JSONL file."""
    completions = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    if 'completion_only' in data:
                        completions.append(data['completion_only'])
                except json.JSONDecodeError:
                    st.warning(f"Could not parse line in {file_path}")
                    continue
    except Exception as e:
        st.error(f"Error loading file {file_path}: {str(e)}")
        return []
    return completions

def main():
    st.title("LM Completions Viewer")
    
    # Loading available files
    with st.spinner('Loading available files...'):
        jsonl_files = get_jsonl_files()
    
    if not jsonl_files:
        st.error("No JSONL files found in the completions_eval_store directory.")
        st.info("Make sure the completions_eval_store directory exists and contains JSONL files.")
        return
    
    # Create sidebar with file selection
    st.sidebar.title("Select File")
    selected_file_rel = st.sidebar.selectbox(
        "Choose a JSONL file to view:",
        options=[f[0] for f in jsonl_files],
        format_func=lambda x: x.replace('.jsonl', '')
    )
    
    # Get the full path of the selected file
    selected_file = next(f[1] for f in jsonl_files if f[0] == selected_file_rel)
    
    # Load and display the selected file
    st.header(f"Viewing: {selected_file_rel}")
    
    # Add a loading spinner while loading the file
    with st.spinner('Loading completions...'):
        completions = load_jsonl_file(selected_file)
    
    if not completions:
        st.warning("No completions found in this file.")
        return
    
    # Display completion count
    st.write(f"Total completions: {len(completions)}")
    
    # Add a search box
    search_term = st.text_input("Search completions:", "")
    
    # Filter completions based on search term
    filtered_completions = completions
    if search_term:
        filtered_completions = [c for c in completions if search_term.lower() in c.lower()]
        st.write(f"Found {len(filtered_completions)} matching completions")
    
    # Add pagination
    items_per_page = 100
    total_pages = (len(filtered_completions) + items_per_page - 1) // items_per_page
    page = st.number_input('Page', min_value=1, max_value=max(1, total_pages), value=1)
    
    # Calculate the range of completions to display
    start_idx = (page - 1) * items_per_page
    end_idx = min(start_idx + items_per_page, len(filtered_completions))
    
    # Display each completion in an expandable section
    for i, completion in enumerate(filtered_completions[start_idx:end_idx], start=start_idx + 1):
        with st.expander(f"Completion {i}"):
            st.text_area("", completion, height=200, key=f"completion_{i}")
    
    # Add page navigation info
    st.write(f"Showing completions {start_idx + 1} to {end_idx} of {len(filtered_completions)}")

if __name__ == "__main__":
    main()
