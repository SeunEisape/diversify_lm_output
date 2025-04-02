import streamlit as st
import json
import os
from pathlib import Path
import glob
import time
import urllib.parse

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

# This will be replaced with the actual file path when generating individual pages
TARGET_FILE = "completions_eval_store/NLP_research/NLP_research_random_prompt_output.jsonl"
IS_RANDOM = "True"

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
                    # Determine if it's a random or normal prompt file
                    is_random = "random" in file.lower()
                    jsonl_files.append((rel_path, full_path, is_random))
    except Exception as e:
        st.error(f"Error finding JSONL files: {str(e)}")
        return []
    return sorted(jsonl_files)

@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_jsonl_file(file_path, is_random):
    """Load and parse a JSONL file."""
    completions = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    if 'completion_only' in data:
                        completion_data = {
                            'text': data['completion_only']
                        }
                        if is_random and 'random_doc' in data:
                            completion_data['random_doc'] = data['random_doc']
                        completions.append(completion_data)
                except json.JSONDecodeError:
                    st.warning(f"Could not parse line in {file_path}")
                    continue
    except Exception as e:
        st.error(f"Error loading file {file_path}: {str(e)}")
        return []
    return completions

def get_file_from_query_params():
    """Get the file path from query parameters if it exists."""
    if 'file' in st.query_params:
        return urllib.parse.unquote(st.query_params['file'])
    return None

def main():
    # Get relative path for display
    rel_path = os.path.relpath(TARGET_FILE, "completions_eval_store")
    st.title(f"LM Completions Viewer: {rel_path}")
    
    # Load and display the file
    st.header(f"Viewing: {rel_path}")
    
    # Add a loading spinner while loading the file
    with st.spinner('Loading completions...'):
        completions = load_jsonl_file(TARGET_FILE, IS_RANDOM)
    
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
        filtered_completions = [c for c in completions if search_term.lower() in c['text'].lower()]
        st.write(f"Found {len(filtered_completions)} matching completions")
    
    # Add pagination
    items_per_page = 100
    total_pages = (len(filtered_completions) + items_per_page - 1) // items_per_page
    page = st.number_input('Page', min_value=1, max_value=max(1, total_pages), value=1)
    
    # Calculate the range of completions to display
    start_idx = (page - 1) * items_per_page
    end_idx = min(start_idx + items_per_page, len(filtered_completions))
    
    # Display each completion directly
    for i, completion in enumerate(filtered_completions[start_idx:end_idx], start=start_idx + 1):
        st.subheader(f"Completion {i}")
        if IS_RANDOM and 'random_doc' in completion:
            st.write("**Random Document:**")
            st.text(completion['random_doc'])
            st.write("**Completion:**")
        st.text_area("", completion['text'], height=200, key=f"completion_{i}")
        st.markdown("---")  # Add a horizontal line between completions
    
    # Add page navigation info
    st.write(f"Showing completions {start_idx + 1} to {end_idx} of {len(filtered_completions)}")

if __name__ == "__main__":
    main()
