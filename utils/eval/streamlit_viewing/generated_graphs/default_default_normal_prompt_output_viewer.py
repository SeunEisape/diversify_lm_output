import streamlit as st
import json
import os
from pathlib import Path
import glob
import time
from PIL import Image
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
TARGET_FILE = ""completions_eval_store/default/default_normal_prompt_output.jsonl""
IS_RANDOM = "False"

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

def get_graphs_directory(file_path):
    """Get the path to the graphs directory for a given JSONL file."""
    file_dir = os.path.dirname(file_path)
    graphs_dir = os.path.join(file_dir, "graphs")
    return graphs_dir

def load_graphs(graphs_dir):
    """Load all graph images from the graphs directory."""
    graphs = {}
    if not os.path.exists(graphs_dir):
        return graphs
    
    for file in os.listdir(graphs_dir):
        if file.endswith(('.png', '.jpg', '.jpeg')):
            try:
                image_path = os.path.join(graphs_dir, file)
                graphs[file] = Image.open(image_path)
            except Exception as e:
                st.warning(f"Could not load graph {file}: {str(e)}")
    
    return graphs

def get_file_from_query_params():
    """Get the file path from query parameters if it exists."""
    query_params = st.experimental_get_query_params()
    if 'file' in query_params:
        return urllib.parse.unquote(query_params['file'][0])
    return None

def main():
    st.title("LM Completions Viewer")
    
    # Loading available files
    with st.spinner('Loading available files...'):
        jsonl_files = get_jsonl_files()
    
    if not jsonl_files:
        st.error("No JSONL files found in the completions_eval_store directory.")
        st.info("Make sure the completions_eval_store directory exists and contains JSONL files.")
        return
    
    # Get file from query parameters if it exists
    query_file = get_file_from_query_params()
    
    # Create sidebar with file selection
    st.sidebar.title("Select File")
    
    # Create a dictionary mapping relative paths to full paths and is_random flag
    file_dict = {f[0]: (f[1], f[2]) for f in jsonl_files}
    
    # If we have a query parameter file, try to use it
    if query_file and query_file in file_dict:
        selected_file_rel = query_file
    else:
        selected_file_rel = st.sidebar.selectbox(
            "Choose a JSONL file to view:",
            options=[f[0] for f in jsonl_files],
            format_func=lambda x: x.replace('.jsonl', '')
        )
    
    # Get the full path and type of the selected file
    selected_file, is_random = file_dict[selected_file_rel]
    
    # Update URL with selected file
    st.experimental_set_query_params(file=urllib.parse.quote(selected_file_rel))
    
    # Add a copy link button
    current_url = st.experimental_get_query_params()
    copy_url = f"{st.experimental_get_query_params()['file'][0]}"
    st.sidebar.markdown(f"**Share this link:**")
    st.sidebar.code(copy_url)
    
    # Create tabs for different views
    tab1, tab2 = st.tabs(["Completions", "Graphs"])
    
    with tab1:
        # Load and display the file
        st.header(f"Viewing: {selected_file_rel}")
        
        # Add a loading spinner while loading the file
        with st.spinner('Loading completions...'):
            completions = load_jsonl_file(selected_file, is_random)
        
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
            if is_random and 'random_doc' in completion:
                st.write("**Random Document:**")
                st.text(completion['random_doc'])
                st.write("**Completion:**")
            st.text_area("", completion['text'], height=200, key=f"completion_{i}")
            st.markdown("---")  # Add a horizontal line between completions
        
        # Add page navigation info
        st.write(f"Showing completions {start_idx + 1} to {end_idx} of {len(filtered_completions)}")
    
    with tab2:
        # Load and display graphs
        graphs_dir = get_graphs_directory(selected_file)
        st.header("Analysis Graphs")
        
        if not os.path.exists(graphs_dir):
            st.warning(f"No graphs directory found at {graphs_dir}")
            return
        
        graphs = load_graphs(graphs_dir)
        if not graphs:
            st.warning("No graphs found in the graphs directory.")
            return
        
        # Display graphs in a grid
        cols = st.columns(2)
        for i, (graph_name, graph_image) in enumerate(graphs.items()):
            with cols[i % 2]:
                st.subheader(graph_name.replace('.png', '').replace('_', ' ').title())
                st.image(graph_image, use_column_width=True)

if __name__ == "__main__":
    main()
