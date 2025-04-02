import os
import json
from pathlib import Path

def generate_page_template(file_path, is_random):
    """Generate a Streamlit page template for a JSONL file."""
    rel_path = os.path.relpath(file_path, "completions_eval_store")
    page_name = rel_path.replace('/', '_').replace('.jsonl', '')
    
    template = f'''import streamlit as st
import json
from pathlib import Path

st.set_page_config(
    page_title="LM Completions Viewer - {page_name}",
    layout="wide"
)

@st.cache_data(ttl=3600)
def load_completions():
    """Load and parse the JSONL file."""
    completions = []
    try:
        with open("{file_path}", 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    if 'completion_only' in data:
                        completion_data = {{
                            'text': data['completion_only']
                        }}
                        {"if 'random_doc' in data:\n                            completion_data['random_doc'] = data['random_doc']" if is_random else ""}
                        completions.append(completion_data)
                except json.JSONDecodeError:
                    st.warning(f"Could not parse line in {{file_path}}")
                    continue
    except Exception as e:
        st.error(f"Error loading file {{file_path}}: {{str(e)}}")
        return []
    return completions

def main():
    st.title("LM Completions Viewer")
    st.header("Viewing: {rel_path}")
    
    # Add a loading spinner while loading the file
    with st.spinner('Loading completions...'):
        completions = load_completions()
    
    if not completions:
        st.warning("No completions found in this file.")
        return
    
    # Display completion count
    st.write(f"Total completions: {{len(completions)}}")
    
    # Add a search box
    search_term = st.text_input("Search completions:", "")
    
    # Filter completions based on search term
    filtered_completions = completions
    if search_term:
        filtered_completions = [c for c in completions if search_term.lower() in c['text'].lower()]
        st.write(f"Found {{len(filtered_completions)}} matching completions")
    
    # Add pagination
    items_per_page = 100
    total_pages = (len(filtered_completions) + items_per_page - 1) // items_per_page
    page = st.number_input('Page', min_value=1, max_value=max(1, total_pages), value=1)
    
    # Calculate the range of completions to display
    start_idx = (page - 1) * items_per_page
    end_idx = min(start_idx + items_per_page, len(filtered_completions))
    
    # Display each completion directly
    for i, completion in enumerate(filtered_completions[start_idx:end_idx], start=start_idx + 1):
        st.subheader(f"Completion {{i}}")
        {"if 'random_doc' in completion:\n            st.write('**Random Document:**')\n            st.text(completion['random_doc'])\n            st.write('**Completion:**')" if is_random else ""}
        st.text_area("", completion['text'], height=200, key=f"completion_{{i}}")
        st.markdown("---")  # Add a horizontal line between completions
    
    # Add page navigation info
    st.write(f"Showing completions {{start_idx + 1}} to {{end_idx}} of {{len(filtered_completions)}}")

if __name__ == "__main__":
    main()
'''
    
    return template, page_name

def main():
    # Create pages directory if it doesn't exist
    pages_dir = Path("utils/eval/streamlit_viewing/pages")
    pages_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all JSONL files
    base_dir = "completions_eval_store"
    for root, _, files in os.walk(base_dir):
        for file in files:
            if file.endswith('.jsonl'):
                full_path = os.path.join(root, file)
                is_random = "random" in file.lower()
                
                # Generate page template
                template, page_name = generate_page_template(full_path, is_random)
                
                # Write the page file
                page_file = pages_dir / f"{page_name}.py"
                with open(page_file, 'w') as f:
                    f.write(template)
                
                print(f"Generated page for {full_path}")

if __name__ == "__main__":
    main() 