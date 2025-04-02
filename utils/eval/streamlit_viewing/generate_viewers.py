import os
import shutil
from pathlib import Path

def generate_viewers():
    """Generate individual Streamlit viewers for each JSONL file in completions_eval_store."""
    # Base directory containing JSONL files
    base_dir = "completions_eval_store"
    
    # Create a directory for the generated viewers if it doesn't exist
    viewers_dir = Path("utils/eval/streamlit_viewing/generated")
    viewers_dir.mkdir(exist_ok=True)
    
    # Read the template file
    template_path = Path("utils/eval/streamlit_viewing/visualize_completions.py")
    with open(template_path, 'r') as f:
        template = f.read()
    
    # Find all JSONL files
    for root, _, files in os.walk(base_dir):
        for file in files:
            if file.endswith('.jsonl'):
                # Get full path and relative path
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, base_dir)
                
                # Determine if it's a random prompt file
                is_random = "random" in file.lower()
                
                # Create a safe filename for the viewer
                viewer_name = rel_path.replace('/', '_').replace('.jsonl', '_viewer.py')
                viewer_path = viewers_dir / viewer_name
                
                # Generate the viewer file
                viewer_content = template.replace(
                    "REPLACE_WITH_FILE_PATH",
                    f'"{full_path}"'
                ).replace(
                    "REPLACE_WITH_IS_RANDOM",
                    str(is_random)
                )
                
                # Write the viewer file
                with open(viewer_path, 'w') as f:
                    f.write(viewer_content)
                
                print(f"Generated viewer for {rel_path} at {viewer_path}")

if __name__ == "__main__":
    generate_viewers() 