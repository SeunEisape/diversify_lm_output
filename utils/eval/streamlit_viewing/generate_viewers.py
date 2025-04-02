import os
import shutil
from pathlib import Path
import argparse

def generate_viewers(base_dir, output_dir=None):
    """
    Generate individual Streamlit viewers for each JSONL file in the specified directory.
    
    Args:
        base_dir (str): Base directory containing JSONL files (can be cloud path)
        output_dir (str, optional): Directory to save generated viewers. If None, uses default location.
    """
    # Create a directory for the generated viewers if it doesn't exist
    if output_dir is None:
        viewers_dir = Path("utils/eval/streamlit_viewing/generated")
    else:
        viewers_dir = Path(output_dir)
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
                    "REPLACE_WITH_BASE_DIR",
                    f'"{base_dir}"'
                ).replace(
                    "REPLACE_WITH_IS_RANDOM",
                    str(is_random)
                )
                
                # Write the viewer file
                with open(viewer_path, 'w') as f:
                    f.write(viewer_content)
                
                print(f"Generated viewer for {rel_path} at {viewer_path}")

def main():
    parser = argparse.ArgumentParser(description='Generate Streamlit viewers for JSONL files')
    parser.add_argument('--base_dir', type=str, required=True,
                      help='Base directory containing JSONL files (can be cloud path)')
    parser.add_argument('--output_dir', type=str,
                      help='Directory to save generated viewers (optional)')
    args = parser.parse_args()
    
    generate_viewers(args.base_dir, args.output_dir)

if __name__ == "__main__":
    main() 