import streamlit as st

st.set_page_config(
    page_title="LM Completions Viewer - Home",
    layout="wide"
)

st.title("LM Completions Viewer")
st.write("Welcome to the LM Completions Viewer! Select a file from the sidebar to view its completions.")

# Add some helpful information
st.markdown("""
### About This Viewer
This application allows you to:
- View completions from different prompt types
- Search through completions
- Navigate through pages of completions
- View random documents for random prompt completions

### How to Use
1. Select a file from the sidebar
2. Use the search box to filter completions
3. Navigate through pages using the page selector
4. View completions and their associated random documents (if available)
""") 