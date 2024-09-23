import streamlit as st
import os

# Create a directory to store uploaded files
if not os.path.exists("uploads"):
    os.makedirs("uploads")

st.title("Upload Pre-Meeting Documents")

uploaded_file = st.file_uploader("Choose a file", type=["pdf", "docx"])

if uploaded_file:
    # Save the uploaded file to the 'uploads' directory
    with open(os.path.join("uploads", uploaded_file.name), "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    st.write(f"File '{uploaded_file.name}' uploaded successfully!")
    st.write("File saved to 'uploads' directory.")
