import streamlit as st
import os

discussion_file = "discussion_points.txt"

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

# Function to load discussion points from the file
def load_discussion_points():
    if os.path.exists(discussion_file):
        with open(discussion_file, "r") as file:
            return [line.strip() for line in file.readlines()]
    return []

# Function to save discussion points to the file
def save_discussion_points(points):
    with open(discussion_file, "w") as file:
        for point in points:
            file.write(point + "\n")

# Load discussion points
discussion_points = load_discussion_points()

# Discussion points section
st.title("Add Discussion Points")
discussion_point = st.text_input("Enter a discussion point")

if st.button("Submit"):
    if discussion_point:
        discussion_points.append(discussion_point)
        save_discussion_points(discussion_points)  # Save to file
        st.session_state["discussion_point"] = ""  # clear input and refresh the list

# Display discussion points with the ability to remove them
st.write("Current Discussion Points:")
if discussion_points:
    for i, point in enumerate(discussion_points):
        col1, col2 = st.columns([4, 1])
        col1.write(f"{i+1}. {point}")
        if col2.button("Remove", key=f"remove_{i}"):
            # Remove the selected discussion point
            discussion_points.pop(i)
            save_discussion_points(discussion_points)  # Save the updated list to the file
            # st.experimental_rerun()  # Rerun the app to refresh the list

# If there are no discussion points
if not discussion_points:
    st.write("No discussion points added yet.")