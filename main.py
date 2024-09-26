import streamlit as st
import os 

import whisper
from moviepy.editor import VideoFileClip
import tempfile

from transformers import BartForConditionalGeneration, BartTokenizer
from sentence_transformers import SentenceTransformer

from sklearn.metrics.pairwise import cosine_similarity
import faiss
import numpy as np

discussion_file = "discussion_points.txt"

# Load the llm models
model = whisper.load_model("base")

summarizer_model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
summarizer_tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")

sentence_model = SentenceTransformer('all-MiniLM-L6-v2')

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

# If there are no discussion points
if not discussion_points:
    st.write("No discussion points added yet.")

# Initialize FAISS index for RAG functionality 
dimension = 384  # Dimension of sentence embeddings from MiniLM model.
index_flat = faiss.IndexFlatL2(dimension)  # Create a flat (non-hierarchical) index.

# Function to create an organized agenda from discussion points using LLM for grammar correction 
def create_agenda(discussion_points):
   agenda_items = []
   for point in discussion_points:
       prompt = f"Create a well-structured agenda item based on this discussion point: '{point}'"
       inputs = summarizer_tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True)
       summary_ids = summarizer_model.generate(inputs["input_ids"], max_length=150, num_beams=4, early_stopping=True)
       agenda_item = summarizer_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
       agenda_items.append(agenda_item)
   return agenda_items

# Generate and display agenda based on discussion points when requested by user 
st.title("Generate Meeting Agenda")

if st.button("Generate Agenda"):
   if discussion_points: 
       agenda_items = create_agenda(discussion_points)
       st.subheader("Meeting Agenda:")
       for item in agenda_items:
           st.markdown(f"- {item}")  # Use Markdown for bullet points 
   else: 
       st.write("No discussion points available to create an agenda.")

# Function to extract audio from video
def extract_audio_from_video(video_file_path):
    video = VideoFileClip(video_file_path)
    audio_file_path = "temp_audio.wav"
    video.audio.write_audiofile(audio_file_path)
    return audio_file_path

# Function to transcribe audio using Whisper
def transcribe_audio(audio_file_path):
    result = model.transcribe(audio_file_path)
    return result["text"]

# Function to generate detailed summary from transcription
def generate_detailed_summary(transcription):
    prompt = f"""
    Summarize the following meeting transcription. Include:
    - What was discussed
    - Key decisions made during the meeting
    - Assigned action items and responsible participants

    Transcription: {transcription}
    
    Detailed Summary:
    """
    
    inputs = summarizer_tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = summarizer_model.generate(inputs["input_ids"], max_length=300, num_beams=4, early_stopping=True)
    
    return summarizer_tokenizer.decode(summary_ids[0], skip_special_tokens=True)


# Streamlit UI
st.title("Video to Text Transcription")
uploaded_file = st.file_uploader("Upload Video", type=["mp4", "mkv", "avi"])

st.write("Transcribing Meeting...")

# Function to check if discussion points are covered in the transcription
def check_discussion_points(discussion_points, transcription):
    covered_points = []
    uncovered_points = []
    
    for point in discussion_points:
        # Check if the discussion point is present in the transcription
        if point.lower() in transcription.lower():
            covered_points.append(point)
        else:
            # Calculate semantic similarity
            point_embedding = sentence_model.encode([point])
            transcription_embedding = sentence_model.encode([transcription])
            similarity = cosine_similarity(point_embedding, transcription_embedding)[0][0]
            
            if similarity > 0.7:  # Adjust the threshold as needed
                covered_points.append(point)
            else:
                uncovered_points.append(point)
    
    return covered_points, uncovered_points

if uploaded_file:
    # Save the uploaded video temporarily in a temp directory
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video_file:
        temp_video_path = temp_video_file.name
        temp_video_file.write(uploaded_file.getbuffer())

    # Extract audio from the video
    audio_file_path = extract_audio_from_video(temp_video_path)

    # Transcribe the extracted audio
    transcription = transcribe_audio(audio_file_path)
    
    # Display transcription
    st.subheader("Transcription:")
    st.write(transcription)

    # Generate detailed summary of transcription
    detailed_summary = generate_detailed_summary(transcription)

    # Display detailed summary
    st.subheader("Detailed Summary:")
    st.write(detailed_summary)

    # Check discussion points coverage
    covered_points, uncovered_points = check_discussion_points(discussion_points, transcription)
    
    # Display results
    st.subheader("Covered Discussion Points:")
    if covered_points:
        for point in covered_points:
            st.markdown(f"- {point}")  # Use Markdown for bullet points
    else:
        st.write("No covered points.")
    
    st.subheader("Uncovered Discussion Points:")
    if uncovered_points:
        for point in uncovered_points:
            st.markdown(f"- {point}")  # Use Markdown for bullet points
    else:
        st.write("All points covered.")  

# Add vectors of discussion points to FAISS index when new points are added.
for point in discussion_points:
   embedding = sentence_model.encode([point])
   index_flat.add(np.array(embedding).astype('float32'))  # Ensure correct type.

# Function to retrieve relevant discussion points using RAG approach.
def retrieve_relevant_discussion(query):
   query_embedding = sentence_model.encode([query])
   D, I = index_flat.search(np.array(query_embedding).astype('float32'), k=5)  # Retrieve top 5 closest vectors.
   return [discussion_points[i] for i in I[0]]

# Example usage of retrieval function (you can call this based on user input).
if st.button("Retrieve Relevant Discussion Points"):
   user_query = st.text_input("Enter your query:")
   if user_query:
       relevant_discussions = retrieve_relevant_discussion(user_query)
       st.subheader("Relevant Discussion Points:")
       for item in relevant_discussions:
           st.markdown(f"- {item}") 