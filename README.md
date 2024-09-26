# DhiMeet - Meeting Management Tool

## Overview
The Meeting Management Tool is designed to help users efficiently manage meetings by organizing pre-meeting documents, handling discussion points, tracking meeting progress, and generating post-meeting summaries. The tool uses both traditional development techniques and generative AI models to automate tasks like agenda creation and meeting transcription.

## Features
1. **Pre-Meeting Document Handling**: Upload and manage documents for meetings.
2. **Discussion Point Management**: Add, track, and remove discussion points.
3. **Agenda Generation**: Automatically generate meeting agendas based on discussion points.
4. **Meeting Transcription**: Extract and transcribe audio from meeting videos using Whisper.
5. **Post-Meeting Summary**: Generate detailed summaries of the meeting, including key discussions, decisions made, and action items.
6. **Discussion Points Coverage**: Identify which discussion points were covered in the meeting based on the transcription.

## Tech Stack
- **Streamlit**: Frontend framework to build the web interface.
- **Whisper**: Model used for meeting transcription.
- **BART (facebook/bart-large-cnn)**: Used for summarization of discussion points and transcription.
- **Sentence Transformers**: For semantic similarity checks.
- **FAISS**: Indexing and retrieval of relevant discussion points (future implementation).
- **MoviePy**: To extract audio from video files.

## Dependencies

The following Python libraries are required to run the project:

- `streamlit`
- `whisper`
- `moviepy`
- `transformers`
- `sentence-transformers`
- `faiss`
- `scikit-learn`
- `numpy`

You can install these dependencies via pip:

```bash
pip install streamlit whisper moviepy transformers sentence-transformers faiss-cpu scikit-learn numpy
```

## Setup Instructions

1. **Clone the Repository**

   Clone the project repository to your local machine:

   ```bash
   git clone https://github.com/riyachudasama23/DhiMeet.git
   ```

2. **Install Dependencies**

   Ensure all the required libraries are installed:

   ```bash
   pip install -r requirements.txt
   ```

   Alternatively, you can install dependencies manually using the `pip` command provided earlier.

3. **Run the Application**

   Start the Streamlit application:

   ```bash
   streamlit run main.py
   ```

4. **Access the Application**

   Once the application is running, open your browser and navigate to:

   ```
   http://localhost:8501
   ```

## Usage Instructions

### 1. Upload Pre-Meeting Documents
   - Use the **Upload Pre-Meeting Documents** section to upload any PDF or DOCX files related to the meeting.
   - The uploaded files will be saved in the `uploads` directory.

### 2. Add Discussion Points
   - In the **Add Discussion Points** section, you can enter discussion points for the meeting.
   - Use the "Submit" button to save them.
   - You can also remove any existing discussion points by clicking "Remove."

### 3. Generate Meeting Agenda
   - After adding discussion points, use the **Generate Agenda** button to create a structured meeting agenda based on the provided points.

### 4. Video to Text Transcription
   - Upload your meeting video in MP4, MKV, or AVI format in the **Video to Text Transcription** section.
   - The tool will extract audio from the video and transcribe it using Whisper.
   - A detailed summary will be generated from the transcription.

### 5. Discussion Points Coverage
   - The tool checks which discussion points were covered in the meeting based on the transcription and flags those that were not discussed.

## Assumptions & Design Choices

1. **Assumption**: The video uploaded contains a clear audio track that can be transcribed accurately by Whisper.
2. **Design Choice**: Whisper is used for transcription because of its state-of-the-art accuracy in transcribing meeting recordings.
3. **Design Choice**: BART is utilized for summarization to ensure that the generated meeting summaries are concise and well-structured.
4. **Assumption**: Discussion points are directly or semantically mentioned in the meeting transcription, allowing for effective tracking using cosine similarity.

## Future Enhancements
- **Retrieval-Augmented Generation (RAG)**: Implement the FAISS index to retrieve and link relevant discussion points with meeting topics.
- **Action Items Tracking**: Automatically track assigned action items and participants based on the meeting summary.

## Contact
For any questions or support, please feel free to reach out to the project maintainers.

---

