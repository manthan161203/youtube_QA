import streamlit as st
import pandas as pd
import time
import os
os.environ["HF_TOKEN"] = st.secrets["HF_TOKEN"]
from youtube_utils import extract_video_id, get_video_details, get_transcript
from langchain_utils import process_with_langchain
from content_generators import (
    generate_summary,
    extract_key_points,
    extract_notable_quotes,
    generate_flashcards,
    generate_concept_map,
    generate_practice_questions
)

st.set_page_config(page_title="YouTube Transcript Analyzer", page_icon="🎬", layout="wide")

# Initialize session state
if "current_video_id" not in st.session_state:
    st.session_state.current_video_id = None
if "transcript_text" not in st.session_state:
    st.session_state.transcript_text = None
if "transcript_segments" not in st.session_state:
    st.session_state.transcript_segments = None
if "transcript_source" not in st.session_state:
    st.session_state.transcript_source = None

# App UI
st.title("YouTube Transcript Analyzer")
st.write("Enter a YouTube URL to analyze its transcript and generate insights about the content.")

# Sidebar for settings
st.sidebar.title("Settings")
api_key = st.sidebar.text_input("OpenAI API Key (required for content generation)", type="password")
if api_key:
    os.environ["OPENAI_API_KEY"] = api_key

embedding_model = st.sidebar.radio(
    "Embedding Model",
    ["huggingface", "openai"],
    index=0
)

llm_model = st.sidebar.selectbox(
    "LLM Model",
    ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"],
    index=0,
    help="Model to use for content generation"
)

whisper_model_size = st.sidebar.selectbox(
    "Whisper Model Size (for fallback)",
    ["tiny", "base", "small", "medium", "large"],
    index=1,
    help="Larger models are more accurate but slower and require more memory"
)

st.sidebar.markdown("---")
st.sidebar.markdown("""
### About
This app uses:
- YouTube Transcript API for fetching subtitles
- Whisper for audio transcription when subtitles aren't available
- LangChain v2 for text processing
- Vector embeddings to analyze transcript content
- OpenAI's models for content generation
""")

# Main layout
col1, col2 = st.columns([2, 1])

with col1:
    # Input for YouTube URL
    youtube_url = st.text_input("YouTube URL", placeholder="https://www.youtube.com/watch?v=...")

if youtube_url:
    video_id = extract_video_id(youtube_url)
    
    if video_id:
        # Check if this is a new video (different from the current one in session)
        new_video = False
        if st.session_state.current_video_id != video_id:
            st.session_state.current_video_id = video_id
            new_video = True
        
        # Show video details
        video_details = get_video_details(video_id)
        
        if "error" not in video_details:
            with col2:
                st.image(video_details["thumbnail"], use_column_width=True)
                st.write(f"**Title:** {video_details['title']}")
                st.write(f"**Channel:** {video_details['author']}")
                minutes, seconds = divmod(video_details['length'], 60)
                st.write(f"**Length:** {minutes} minutes, {seconds} seconds")
        
        # Generate transcript if it's a new video
        if new_video:
            transcript, transcript_segments, transcript_source = get_transcript(video_id, whisper_model_size)
            
            if transcript_source in ["youtube", "whisper"]:
                # Show transcript source
                if transcript_source == "youtube":
                    st.success("✅ Transcript obtained from YouTube subtitles")
                else:
                    st.success("✅ Transcript generated using Whisper speech recognition")
                
                # Store transcript text in session state
                st.session_state.transcript_text = transcript
                
                # Process with LangChain
                with st.spinner("Processing with LangChain..."):
                    docs, vectorstore = process_with_langchain(
                        transcript, 
                        transcript_segments,
                        embed_model=embedding_model
                    )
                    
                    if docs and vectorstore:
                        st.session_state.docs = docs
                        st.session_state.vectorstore = vectorstore
                        st.success("✅ Vector embeddings created successfully")
                    else:
                        st.error("Failed to create vector embeddings")
            else:
                st.error(transcript)
        
        # Store transcript segments in session state if it's a new video
        if new_video:
            st.session_state.transcript_segments = transcript_segments
            st.session_state.transcript_source = transcript_source
            
        # Show transcript tabs
        tab1, tab2, tab3 = st.tabs(["Full Transcript", "Segments", "Content Analysis"])
        
        with tab1:
            st.text_area("Full Transcript", st.session_state.transcript_text, height=300)
            if st.session_state.get("transcript_text"):
                st.download_button(
                    label="Download Full Transcript",
                    data=st.session_state.transcript_text,
                    file_name=f"youtube_transcript_{video_id}.txt",
                )
            else:
                st.warning("Transcript is not available yet. Please extract it first.")
                    
        with tab2:
            if "transcript_segments" in st.session_state and st.session_state.transcript_segments:
                df = pd.DataFrame(st.session_state.transcript_segments)
                # Format time
                df['start_time'] = df['start'].apply(lambda x: time.strftime('%H:%M:%S', time.gmtime(x)))
                
                if "transcript_source" in st.session_state and st.session_state.transcript_source == "youtube":
                    df = df[['start_time', 'text']]
                    df.columns = ['Timestamp', 'Text']
                else:  # whisper
                    df = df[['start_time', 'text']]
                    df.columns = ['Timestamp', 'Text']
                
                st.dataframe(df, use_container_width=True)
                
                # CSV download
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Segments as CSV",
                    data=csv,
                    file_name=f"youtube_transcript_segments_{video_id}.csv",
                    mime="text/csv",
                )
            else:
                st.info("No segments available for this transcript.")
        
        with tab3:
            # Content generation buttons
            if "docs" in st.session_state and os.environ.get("OPENAI_API_KEY"):
                # Feature buttons in a 2x2 grid
                col1, col2 = st.columns(2)
                
                with col1:
                    if st.button("📝 Summarize Video", use_container_width=True):
                        with st.spinner("Generating summary..."):
                            summary = generate_summary(st.session_state.docs, video_details, llm_model)
                            st.session_state.generated_content = {"type": "summary", "content": summary}
                    
                    if st.button("💡 Key Points", use_container_width=True):
                        with st.spinner("Extracting key points..."):
                            key_points = extract_key_points(st.session_state.docs, video_details, llm_model)
                            st.session_state.generated_content = {"type": "key_points", "content": key_points}
                
                with col2:
                    if st.button("🔤 Notable Quotes", use_container_width=True):
                        with st.spinner("Finding notable quotes..."):
                            quotes = extract_notable_quotes(st.session_state.docs, video_details, llm_model)
                            st.session_state.generated_content = {"type": "quotes", "content": quotes}
                    
                    if st.button("🧠 Study Flashcards", use_container_width=True):
                        with st.spinner("Creating flashcards..."):
                            flashcards = generate_flashcards(st.session_state.docs, video_details, llm_model)
                            st.session_state.generated_content = {"type": "flashcards", "content": flashcards}
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("🗺️ Concept Map", use_container_width=True):
                        with st.spinner("Creating concept map..."):
                            concept_map = generate_concept_map(st.session_state.docs, video_details, llm_model)
                            st.session_state.generated_content = {"type": "concept_map", "content": concept_map}
                
                with col2:
                    if st.button("❓ Practice Questions", use_container_width=True):
                        with st.spinner("Creating practice questions..."):
                            questions = generate_practice_questions(st.session_state.docs, video_details, llm_model)
                            st.session_state.generated_content = {"type": "questions", "content": questions}
                
                # Display generated content if any
                if "generated_content" in st.session_state:
                    st.markdown("---")
                    content_type = st.session_state.generated_content["type"]
                    content = st.session_state.generated_content["content"]
                    
                    type_titles = {
                        "summary": "📝 Video Summary",
                        "key_points": "💡 Key Points",
                        "quotes": "🔤 Notable Quotes",
                        "flashcards": "🧠 Study Flashcards",
                        "concept_map": "🗺️ Concept Map",
                        "questions": "❓ Practice Questions"
                    }
                    
                    st.subheader(type_titles.get(content_type, "Generated Content"))
                    st.markdown(content)
                    
                    # Download button for the generated content
                    st.download_button(
                        label=f"Download {type_titles.get(content_type, 'Content')}",
                        data=content,
                        file_name=f"{video_id}_{content_type}.md",
                        mime="text/markdown"
                    )
            else:
                if not os.environ.get("OPENAI_API_KEY"):
                    st.warning("Please add your OpenAI API key in the sidebar to enable content generation features.")
                else:
                    st.warning("Please process a video transcript first.")
    else:
        st.error("Could not extract a valid YouTube video ID from the provided URL.")
else:
    st.info("Please enter a YouTube URL to begin analysis.")

# Footer
st.markdown("---")
st.markdown("""
### How to use:
1. Enter your OpenAI API key in the sidebar (required for content generation)
2. Select your preferred embedding model and LLM model
3. Choose Whisper model size for fallback transcription
4. Paste any YouTube video URL in the input box
5. Explore the transcript in the "Full Transcript" and "Segments" tabs
6. Use the feature buttons in the "Content Analysis" tab to generate:
   - Video summaries
   - Key points
   - Notable quotes
   - Study flashcards
   - Concept maps
   - Practice questions
7. Download any generated content for later reference
""")