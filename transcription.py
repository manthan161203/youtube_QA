import os
import streamlit as st
import whisper
import yt_dlp

def download_youtube_audio(video_id, output_directory="downloads"):
    """
    Download YouTube video audio using yt-dlp.
    Returns the path to the downloaded file.
    """
    try:
        # Create output directory if it doesn't exist
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

        youtube_url = f"https://www.youtube.com/watch?v={video_id}"
        
        # yt-dlp options for audio download
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': os.path.join(output_directory, '%(title)s.%(ext)s'),
            'quiet': True,
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'wav',
                'preferredquality': '192',
            }],
            'prefer_ffmpeg': True,
            'keepvideo': False
        }

        # Download audio
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(youtube_url, download=True)
            downloaded_file = ydl.prepare_filename(info).replace(f".{info['ext']}", ".wav")
            
        return downloaded_file
    except Exception as e:
        st.error(f"Download error: {str(e)}")
        return None

def transcribe_audio(file_path, model_size="base"):
    """
    Transcribe audio file using Whisper.
    
    Args:
        file_path: Path to the audio file
        model_size: Whisper model size ('tiny', 'base', 'small', 'medium', 'large')
    
    Returns:
        Transcription text and segments
    """
    try:
        with st.status(f"Loading Whisper {model_size} model...") as status:
            model = whisper.load_model(model_size)
            status.update(label="Model loaded successfully")
            
            status.update(label=f"Transcribing audio with Whisper {model_size}...")
            # Use device auto-detection to use GPU if available
            result = model.transcribe(file_path, fp16=False)  # Use FP32 for CPU compatibility
            status.update(label="Transcription complete!", state="complete")
        
        return result["text"], result.get("segments", [])
    
    except Exception as e:
        st.error(f"Transcription error: {str(e)}")
        return None, []

def download_and_transcribe(video_id, whisper_model_size="base"):
    """Download audio and transcribe using Whisper."""
    output_dir = "downloads"
    
    with st.spinner("Downloading audio..."):
        audio_file = download_youtube_audio(video_id, output_dir)
    
    if not audio_file:
        return "Failed to download audio for transcription.", [], "error"
    
    with st.spinner(f"Transcribing audio using Whisper {whisper_model_size}..."):
        transcription, segments = transcribe_audio(audio_file, whisper_model_size)
    
    if not transcription:
        return "Transcription failed.", [], "error"
    
    # Convert Whisper segments to YouTube-like format
    formatted_segments = []
    if segments:
        for segment in segments:
            formatted_segments.append({
                'text': segment.get('text', ''),
                'start': segment.get('start', 0),
                'duration': segment.get('end', 0) - segment.get('start', 0)
            })
    
    # Clean up downloaded file to save space
    try:
        os.remove(audio_file)
    except:
        pass
    
    return transcription, formatted_segments, "whisper"