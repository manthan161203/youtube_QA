import re
import pytube
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import NoTranscriptFound

from transcription import download_and_transcribe

def extract_video_id(youtube_url):
    """Extract the video ID from a YouTube URL."""
    patterns = [
        r"(?:v=|\/)([0-9A-Za-z_-]{11}).*",  # Standard YouTube URLs
        r"(?:embed\/)([0-9A-Za-z_-]{11})",  # Embedded URLs
        r"(?:shorts\/)([0-9A-Za-z_-]{11})",  # Shorts URLs
        r"^([0-9A-Za-z_-]{11})$"            # Direct video ID
    ]
    
    for pattern in patterns:
        match = re.search(pattern, youtube_url)
        if match:
            return match.group(1)
    return None

def get_video_details(video_id):
    """Get video title and other details."""
    try:
        yt = pytube.YouTube(f"https://www.youtube.com/watch?v={video_id}")
        return {
            "title": yt.title,
            "author": yt.author,
            "length": yt.length,
            "thumbnail": yt.thumbnail_url,
            "description": yt.description
        }
    except Exception as e:
        return {"error": f"Error retrieving video details: {str(e)}"}

def get_transcript(video_id, whisper_model_size="base"):
    """Get transcript using YouTube Transcript API or fall back to Whisper."""
    try:
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
        transcript_text = " ".join([segment['text'] for segment in transcript_list])
        return transcript_text.strip(), transcript_list, "youtube"
    except (NoTranscriptFound, Exception) as e:
        # Fall back to Whisper transcription for any error
        return download_and_transcribe(video_id, whisper_model_size)