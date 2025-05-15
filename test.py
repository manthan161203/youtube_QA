import os
import sys
import yt_dlp
import subprocess
import whisper
from pathlib import Path
from pydub import AudioSegment

def download_youtube_audio(youtube_url, output_directory="downloads"):
    """
    Download YouTube video audio using yt-dlp.
    Returns the path to the downloaded file.
    """
    try:
        # Create output directory if it doesn't exist
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

        # yt-dlp options for audio download
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': os.path.join(output_directory, '%(title)s.%(ext)s'),
            'quiet': False,
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
        print(f"❌ Download error: {str(e)}")
        return None

def transcribe_audio(file_path, model_size="base"):
    """
    Transcribe audio file using Whisper.
    
    Args:
        file_path: Path to the audio file
        model_size: Whisper model size ('tiny', 'base', 'small', 'medium', 'large')
    
    Returns:
        Transcription text
    """
    try:
        print(f"🔄 Loading Whisper {model_size} model...")
        model = whisper.load_model(model_size)
        
        print(f"📂 Processing file: {file_path}")
        
        # Run Whisper transcription
        result = model.transcribe(file_path, fp16=False)  # Use FP32 for CPU compatibility
        
        return result["text"]
    
    except Exception as e:
        print(f"❌ Transcription error: {str(e)}")
        return None

def save_transcription(text, output_file="transcription.txt"):
    """Save transcription to a text file."""
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"✅ Transcription saved to: {output_file}")
    except Exception as e:
        print(f"❌ Error saving transcription: {str(e)}")

def main():
    # Clear terminal
    os.system('cls' if os.name == 'nt' else 'clear')
    
    print("🎬 YouTube Audio Transcriber 🎬")
    print("===============================")
    
    # Get YouTube URL
    youtube_url = input("Enter YouTube URL: ").strip()
    if not youtube_url:
        print("❌ No URL provided. Exiting.")
        return
    
    # Get model size
    print("\nWhisper Model Options:")
    print("1. tiny (fastest, least accurate)")
    print("2. base (fast, decent accuracy)")
    print("3. small (balanced)")
    print("4. medium (slower, more accurate)")
    print("5. large (slowest, most accurate)")
    
    model_choice = input("\nSelect model (1-5) [default: 2]: ").strip() or "2"
    model_sizes = {
        "1": "tiny",
        "2": "base", 
        "3": "small",
        "4": "medium",
        "5": "large"
    }
    
    model_size = model_sizes.get(model_choice, "base")
    
    # Set output directory
    output_dir = "downloads"
    
    print("\n📥 Downloading audio...")
    audio_file = download_youtube_audio(youtube_url, output_dir)
    
    if not audio_file:
        print("❌ Failed to download audio. Exiting.")
        return
    
    print(f"✅ Audio downloaded: {audio_file}")
    
    print("\n🎤 Transcribing audio...")
    transcription = transcribe_audio(audio_file, model_size)
    
    if not transcription:
        print("❌ Transcription failed.")
        return
    
    print("\n📝 Transcription Result:")
    print("======================")
    print(transcription)
    
    # Save transcription
    video_name = os.path.basename(audio_file).rsplit(".", 1)[0]
    output_file = f"{output_dir}/{video_name}_transcription.txt"
    save_transcription(transcription, output_file)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n❌ Process interrupted by user.")
    except Exception as e:
        print(f"\n❌ An error occurred: {str(e)}")