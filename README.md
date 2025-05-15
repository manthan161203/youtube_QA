# YouTube Transcript Analyzer

A Streamlit application that allows you to analyze YouTube video transcripts and generate insights.

## Features

- Extract transcripts from YouTube videos
- Generate AI-powered insights about video content
- Support for both YouTube's built-in transcripts and Whisper transcription
- Content generation features:
  - Video summaries
  - Key points
  - Notable quotes
  - Study flashcards
  - Concept maps
  - Practice questions

## Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd youtube-transcript-analyzer
```

2. Create a virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
Create a `.env` file in the root directory with your OpenAI API key:
```
OPENAI_API_KEY=your_api_key_here
```

## Usage

1. Run the Streamlit app:
```bash
streamlit run src/ui/app.py
```

2. Open your browser and navigate to the URL shown in the terminal (usually http://localhost:8501)

3. Enter a YouTube URL in the sidebar and click "Process Video"

4. Once the video is processed, you can:
   - View the video details and transcript
   - Generate various types of content analysis
   - Download the generated content

## Project Structure

```
youtube-transcript-analyzer/
├── src/
│   ├── utils/
│   │   ├── youtube_utils.py
│   │   └── transcription_utils.py
│   ├── models/
│   │   └── content_generation.py
│   ├── services/
│   │   └── transcript_service.py
│   └── ui/
│       └── app.py
├── requirements.txt
└── README.md
```

## Dependencies

- Streamlit: Web application framework
- LangChain: Framework for developing applications powered by language models
- OpenAI: API for GPT models
- FAISS: Vector similarity search
- Sentence Transformers: Text embeddings
- PyTube: YouTube video download
- yt-dlp: YouTube video download (backup)
- YouTube Transcript API: Extract YouTube transcripts
- Whisper: Speech recognition

## License

MIT License 