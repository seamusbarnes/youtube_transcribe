import os
import yt_dlp
from pydub import AudioSegment
import whisper
from deep_translator import GoogleTranslator

# Step 1: Download YouTube Video
def download_audio(youtube_url, output_path="downloaded_audio.mp4"):
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': output_path,
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([youtube_url])

# Step 2: Transcribe Audio (Dutch)
def transcribe_audio(audio_path, model_size="medium"):
    model = whisper.load_model(model_size)
    result = model.transcribe(audio_path, language="nl")
    return result["text"]

# Step 3: Translate Dutch to English
def translate_text(dutch_text):
    translator = GoogleTranslator(source="nl", target="en")
    return translator.translate(dutch_text)

# Full Workflow
def youtube_to_translated_text(youtube_url):
    audio_file = "audio.mp3"

    print("Downloading YouTube audio...")
    download_audio(youtube_url, audio_file)

    print("Transcribing audio (Dutch)...")
    dutch_text = transcribe_audio(audio_file)

    print("Translating to English...")
    english_text = translate_text(dutch_text)

    print("\nDutch Transcription:\n", dutch_text)
    print("\nEnglish Translation:\n", english_text)

    return dutch_text, english_text

# Example usage
youtube_url = "https://www.youtube.com/watch?v=NgdXuqrRaDU"
dutch_text, english_text = youtube_to_translated_text(youtube_url)
