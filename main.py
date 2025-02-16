import os
import subprocess
from datetime import datetime
import yt_dlp
import whisper
import re
import pandas as pd
import argparse
from deep_translator import GoogleTranslator
import warnings
warnings.filterwarnings('ignore')


# === CONFIGURATION ===
FILENAME_BASE = "audio"  # Base filename for downloaded audio
SOURCE_LANG = "nl"  # Source language (Dutch)
TARGET_LANG = "en"  # Target language (English)
MODEL_SIZE = "small"  # Whisper model size ("tiny", "small", "medium", "large")


# === HELPER FUNCTIONS ===
def check_ffmpeg():
    """Check if FFmpeg is installed."""
    command = ['ffmpeg', '-version']
    result = subprocess.run(command, capture_output=True, text=True)
    print("FFmpeg version:", " ".join(result.stdout.split()[:3]))


def download_audio(url, filename_base):
    """Download audio from a YouTube video using yt-dlp and convert to MP3."""
    ydl_options = {
        'format': 'bestaudio/best',
        'outtmpl': filename_base,
        'postprocessors': [{'key': 'FFmpegExtractAudio', 'preferredcodec': 'mp3', 'preferredquality': '128'}],
        'quiet': True,
        'no_warnings': True,
    }

    print(f"Downloading audio from: {url}")
    # print(f'Video title: {get_video_title(url)}')
    with yt_dlp.YoutubeDL(ydl_options) as ydl:
        info = ydl.extract_info(url, download=False)  # Don't download
        video_title = info.get('title') or 'Unknown Title'
        print(f'Video title: {video_title}')
        ydl.download([url])
    print("Download complete!")


def get_file_size(filename):
    """Return file size in MB."""
    size_mb = os.path.getsize(filename) / (1024 * 1024)
    print(f"Audio file size: {size_mb:.2f} MB")
    return size_mb


def transcribe_audio(filename_base, language, model_size):
    """Transcribe an audio file using Whisper."""
    model = whisper.load_model(model_size)
    print(f"Loaded Whisper model: {model_size}")

    audio_filename = f"{filename_base}.mp3"
    t0 = datetime.now()
    print('--------------------')
    print('Beginning transcription...')
    result = model.transcribe(audio_filename, language=language)
    print(f'Transcription complete, time taken = {(datetime.now()-t0).total_seconds():.2f}s')
    print('--------------------')
    return result["text"]


def save_text(text, filename):
    """Save text to a file."""
    with open(filename, 'w', encoding='utf-8') as file:
        file.write(text)


def split_text(text, max_length=5000):
    """Split text into chunks of max_length while preserving sentence structure."""
    if len(text) <= max_length:
        return [text]

    chunks = []
    sentences = re.split(r'(?<=[.!?])\s+', text)  # Split at sentence boundaries
    current_chunk = ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= max_length:
            current_chunk += sentence + " "
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + " "
    
    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    print(f"Text split into {len(chunks)} chunks for translation.")
    return chunks


def translate_chunks(chunks, source_lang, target_lang):
    """Translate a list of text chunks using Google Translator."""
    t0 = datetime.now()
    print('--------------------')
    print('Beginning translation...')
    translator = GoogleTranslator(source=source_lang, target=target_lang)
    translated_chunks = [translator.translate(chunk) for chunk in chunks]
    print(f'Translation complete, time taken = {(datetime.now()-t0).total_seconds():.2f}s')
    print('--------------------')
    return translated_chunks


def convert_to_df(source_text, translated_text):
    """Convert Dutch and English texts into a pandas DataFrame."""
    source_sentences = re.split(r'(?<=[.!?])\s+', source_text.strip())
    translated_sentences = re.split(r'(?<=[.!?])\s+', translated_text.strip())

    max_len = max(len(source_sentences), len(translated_sentences))
    source_sentences += [""] * (max_len - len(source_sentences))
    translated_sentences += [""] * (max_len - len(translated_sentences))

    df = pd.DataFrame({"Dutch (Original)": source_sentences, "English (Translation)": translated_sentences})
    return df


def delete_audio_file(filename_base):
    """Delete the audio file if requested."""
    audio_filename = f"{filename_base}.mp3"
    if os.path.exists(audio_filename):
        try:
            os.remove(audio_filename)
            print(f"Deleted audio file: {audio_filename}")
        except Exception as e:
            print(f'Failed to delete {audio_filename}: {e}')
    else:
        print(f"Audio file not found: {audio_filename}")


# === MAIN WORKFLOW ===
if __name__ == "__main__":
    # === ARGUMENT PARSER ===
    parser = argparse.ArgumentParser(
        description="YouTube Video Transcriber & Translator")
    parser.add_argument(
        "--url",
        default="https://www.youtube.com/watch?v=JKjeLNCnkcM",
        help="Provide a YouTube url")
    parser.add_argument(
        "--delete-mp3",
        action="store_true",
        help="Delete the .mp3 file after transcription.")
    parser.add_argument(
        "--save-csv",
        action="store_true",
        help="Save the Dutch-English transcription as a CSV file.")
    args = parser.parse_args()

    YOUTUBE_URL  = args.url

    check_ffmpeg()

    # Step 1: Download Audio
    download_audio(YOUTUBE_URL, FILENAME_BASE)
    get_file_size(f"{FILENAME_BASE}.mp3")

    # Step 2: Transcribe Audio
    transcription = transcribe_audio(
        FILENAME_BASE,
        SOURCE_LANG,
        MODEL_SIZE)
    save_text(transcription, "transcription.txt")

    print(f"Transcription (first 100 words): {' '.join(transcription.split()[:100])}...")

    # Step 3: Split text into chunks (max 5000 characters per chunk)
    text_chunks = split_text(transcription)

    # Step 4: Translate each chunk
    translated_chunks = translate_chunks(text_chunks, SOURCE_LANG, TARGET_LANG)
    translated_text = "\n\n***\n\n".join(translated_chunks)

    # Step 5: Save translated text
    save_text(translated_text, "translated.txt")
    print(f"Translation (first 100 words): {' '.join(translated_text.split()[:100])}...")

    # Step 6: Convert to DataFrame and Save CSV (optional)
    if args.save_csv:
        df = convert_to_df(transcription, translated_text)
        df.to_csv("dutch_english_translation.csv", index=False)
        print("Saved Dutch-English translation as CSV: dutch_english_translation.csv")

    # Step 7: Delete audio file if requested
    if args.delete_mp3:
        delete_audio_file(FILENAME_BASE)
