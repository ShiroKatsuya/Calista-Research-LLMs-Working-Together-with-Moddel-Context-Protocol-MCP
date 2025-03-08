import nltk
from gtts import gTTS
import os
import time
import subprocess
from deep_translator import GoogleTranslator
import sys
sys.path.insert(0, 'silero_tts')
from silero_tts import SileroTTS
from concurrent.futures import ThreadPoolExecutor
import tkinter as tk
from pydub import AudioSegment
import speech_recognition as sr
import threading
import re

# Ensure NLTK's Punkt tokenizer is downloaded
nltk.download('punkt_tab')

def save_audio(teks):

    # Clean filename by removing invalid characters and whitespace
    filename = "".join(c for c in teks if c.isalnum() or c in (' ', '-', '_'))[:50]  # Limit length
    filename = filename.strip().replace(' ', '_')
    if not filename:  # Fallback if filename is empty after cleaning
        filename = "audio"
    
    tts = SileroTTS(
        model_id='v3_en',
        language='en',
        speaker='en_67',  # Using a clearer speaker
        sample_rate=48000,  # Ensuring sample rate does not exceed 48000
        device='cuda',
        put_accent=True,
        put_yo=True,
        num_threads=8  # Optimized number of threads for better processing
    )
    output_path = f"{filename}.wav"
    tts.tts(teks, output_path)
    return output_path


def voice(teks, chunk_length_ms=5500):  # 'chunk_length_ms' is no longer required
    recognizer = sr.Recognizer()

    print("isis teks", teks)

    tts = SileroTTS(
        model_id='v3_en',
        language='en',
        speaker='en_67',  # Using a clearer speaker
        sample_rate=48000,  # Ensuring sample rate does not exceed 48000
        device='cuda',
        put_accent=True,
        put_yo=True,
        num_threads=8  # Optimized number of threads for better processing
    )

    """
    Fungsi untuk mengubah teks menjadi suara, memainkannya, dan menghasilkan subtitle teks.

    Parameters:
    teks (str): Teks yang akan diubah menjadi suara.
    chunk_length_ms (int): Panjang setiap segmen audio dalam milidetik. (Unused now)
    """
    print("Memproses Text-to-Speech dengan SileroTTS.")

    # Periksa jika pengguna ingin keluar
    if teks.lower() == "exit":
        print("Keluar dari program.")
        return

    # Bersihkan teks dari karakter khusus
    cleaned_text = re.sub(r"\*(.*?)\*", r"\1", teks)

    # Terjemahkan teks ke dalam Bahasa Inggris
    translated = GoogleTranslator(source='auto', target='en').translate(cleaned_text)
    print(f"Teks yang akan diubah menjadi suara: {translated}")

    # Split translated text into sentences
    sentences = nltk.sent_tokenize(translated)
    print(f"Jumlah kalimat: {len(sentences)}")

    # Generate audio for each sentence and calculate durations
    sentence_audio_files = []
    sentence_durations = []
    for idx, sentence in enumerate(sentences):
        sentence_audio = f"temp_sentence_{idx}.wav"
        try:
            tts.tts(sentence, sentence_audio)
            sentence_audio_files.append(sentence_audio)
            audio = AudioSegment.from_wav(sentence_audio)
            sentence_durations.append(len(audio))  # Duration in ms
            print(f"Kalimat {idx+1}: '{sentence}' durasi {len(audio)} ms")
        except Exception as e:
            print(f"Error saat memproses TTS untuk kalimat {idx}: {e}")
            continue

    if not sentence_audio_files:
        print("Tidak ada audio yang dihasilkan.")
        return

    # Combine all sentence audio files into one audio file
    combined = AudioSegment.empty()
    for sentence_audio in sentence_audio_files:
        try:
            audio = AudioSegment.from_wav(sentence_audio)
            combined += audio
            os.remove(sentence_audio)  # Cleanup individual sentence audio files
        except Exception as e:
            print(f"Error saat menggabungkan audio '{sentence_audio}': {e}")

    audio_file = "output_ai.wav"
    try:
        combined.export(audio_file, format="wav")
        print(f"Audio gabungan disimpan sebagai {audio_file}")
    except Exception as e:
        print(f"Error saat menyimpan audio gabungan: {e}")
        return

    # Create subtitles list with timing
    subtitles = []
    current_time = 0
    for idx, sentence in enumerate(sentences):
        indo_segment = GoogleTranslator(source='en', target='id').translate(sentence)
        subtitle_text = f"[{idx+1}/{len(sentences)}]\n{sentence}\n{indo_segment}"
        subtitle = {
            'start': current_time,
            'end': current_time + sentence_durations[idx],
            'text': subtitle_text
        }
        subtitles.append(subtitle)
        current_time += sentence_durations[idx]

    # Setup subtitle display window
    root = tk.Tk()
    root.title("Subtitle")
    root.attributes("-topmost", True)
    root.configure(bg='black')
    root.overrideredirect(True)

    window_width = 800
    window_height = 200  # Increased height to better accommodate longer subtitles
    x_pos = 100
    y_pos = 700

    root.geometry(f"{window_width}x{window_height}+{x_pos}+{y_pos}")

    label = tk.Label(
        root,
        text="",
        fg="white",
        bg="black",
        font=("Comic Sans MS", 16),
        wraplength=window_width-50,
        justify="center"
    )
    label.pack(expand=True)

    # Event untuk sinkronisasi
    audio_started = threading.Event()

    # Putar audio dalam thread terpisah
    def play_audio():
        try:
            audio_started.set()  # Tandai bahwa audio mulai diputar
            subprocess.run(
                ["ffplay", "-nodisp", "-autoexit", "-sync", "ext", audio_file],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
        except Exception as e:
            print(f"Error saat memutar audio: {e}")
            root.destroy()

    audio_thread = threading.Thread(target=play_audio)

    # Fungsi untuk update subtitle berdasarkan timing per sentence
    def update_subtitle():
        nonlocal start_time
        current_time_ms = (time.time() - start_time) * 1000  # Convert to ms
        for subtitle in subtitles:
            if subtitle['start'] <= current_time_ms < subtitle['end']:
                if label.cget("text") != subtitle['text']:
                    label.config(text=subtitle['text'])
                break
        else:
            label.config(text="")  # Clear subtitle if no match

        # Schedule the next check
        if current_time_ms < subtitles[-1]['end']:
            root.after(100, update_subtitle)  # Check every 100 ms
        else:
            root.destroy()

    # Start audio and subtitles
    audio_thread.start()
    audio_started.wait()  # Tunggu hingga audio benar-benar mulai
    start_time = time.time()
    update_subtitle()  # Start subtitle immediately

    root.mainloop()

    # Cleanup
    audio_thread.join()
    try:
        os.remove(audio_file)
        print(f"File audio '{audio_file}' telah dihapus.")
    except Exception as e:
        print(f"Error menghapus file audio: {e}")