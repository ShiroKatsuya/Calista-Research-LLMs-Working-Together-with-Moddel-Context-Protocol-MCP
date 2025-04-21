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
import queue

# # Ensure NLTK's Punkt tokenizer is downloaded
# nltk.download('punkt_tab')

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


def voice(teks, chunk_length_ms=5500):
    print("isis teks", teks)

    tts = SileroTTS(
        model_id='v3_en',
        language='en',
        speaker='en_67',
        sample_rate=48000,
        device='cuda',
        put_accent=True,
        put_yo=True,
        num_threads=8
    )

    print("Memproses Text-to-Speech dengan SileroTTS.")

    # Check if user wants to exit
    if teks.lower() == "exit":
        print("Keluar dari program.")
        return

    # Clean text from special characters
    cleaned_text = re.sub(r"\*(.*?)\*", r"\1", teks)

    # Translate text to English
    translated = GoogleTranslator(source='auto', target='en').translate(cleaned_text)
    print(f"Teks yang akan diubah menjadi suara: {translated}")

    # Split translated text into sentences
    sentences = nltk.sent_tokenize(translated)
    print(f"Jumlah kalimat: {len(sentences)}")
    
    if not sentences:
        print("Tidak ada kalimat yang dihasilkan.")
        return
        
    # Setup subtitle display window
    root = tk.Tk()
    root.title("Subtitle")
    root.attributes("-topmost", True)
    root.configure(bg='black')
    root.overrideredirect(True)

    window_width = 800
    window_height = 200
    x_pos = 100
    y_pos = 700

    root.geometry(f"{window_width}x{window_height}+{x_pos}+{y_pos}")

    label = tk.Label(
        root,
        text="Mempersiapkan audio...",
        fg="white",
        bg="black",
        font=("Comic Sans MS", 16),
        wraplength=window_width-50,
        justify="center"
    )
    label.pack(expand=True)
    
    # Create a queue for segments
    segment_queue = queue.Queue()
    processing_complete = threading.Event()
    playing_complete = threading.Event()
    
    # Flag to indicate if processing should continue
    running = True
    
    # For managing the current segment being played
    current_segment_idx = -1
    total_segments = len(sentences)
    
    # Lock for thread synchronization
    lock = threading.Lock()
    
    # Function to process segments in parallel
    def process_segments():
        for idx, sentence in enumerate(sentences):
            if not running:
                break
                
            try:
                # Generate English audio
                sentence_audio = f"temp_sentence_{idx}.wav"
                tts.tts(sentence, sentence_audio)
                
                # Calculate duration
                audio = AudioSegment.from_wav(sentence_audio)
                duration_ms = len(audio)
                
                # Translate to Indonesian for subtitle
                indo_segment = GoogleTranslator(source='en', target='id').translate(sentence)
                subtitle_text = f"[{idx+1}/{total_segments}]\n{sentence}\n{indo_segment}"
                
                # Add to queue
                segment_info = {
                    'index': idx,
                    'audio_file': sentence_audio,
                    'subtitle': subtitle_text,
                    'duration': duration_ms
                }
                
                print(f"Segment {idx+1} processed: '{sentence}' duration {duration_ms} ms")
                segment_queue.put(segment_info)
                
            except Exception as e:
                print(f"Error processing segment {idx}: {e}")
        
        processing_complete.set()
    
    # Function to play segments as they become available
    def play_segments():
        nonlocal current_segment_idx
        
        while running:
            try:
                # Get next segment or wait for 100ms
                try:
                    segment = segment_queue.get(timeout=0.1)
                except queue.Empty:
                    # If processing is done and queue is empty, we're finished
                    if processing_complete.is_set() and segment_queue.empty():
                        break
                    continue
                
                with lock:
                    current_segment_idx = segment['index']
                
                # Update subtitle
                label.config(text=segment['subtitle'])
                
                # Play audio
                try:
                    subprocess.run(
                        ["ffplay", "-nodisp", "-autoexit", "-sync", "ext", segment['audio_file']],
                        check=True,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL
                    )
                except Exception as e:
                    print(f"Error playing audio segment {segment['index']}: {e}")
                
                # Cleanup audio file
                try:
                    os.remove(segment['audio_file'])
                except Exception as e:
                    print(f"Error removing audio file: {e}")
                
                # Mark task as done
                segment_queue.task_done()
                
            except Exception as e:
                print(f"Error in play_segments: {e}")
        
        playing_complete.set()
    
    # Function to handle window closing
    def on_closing():
        nonlocal running
        running = False
        root.destroy()
    
    root.protocol("WM_DELETE_WINDOW", on_closing)
    
    # Start processing thread
    processing_thread = threading.Thread(target=process_segments)
    processing_thread.daemon = True
    processing_thread.start()
    
    # Start playing thread
    playing_thread = threading.Thread(target=play_segments)
    playing_thread.daemon = True
    playing_thread.start()
    
    # Update progress indicator
    def update_status():
        if not running:
            return
            
        # If all segments have finished playing
        if playing_complete.is_set():
            root.destroy()
            return
            
        # Schedule next update
        root.after(100, update_status)
    
    # Start the status updates
    update_status()
    
    # Start the main loop
    root.mainloop()
    
    # Cleanup when window closes
    running = False
    
    # Wait for threads to finish
    processing_thread.join(timeout=1)
    playing_thread.join(timeout=1)
    
    print("Voice processing completed.")