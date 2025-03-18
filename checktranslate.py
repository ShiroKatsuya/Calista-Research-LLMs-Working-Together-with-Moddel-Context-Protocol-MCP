from deep_translator import GoogleTranslator

transcription = "I am a student"
translate = GoogleTranslator(source='auto', target='id').translate(transcription)
print(f"Transkripsi: {translate}")