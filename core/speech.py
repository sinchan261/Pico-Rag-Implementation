# import speech_recognition as sr
# import pyttsx3

# class SpeechEngine:
#     def __init__(self):
#         self.recognizer = sr.Recognizer()
#         self.recognizer.energy_threshold = 4000
#         self.recognizer.pause_threshold = 1.0

#     def listen(self) -> str:
#         for attempt in range(3):  # Try 3 times
#             with sr.Microphone() as source:
#                 print("\n🔴 Listening...", end='', flush=True)
#                 try:
#                     self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
#                     audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=8)
#                     print("\r🟢 Processing...", end='', flush=True)
#                     text = self.recognizer.recognize_google(audio).lower()
#                     print(f"\rYou: {text}")
#                     return text
#                 except sr.UnknownValueError:
#                     print("\r🔴 Didn't catch that", end='', flush=True)
#                     self.speak("Could you repeat that?")
#                 except Exception as e:
#                     print(f"\rError: {e}", end='', flush=True)
#         return ""

#     def speak(self, text: str):
#         try:
#             engine = pyttsx3.init()
#             engine.setProperty("rate", 180)
#             engine.setProperty("volume", 1.0)
#             engine.say(text)
#             engine.runAndWait()
#             engine.stop()
#         except Exception as e:
#             print(f"Speech error: {e}")



# # core/speech.py
# import asyncio
# import tempfile
# import os
# import edge_tts
# import speech_recognition as sr
# import pygame


# class SpeechEngine:
#     """
#     Handles both STT (speech-to-text with Google) and 
#     TTS (Edge-TTS) with pygame for playback.
#     """

#     VOICE_MAP = {
#         "en": "en-US-JennyNeural",     # English
#         "hi": "hi-IN-SwaraNeural",     # Hindi
#         "bn": "bn-IN-TanishaaNeural",  # Bengali
#     }

#     def __init__(self):
#         self.recognizer = sr.Recognizer()
#         self.recognizer.energy_threshold = 4000
#         self.recognizer.pause_threshold = 1.0
#         pygame.mixer.init()  # Initialize pygame mixer

#     # ------------------- STT -------------------
#     def listen(self) -> str:
#         for attempt in range(3):  # Retry max 3 times
#             with sr.Microphone() as source:
#                 print("\n🔴 Listening...", end="", flush=True)
#                 try:
#                     self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
#                     audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=8)
#                     print("\r🟢 Processing...", end="", flush=True)
#                     text = self.recognizer.recognize_google(audio).lower()
#                     print(f"\rYou: {text}")
#                     return text
#                 except sr.UnknownValueError:
#                     print("\r🔴 Didn't catch that", end="", flush=True)
#                     self.speak("Could you repeat that?")
#                 except Exception as e:
#                     print(f"\rError: {e}", end="", flush=True)
#         return ""

#     # ------------------- TTS -------------------
#     async def _text_to_file(self, text: str, voice: str, out_path: str):
#         """Stream TTS into a file completely before playback."""
#         communicate = edge_tts.Communicate(text, voice=voice)
#         async for chunk in communicate.stream():
#             if chunk["type"] == "audio":
#                 with open(out_path, "ab") as f:
#                     f.write(chunk["data"])

#     def _play_audio(self, file_path: str):
#         """Play audio file using pygame mixer (blocking until finished)."""
#         try:
#             pygame.mixer.music.load(file_path)
#             pygame.mixer.music.play()
#             while pygame.mixer.music.get_busy():
#                 pygame.time.Clock().tick(10)
#         except Exception as e:
#             print(f"⚠️ Error playing audio: {e}")

#     def speak(self, text: str, lang_code: str = "en"):
#         """Speak text by generating full audio first, then playing it."""
#         voice = self.VOICE_MAP.get(lang_code[:2], self.VOICE_MAP["en"])
#         tmp = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
#         tmp_path = tmp.name
#         tmp.close()

#         async def runner():
#             # Generate audio fully into file
#             await self._text_to_file(text, voice, tmp_path)
#             # Play the audio after file is ready
#             self._play_audio(tmp_path)
#             try:
#                 os.remove(tmp_path)
#             except Exception:
#                 pass

#         try:
#             asyncio.run(runner())
#         except RuntimeError:
#             # If already inside an event loop
#             loop = asyncio.new_event_loop()
#             asyncio.set_event_loop(loop)
#             loop.run_until_complete(runner())
#             loop.close()




# # core/speech.py
# import asyncio
# import tempfile
# import os
# import edge_tts
# import speech_recognition as sr
# import pygame
# import threading
# import queue


# class SpeechEngine:
#     """
#     Handles both STT (speech-to-text with Google) and 
#     TTS (Edge-TTS) with pygame for playback (streaming).
#     """

#     VOICE_MAP = {
#         "en": "en-US-JennyNeural",     # English
#         "hi": "hi-IN-SwaraNeural",     # Hindi
#         "bn": "bn-IN-TanishaaNeural",  # Bengali
#     }

#     def __init__(self):
#         self.recognizer = sr.Recognizer()
#         self.recognizer.energy_threshold = 4000
#         self.recognizer.pause_threshold = 1.0

#         pygame.mixer.init()
#         self.audio_queue = queue.Queue()
#         self.play_thread = threading.Thread(target=self._player_loop, daemon=True)
#         self.play_thread.start()

#     # ------------------- STT -------------------
#     def listen(self) -> str:
#         for attempt in range(3):  # Retry max 3 times
#             with sr.Microphone() as source:
#                 print("\n🔴 Listening...", end="", flush=True)
#                 try:
#                     self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
#                     audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=8)
#                     print("\r🟢 Processing...", end="", flush=True)
#                     text = self.recognizer.recognize_google(audio).lower()
#                     print(f"\rYou: {text}")
#                     return text
#                 except sr.UnknownValueError:
#                     print("\r🔴 Didn't catch that", end="", flush=True)
#                     self.speak("Could you repeat that?")
#                 except Exception as e:
#                     print(f"\rError: {e}", end="", flush=True)
#         return ""

#     # ------------------- TTS -------------------
#     async def _text_to_file(self, text: str, voice: str, out_path: str):
#         """Generate audio chunks for text and write to file."""
#         communicate = edge_tts.Communicate(text, voice=voice)
#         async for chunk in communicate.stream():
#             if chunk["type"] == "audio":
#                 with open(out_path, "ab") as f:
#                     f.write(chunk["data"])

#     def _player_loop(self):
#         """Continuously play queued audio files in order."""
#         while True:
#             file_path = self.audio_queue.get()
#             if file_path is None:
#                 break
#             try:
#                 pygame.mixer.music.load(file_path)
#                 pygame.mixer.music.play()
#                 while pygame.mixer.music.get_busy():
#                     pygame.time.Clock().tick(10)
#             except Exception as e:
#                 print(f"⚠️ Error playing audio: {e}")
#             finally:
#                 try:
#                     os.remove(file_path)
#                 except Exception:
#                     pass

#     def speak(self, text: str, lang_code: str = "en"):
#         """
#         Start speaking as soon as sentences are ready.
#         Splits text into smaller chunks for streaming speech.
#         """
#         voice = self.VOICE_MAP.get(lang_code[:2], self.VOICE_MAP["en"])

#         async def runner():
#             buffer = ""
#             for char in text:
#                 buffer += char
#                 if char in [".", "!", "?", "\n"]:  # sentence boundary
#                     tmp = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
#                     tmp_path = tmp.name
#                     tmp.close()
#                     await self._text_to_file(buffer.strip(), voice, tmp_path)
#                     self.audio_queue.put(tmp_path)
#                     buffer = ""

#             # leftover text (if last sentence had no punctuation)
#             if buffer.strip():
#                 tmp = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
#                 tmp_path = tmp.name
#                 tmp.close()
#                 await self._text_to_file(buffer.strip(), voice, tmp_path)
#                 self.audio_queue.put(tmp_path)

#         try:
#             asyncio.run(runner())
#         except RuntimeError:
#             loop = asyncio.new_event_loop()
#             asyncio.set_event_loop(loop)
#             loop.run_until_complete(runner())
#             loop.close()





# # core/speech.py
# import asyncio
# import tempfile
# import os
# import edge_tts
# import speech_recognition as sr
# import pygame
# import threading
# import queue

# class SpeechEngine:
#     """
#     Handles both STT (speech-to-text with Google) and 
#     TTS (Edge-TTS) with pygame for playback (streaming).
#     """

#     VOICE_MAP = {
#         "en": "en-US-JennyNeural",     # English
#         "hi": "hi-IN-SwaraNeural",     # Hindi
#         "bn": "bn-IN-TanishaaNeural",  # Bengali
#     }

#     def __init__(self):
#         self.recognizer = sr.Recognizer()
#         self.recognizer.energy_threshold = 4000
#         self.recognizer.pause_threshold = 1.0

#         pygame.mixer.init()
#         self.audio_queue = queue.Queue()
#         self.play_thread = threading.Thread(target=self._player_loop, daemon=True)
#         self.play_thread.start()

#     # ------------------- STT -------------------
#     def listen(self) -> str:
#         for attempt in range(3):  # Retry max 3 times
#             with sr.Microphone() as source:
#                 print("\n🔴 Listening...", end="", flush=True)
#                 try:
#                     self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
#                     audio = self.recognizer.listen(source, timeout=10, phrase_time_limit=20)
#                     print("\r🟢 Processing...", end="", flush=True)
#                     text = self.recognizer.recognize_google(audio).lower()
#                     print(f"\rYou: {text}")
#                     return text
#                 except sr.UnknownValueError:
#                     print("\r🔴 Didn't catch that", end="", flush=True)
#                     self.speak("Could you repeat that?")
#                 except Exception as e:
#                     print(f"\rError: {e}", end="", flush=True)
#         return ""

#     # ------------------- TTS -------------------
#     async def _text_to_file(self, text: str, voice: str, out_path: str):
#         """Generate audio chunks for text and write to file."""
#         communicate = edge_tts.Communicate(text, voice=voice)
#         async for chunk in communicate.stream():
#             if chunk["type"] == "audio":
#                 with open(out_path, "ab") as f:
#                     f.write(chunk["data"])

#     def _player_loop(self):
#         """Continuously play queued audio files in order."""
#         while True:
#             file_path = self.audio_queue.get()
#             if file_path is None:
#                 break
#             try:
#                 pygame.mixer.music.load(file_path)
#                 pygame.mixer.music.play()
#                 while pygame.mixer.music.get_busy():
#                     pygame.time.Clock().tick(10)
#             except Exception as e:
#                 print(f"⚠️ Error playing audio: {e}")
#             finally:
#                 try:
#                     os.remove(file_path)
#                 except Exception:
#                     pass

#     def speak(self, text: str, lang_code: str = "en"):
#         """
#         Start speaking as soon as sentences are ready.
#         Splits text into smaller chunks for streaming speech.
#         """
#         voice = self.VOICE_MAP.get(lang_code[:2], self.VOICE_MAP["en"])

#         async def runner():
#             buffer = ""
#             for char in text:
#                 buffer += char
#                 if char in [".", "!", "?", "\n"]:  # sentence boundary
#                     tmp = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
#                     tmp_path = tmp.name
#                     tmp.close()
#                     await self._text_to_file(buffer.strip(), voice, tmp_path)
#                     self.audio_queue.put(tmp_path)
#                     buffer = ""

#             # leftover text (if last sentence had no punctuation)
#             if buffer.strip():
#                 tmp = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
#                 tmp_path = tmp.name
#                 tmp.close()
#                 await self._text_to_file(buffer.strip(), voice, tmp_path)
#                 self.audio_queue.put(tmp_path)

#         try:
#             asyncio.run(runner())
#         except RuntimeError:
#             loop = asyncio.new_event_loop()
#             asyncio.set_event_loop(loop)
#             loop.run_until_complete(runner())    // unlock new
#             loop.close()




# core/speech.py
import asyncio
import tempfile
import os
import edge_tts
import speech_recognition as sr
import pygame
import threading
import queue


class SpeechEngine:
    """
    Handles STT (speech-to-text with Google) and
    TTS (Edge-TTS) with pygame for playback (streaming).
    - Adds is_speaking flag for multitasking assistant.
    - Safe listen() retries and returns None on failure.
    """

    VOICE_MAP = {
        "en": "en-US-JennyNeural",     # English
        "hi": "hi-IN-SwaraNeural",     # Hindi
        "bn": "bn-IN-TanishaaNeural",  # Bengali
    }

    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.recognizer.energy_threshold = 4000
        self.recognizer.pause_threshold = 1.0

        pygame.mixer.init()
        self.audio_queue = queue.Queue()
        self.play_thread = threading.Thread(target=self._player_loop, daemon=True)
        self.play_thread.start()

        self.is_speaking = False
        self._speak_lock = threading.Lock()
        self._didnt_catch = 0

    # ------------------- STT -------------------
    def listen(self) -> str | None:
        """
        Captures microphone input and transcribes it.
        Returns lowercase text, or None if nothing was understood.
        """
        for attempt in range(3):  # Retry max 3 times
            with sr.Microphone() as source:
                print("\n🔴 Listening...", end="", flush=True)
                try:
                    self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                    audio = self.recognizer.listen(
                        source, timeout=20, phrase_time_limit=30
                    )
                    print("\r🟢 Processing...", end="", flush=True)
                    text = self.recognizer.recognize_google(audio).lower()
                    print(f"\rYou: {text}")
                    self._didnt_catch = 0
                    return text
                except sr.UnknownValueError:
                    print("\r🔴 Didn't catch that", end="", flush=True)
                    self._didnt_catch += 1
                    return None
                except Exception as e:
                    print(f"\r⚠️ Error: {e}", end="", flush=True)
                    return None
        return None

    def didnt_catch_count(self) -> int:
        return self._didnt_catch

    def reset_catch_counter(self):
        self._didnt_catch = 0

    # ------------------- TTS -------------------
    async def _text_to_file(self, text: str, voice: str, out_path: str):
        """Generate audio chunks for text and write to file."""
        communicate = edge_tts.Communicate(text, voice=voice)
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                with open(out_path, "ab") as f:
                    f.write(chunk["data"])

    def _player_loop(self):
        """Continuously play queued audio files in order."""
        while True:
            file_path = self.audio_queue.get()
            if file_path is None:
                break
            try:
                self.is_speaking = True
                pygame.mixer.music.load(file_path)
                pygame.mixer.music.play()
                while pygame.mixer.music.get_busy():
                    pygame.time.Clock().tick(10)
            except Exception as e:
                print(f"⚠️ Error playing audio: {e}")
            finally:
                self.is_speaking = False
                try:
                    os.remove(file_path)
                except Exception:
                    pass

    def speak(self, text: str, lang_code: str = "en"):
        """
        Convert text to speech and queue it for playback.
        Splits into smaller chunks per sentence.
        """
        if not text:
            return

        voice = self.VOICE_MAP.get(lang_code[:2], self.VOICE_MAP["en"])

        async def runner():
            buffer = ""
            for char in text:
                buffer += char
                if char in [".", "!", "?", "\n"]:  # sentence boundary
                    tmp = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
                    tmp_path = tmp.name
                    tmp.close()
                    await self._text_to_file(buffer.strip(), voice, tmp_path)
                    self.audio_queue.put(tmp_path)
                    buffer = ""

            # leftover text (if last sentence had no punctuation)
            if buffer.strip():
                tmp = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
                tmp_path = tmp.name
                tmp.close()
                await self._text_to_file(buffer.strip(), voice, tmp_path)
                self.audio_queue.put(tmp_path)

        with self._speak_lock:
            try:
                asyncio.run(runner())
            except RuntimeError:
                # If event loop is already running (e.g., in Jupyter)
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(runner())
                loop.close()
